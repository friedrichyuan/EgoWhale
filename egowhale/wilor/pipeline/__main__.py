"""Run WiLoR internal stages locally or through its existing Ray actor."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stages', nargs='+', choices=('camera', 'raw', 'masks', 'filtered', 'refined', 'final'))
    parser.add_argument('--executor', choices=('local', 'ray'), default='local')
    args = parser.parse_args(argv)
    if args.executor == 'ray':
        if args.stages:
            parser.error('Ray actor runs configured pipeline; use local for selected stages')
        import ray

        from egowhale.wilor.actor import WiLoRActor
        ray.init(ignore_reinit_error=True, include_dashboard=False, object_store_memory=256 * 1024 * 1024)
        actor = ray.remote(num_gpus=1)(WiLoRActor).remote({'pipeline_config': str(args.config.resolve())})
        try:
            result = ray.get(actor.run.remote('wilor', args.video.resolve(), args.out.resolve()))
            if not result.ran and result.error:
                raise RuntimeError(result.error)
        finally:
            try:
                ray.get(actor.close.remote(), timeout=10)
            finally:
                ray.kill(actor)
                ray.shutdown()
    else:
        from .runner import load_options, run_pipeline
        run_pipeline(args.video, args.out, load_options(args.config), stages=tuple(args.stages) if args.stages else None)
    return 0


if __name__ == '__main__':
    sys.exit(main())
