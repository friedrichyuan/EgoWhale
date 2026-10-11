"""python -m egowhale.batch <config.yaml> [--name NAME] [--limit N] [--episodes PATH ...] [--out DIR]"""

from __future__ import annotations

import argparse
from pathlib import Path

from egowhale import config as configs
from egowhale.batch.dag import Job
from egowhale.batch.report import watched
from egowhale.batch.resources import pools_from
from egowhale.batch.scheduler import serve
from egowhale.config import ConfigError


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the episode DAG on Ray pools, as a config in cfgs/ describes.")
    parser.add_argument("config", type=Path, help="YAML config, e.g. cfgs/bench8_len1000.yaml")
    parser.add_argument("--name", help="experiment name; results go to <out>/<name>/")
    parser.add_argument("--limit", type=int, help="run only the first N episodes")
    parser.add_argument("--episodes", nargs="+", help="replace the config's episodes: .txt lists, HDF5 files, or dataset directories")
    parser.add_argument("--out", help="replace the config's output root; results go to <out>/<name>/")
    args = parser.parse_args()
    try:
        cfg = configs.load(args.config, name=args.name, limit=args.limit, episodes=args.episodes, out=args.out)
        sources = configs.episodes(cfg)
        if not sources:
            raise ConfigError("no episodes")
        out = Path(cfg["out"])
        out.mkdir(parents=True, exist_ok=True)
        import yaml

        (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        jobs = [Job(src, out / src.parent.name / src.stem, cfg["piece_frames"]) for src in sources]
        pools, share = pools_from(cfg)
        watch = watched(len(jobs))
        for index, job in enumerate(jobs):
            job.index = index
            job.watch = index in watch
        serve(jobs, cfg, pools, share)
    except ConfigError as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
