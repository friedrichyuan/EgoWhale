#!/usr/bin/env python3
"""Run only EgoWhale's ego2robot-synchronized WiLoR node."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--python', type=Path)
    parser.add_argument('--camera-source', choices=('auto', 'egodex', 'provided', 'da3'), default='auto')
    parser.add_argument('--intrinsics-h5', type=Path)
    parser.add_argument('--intrinsics-json', type=Path)
    args = parser.parse_args()
    from egowhale.wilor.step import WiLoR
    node = WiLoR(camera_mode=args.camera_source, calibration=args.intrinsics_h5,
                 python=args.python, pipeline_config=args.config, intrinsics_json=args.intrinsics_json)
    try:
        node.run(args.video.resolve(), args.output.resolve())
    finally:
        node.release()
    print(f"完成：{args.output.resolve() / 'action/wilor_ego2robot/optimized_hand_points.mp4'}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
