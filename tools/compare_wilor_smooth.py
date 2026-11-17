"""Render one set of WiLoR tracks beside its smooth candidate, even if rejected."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from egowhale.wilor.synced import SYNC_ROOT

sys.path.insert(0, str(SYNC_ROOT))
from ego2robot.processors.hand_processor import HandProcessor
from ego2robot.utils.optimization_quality import assess_smooth_result


def draw_points(frame, hand_data, index):
    frame = frame.copy()
    for side, color in (('left', (220, 0, 220)), ('right', (0, 220, 0))):
        data = hand_data.get(side)
        if data is None or not data['hand_detected'][index]:
            continue
        for point in data['kpts_2d'][index]:
            if np.isfinite(point).all():
                x, y = map(int, point)
                if 0 < x < frame.shape[1] and 0 < y < frame.shape[0]:
                    cv2.circle(frame, (x, y), 2, color, -1)
    return frame


def export_comparison(video, work, output):
    video, work, output = map(lambda p: Path(p).resolve(), (video, work, output))
    ffmpeg = shutil.which('ffmpeg') or str(Path(sys.executable).parent / 'ffmpeg')
    if not Path(ffmpeg).is_file():
        raise FileNotFoundError('ffmpeg is missing from PATH and the active Python environment')
    info = json.loads((work / 'input.json').read_text())
    status_file = work / 'optimization_status.json'
    status = json.loads(status_file.read_text()) if status_file.is_file() else {}
    fallback_file = work / 'hmp_fallback.json'
    fallback = json.loads(fallback_file.read_text()) if fallback_file.is_file() else {}
    completed_file = work / 'smooth_completed.json'
    completed = json.loads(completed_file.read_text()) if completed_file.is_file() else {}
    source = (status.get('quality', {}).get('source_file') or fallback.get('smooth_result')
              or completed.get('smooth_result'))
    if not source:
        raise RuntimeError('没有记录本次完成的 smooth 结果；不能把 WiLoR 回退视频冒充 smooth 对比')
    cfg = OmegaConf.load(work / 'resolved_config.yaml')
    hands = work / 'hand_processor'
    baseline = {}
    for side in ('left', 'right'):
        with np.load(hands / f'hand_data_raw_{side}.npz', allow_pickle=True) as data:
            baseline[side] = dict(data)
    width, height = info['size']
    count, fps = info['frames'], info['fps']
    camera = json.loads((work / 'cam_intrinsics.json').read_text())['left']
    processor = object.__new__(HandProcessor)
    processor.cfg = cfg
    seq_name = Path(source).name.rsplit('_', 3)[0]
    candidate = processor._postprocess_dynhamr(
        cfg.dynhamr_root, str(Path(cfg.dynhamr_root) / 'test'), seq_name, hands,
        camera['fx'], camera['fy'], camera['cx'], camera['cy'], count,
        cfg.dynhamr_gap_fill, cfg.dynhamr_z_min, cfg.dynhamr_z_max,
        result_path=source, write_output=False)
    quality = assess_smooth_result(baseline, candidate, count, width, height, cfg)
    output.mkdir(parents=True, exist_ok=True)
    state = 'ACCEPTED' if quality['accepted'] else 'REJECTED - diagnostic only'
    filenames = ('wilor_only', 'smooth_candidate', 'comparison')
    capture = cv2.VideoCapture(str(video))
    with tempfile.TemporaryDirectory(prefix='.render-', dir=output) as temporary:
        writers = {}
        try:
            for name in filenames:
                size = (width * 2, height + 40) if name == 'comparison' else (width, height)
                writer = cv2.VideoWriter(str(Path(temporary) / f'{name}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
                if not writer.isOpened():
                    raise RuntimeError(f'Cannot open {name} video writer')
                writers[name] = writer
            for i in range(count):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f'视频提前结束：{i}/{count}')
                frame = cv2.resize(frame, (width, height))
                before = draw_points(frame, baseline, i)
                after = draw_points(frame, candidate, i)
                joined = np.zeros((height + 40, width * 2, 3), dtype=np.uint8)
                joined[40:, :width], joined[40:, width:] = before, after
                cv2.putText(joined, 'WiLoR (same detections, no optimization)', (8, 26), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                cv2.putText(joined, f'Smooth: {state}', (width + 8, 26), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                for name, view in zip(filenames, (before, after, joined)):
                    writers[name].write(view)
                if (i + 1) % 30 == 0 or i + 1 == count:
                    print(f'[对比视频] {i + 1}/{count}', flush=True)
        finally:
            capture.release()
            for writer in writers.values():
                writer.release()
        for name in filenames:
            subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                            '-i', str(Path(temporary) / f'{name}.mp4'), '-an', '-c:v', 'libx264',
                            '-pix_fmt', 'yuv420p', '-crf', '18', '-movflags', '+faststart',
                            str(output / f'{name}.mp4')], check=True)
    selected = work / 'optimized_hand_points.mp4'
    if selected.is_file():
        shutil.copy2(selected, output / 'selected_result.mp4')
    report = {'video': str(video), 'frames': count, 'smooth_source': str(source),
              'optimization': status, 'quality': quality,
              'comparison_baseline': 'same WiLoR detections/tracking/filters before root/smooth',
              'counts': {name: {side: int(data[side]['hand_detected'].sum()) if side in data else 0
                                for side in ('left', 'right')}
                         for name, data in (('wilor', baseline), ('smooth', candidate))}}
    (output / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'对比视频：{output / "comparison.mp4"}', flush=True)
    print(f'smooth 验收：{state}；详情：{output / "comparison.json"}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--work', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    export_comparison(args.video, args.work, args.output)


if __name__ == '__main__':
    main()
