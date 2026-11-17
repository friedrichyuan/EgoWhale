"""One-video, unknown-intrinsics WiLoR launcher. GT is opened only after inference."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import yaml

from egowhale.media import video_shape
from egowhale.paths import HANDS_WILOR
from egowhale.wilor.pipeline.contracts import read_artifact
from egowhale.wilor.pipeline.runner import DEFAULTS, load_options, model_identity, run_pipeline


def resolve(value):
    p = Path(value).expanduser()
    return (p if p.is_absolute() else ROOT / p).resolve()


def settings(mode, da3_python, wilor_python, samples, batch, sam3_python=None, dynhamr_python=None):
    if samples < 1 or batch < 1:
        raise ValueError('DA3_SAMPLES and BATCH_SIZE must be positive')
    given = yaml.safe_load((ROOT / 'egowhale/wilor/cfgs/wilor.yaml').read_text())['legacy'] if mode == 'full' else {}
    # Unknown intrinsics are an enforced input contract, not an optional calibration mode.
    given['camera'] = {**given.get('camera', {}), 'source': 'da3', 'calibration': None,
                       'poses': None, 'motion': False, 'python': str(resolve(da3_python)), 'samples': samples}
    given['wilor'] = {'python': str(resolve(wilor_python)), 'batch_size': batch}
    if mode == 'full':
        for key in ('mask', 'tracking', 'jump', 'refine', 'depth_bounds', 'gaps'):
            given.setdefault(key, {})['enabled'] = True
        given['refine'].update(run_prior=True, boundary_trim=3)
        if sam3_python:
            given['mask']['python'] = str(resolve(sam3_python))
        if dynhamr_python:
            given['refine']['python'] = str(resolve(dynhamr_python))
    given['publish'] = {'stage': 'final' if mode == 'full' else 'raw'}
    return given


def preflight(options):
    failures = []
    sections = ['camera', 'wilor']
    if options['mask']['enabled']:
        sections += ['mask']
    if options['refine']['enabled']:
        sections += ['refine']
    for section in sections:
        try:
            model_identity(section, options)
        except (ValueError, OSError, subprocess.CalledProcessError) as error:
            failures.append(f'{section}: {error}')
    if failures:
        raise RuntimeError('Resources are not ready:\n  ' + '\n  '.join(failures)
                           + '\n软件环境检查：conda activate egowhale-wilor; python tools/check_wilor_env.py'
                           + '\n相机、WiLoR、SAM3 和 DynHaMR 依赖仍需齐全；HMP 失败时会回退已有手部预测。'
                           + '\n请检查模型资源路径及 Python 依赖。')
    for section, code in [('camera', "from depth_anything_3.api import DepthAnything3"),
                          ('wilor', 'import ultralytics, smplx, pytorch_lightning')]:
        o = options[section]
        env = {**os.environ, 'PYTHONPATH': str(Path(o['repo']) / 'src') + os.pathsep + str(ROOT)}
        command = [o['python'], '-c', code]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f'{section} dependencies could not import:\n{result.stderr[-3000:]}')
    result = subprocess.run([options['wilor']['python'], '-c',
                             'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; '
                             'print("GPU:", torch.cuda.get_device_name(0))'], check=True, text=True)
    return result.returncode


@contextlib.contextmanager
def stage(index, total, name):
    started = time.monotonic()
    stop = threading.Event()

    def log(message):
        print(f'[{datetime.now():%H:%M:%S}] 总进度 {(index - 1) / total:.0%} '
              f'| 阶段 {index}/{total}: {name} | {message}', flush=True)

    def heartbeat():
        while not stop.wait(20):
            log(f'处理中，已运行 {time.monotonic() - started:.0f}s；详见模型日志')

    log('开始')
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield
    except BaseException:
        log('失败；未标记完成')
        raise
    else:
        print(f'[{datetime.now():%H:%M:%S}] 总进度 {index / total:.0%} | {name} 完成 '
              f'({time.monotonic() - started:.1f}s)', flush=True)
    finally:
        stop.set()
        thread.join()


def render_points(video, target, pixels, valid, confidence=None):
    cap = cv2.VideoCapture(str(video))
    count, height, width, fps = video_shape(video)
    if len(pixels) != count or valid.shape != pixels.shape[:-1]:
        cap.release()
        raise ValueError('Video and point timeline/shapes do not match')
    temp = target.with_name(target.stem + '.partial.mp4')
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not cap.isOpened() or not writer.isOpened():
        cap.release()
        writer.release()
        raise RuntimeError(f'Cannot open video input/writer: {target}')
    try:
        for frame in range(count):
            ok, image = cap.read()
            if not ok:
                raise ValueError('Video ended before all annotations')
            mask = valid[frame] & np.isfinite(pixels[frame]).all(-1)
            if confidence is not None:
                mask &= confidence[frame] >= .5
            points = pixels[frame][mask]
            color = (0, 220, 0) if confidence is not None else (0, 0, 255)
            for u, v in points:
                if 0 <= u < width and 0 <= v < height:
                    cv2.circle(image, (round(float(u)), round(float(v))), 5, color, -1, cv2.LINE_AA)
            writer.write(image)
            if frame == 0 or frame + 1 == count or (frame + 1) % max(1, count // 20) == 0:
                print(f'[{target.name}] {frame + 1}/{count} 帧 ({(frame + 1) / count:.1%})', flush=True)
    finally:
        cap.release()
        writer.release()
    temp.replace(target)


def export_prediction(video, path, target):
    kind = json.loads(path.with_suffix('.json').read_text())['kind']
    if kind not in ('raw', 'prediction'):
        raise ValueError('Expected a published prediction or raw hand artifact')
    artifact = read_artifact(path, expected_kind=kind)
    a = artifact.arrays
    if not np.array_equal(a['frame_index'], np.arange(len(a['valid']))):
        raise ValueError('Prediction timeline is not aligned')
    # Display exactly the stored per-frame projected points. No GT calibration enters here.
    mask = np.repeat(a['valid'][..., None], 21, axis=-1) & (a['joints_camera'][..., 2] > 0)
    render_points(video, target, a['keypoints_2d'], mask)


def export_ground_truth(video, annotation, target):
    # Annotation is opened only here, after model inference and prediction export.
    from tools.evaluate_wilor_egodex import load_ground_truth, project

    xyz, confidence, gt_intrinsic = load_ground_truth(annotation)
    # Use the annotation camera ONLY to correctly draw the GT video, never to infer predictions.
    pixels = project(xyz, gt_intrinsic)
    valid = np.isfinite(xyz).all(-1) & (xyz[..., 2] > 0)
    render_points(video, target, pixels, valid, confidence)


def validate_full_prediction(artifact):
    """Accept completed optimization or an explicitly recorded HMP fallback."""
    meta = artifact.meta
    if meta.get('published_stage') != 'final':
        raise ValueError('完整模式必须发布 final 结果，不能使用逐帧 raw 结果')
    options = meta.get('pipeline_options', {})
    for key in ('mask', 'tracking', 'jump', 'refine', 'depth_bounds', 'gaps'):
        if not options.get(key, {}).get('enabled'):
            raise ValueError(f'完整模式缺少已启用的阶段：{key}')
    fallback = (meta.get('optimizer_fallback') and meta.get('fallback_stage') == 'hmp'
                and bool(meta.get('fallback_reason')) and not meta.get('prior_executed')
                and meta.get('optimizer_stage') == 'filtered')
    if not fallback and (not meta.get('optimizer_executed') or not meta.get('prior_executed')
            or meta.get('optimizer_stage') != 'prior'):
        raise ValueError('DynHaMR/HMP 未实际完成，不能导出为完整流程结果；请检查检测是否为空及优化器日志')
    if meta.get('boundary_trim') != 3:
        raise ValueError('完整结果缺少官方边界清理记录')


def execute(video, output, options, mode, annotation):
    stages = [('camera', 'DA3 从 RGB 估计内参'), ('raw', 'WiLoR 逐帧重建')]
    if mode == 'full':
        stages += [('masks', 'SAM3 手部掩码'), ('filtered', '过滤、关联与跳变处理'),
                   ('refined', 'DynHaMR + HMP + 边界清理'), ('final', '短缺帧补全')]
    total = len(stages) + 2 + int(annotation is not None)
    with stage(1, total, '保存原始输入视频'):
        shutil.copy2(video, output / 'input.mp4')
    for index, (key, title) in enumerate(stages, start=2):
        with stage(index, total, title):
            run_pipeline(video, output, options, stages=(key,))
    with stage(len(stages) + 2, total, '导出预测点视频（不连线）'):
        if mode == 'full':
            prediction = read_artifact(output / HANDS_WILOR, expected_kind='prediction')
            validate_full_prediction(prediction)
            if prediction.meta.get('optimizer_fallback'):
                print('HMP 未完成，继续导出 WiLoR 回退结果：' + prediction.meta['fallback_reason'], flush=True)
        export_prediction(video, output / HANDS_WILOR, output / 'prediction.mp4')
    if annotation is not None:
        with stage(total, total, '导出 EgoDex 标注点视频（仅此阶段读取标注）'):
            export_ground_truth(video, annotation, output / 'egodex_gt.mp4')
    else:
        print('未提供 EgoDex 标注：不生成 egodex_gt.mp4。推理不受影响。', flush=True)
    prediction_meta = json.loads((output / HANDS_WILOR).with_suffix('.json').read_text())
    (output / 'SUCCESS.json').write_text(json.dumps({'video': str(video), 'mode': mode,
        'camera_source': 'da3', 'inference_calibration': None, 'annotation_used_only_after_inference': str(annotation) if annotation else None,
        'optimizer_fallback': bool(prediction_meta.get('optimizer_fallback')),
        'prior_executed': bool(prediction_meta.get('prior_executed')),
        'fallback_reason': prediction_meta.get('fallback_reason'),
        'completed_at': datetime.now().astimezone().isoformat()}, indent=2) + '\n')
    print(f'全部完成：{output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--output-root', default='outputs/wilor_video')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--mode', choices=('raw', 'full'), default='full', help='full: all WiLoR optimizations (default); raw: debug only')
    parser.add_argument('--gt', default='')
    parser.add_argument('--da3-python', default='thirdparty/.venvs/wilor-da3/bin/python')
    parser.add_argument('--wilor-python', default=DEFAULTS['wilor']['python'])
    parser.add_argument('--sam3-python', default=None)
    parser.add_argument('--dynhamr-python', default=None)
    parser.add_argument('--da3-samples', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--check', action='store_true', help='Check files/dependencies only; no model loading or video export')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    video = resolve(args.video)
    if not video.is_file() or video.suffix.lower() != '.mp4':
        parser.error('VIDEO must be an existing MP4 file')
    annotation = resolve(args.gt) if args.gt else video.with_suffix('.hdf5')
    if args.gt and not annotation.is_file():
        parser.error('GT_HDF5 does not exist')
    annotation = annotation if annotation.is_file() else None
    given = settings(args.mode, args.da3_python, args.wilor_python, args.da3_samples, args.batch_size,
                     args.sam3_python, args.dynhamr_python)
    # --check uses a temporary config and creates no output/video artifacts.
    if args.check:
        import tempfile

        with tempfile.TemporaryDirectory(prefix='wilor_check_') as tmp:
            path = Path(tmp) / 'config.yaml'
            path.write_text(yaml.safe_dump(given))
            preflight(load_options(path))
        print('检查通过；未运行模型，未导出视频。')
        return
    name = f'{datetime.now():%Y%m%d_%H%M%S_%f}_{video.parent.name}_{video.stem}'
    output = resolve(args.output_root) / name
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(output / '.ultralytics'))
    os.environ.setdefault('MPLCONFIGDIR', str(output / '.matplotlib'))
    config = output / 'config.yaml'
    config.write_text(yaml.safe_dump(given, sort_keys=False))
    # tee the entire process (including model-child stderr) to disk and the terminal.
    tee = subprocess.Popen(['tee', str(output / 'run.log')], stdin=subprocess.PIPE)
    saved = os.dup(1), os.dup(2)
    try:
        os.dup2(tee.stdin.fileno(), 1)
        os.dup2(tee.stdin.fileno(), 2)
        print(f'结果目录：{output}\nGPU：{args.gpu}；模式：{args.mode}\n'
              '推理输入：MP4 + DA3 估计内参（不读取 EgoDex 内参）\n'
              '总百分比按已完成阶段计数，不代表剩余耗时；帧/窗口进度由各阶段单独显示。', flush=True)
        options = load_options(config)
        try:
            preflight(options)
            execute(video, output, options, args.mode, annotation)
        except BaseException:
            import traceback

            traceback.print_exc()
            raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        for descriptor, original in zip((1, 2), saved):
            os.dup2(original, descriptor)
            os.close(original)
        tee.stdin.close()
        tee.wait()


if __name__ == '__main__':
    main()
