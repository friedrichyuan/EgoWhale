"""Run the complete single-video WiLoR launcher sequentially over a dataset."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'scripts/run_wilor_video.sh'


def resolve(path):
    p = Path(path).expanduser()
    return (p if p.is_absolute() else ROOT / p).resolve()


def identity(path):
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def completed(marker, video):
    try:
        data = json.loads(marker.read_text())
        run = Path(data['output'])
        success = json.loads((run / 'SUCCESS.json').read_text())
        annotation = video.with_suffix('.hdf5')
        return (data['input'] == identity(video)
                and data.get('annotation') == (identity(annotation) if annotation.is_file() else None)
                and success['video'] == str(video) and success['mode'] == 'full'
                and success.get('camera_source') == 'da3' and success.get('inference_calibration') is None
                and (run / 'prediction.mp4').is_file() and (run / 'action/hands_wilor.npz').is_file()
                and (not annotation.is_file() or (run / 'egodex_gt.mp4').is_file()))
    except (OSError, KeyError, TypeError, ValueError):
        return False


def command(video, output, gpu, *, check=False):
    args = ['bash', str(LAUNCHER), '--video', str(video), '--output-root', str(output),
            '--gpu', gpu, '--mode', 'full', '--gt', '']
    return [*args, '--check'] if check else args


def run_child(args):
    # The single-video launcher also writes its own complete run.log.
    child = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, errors='replace', bufsize=1)
    oom = False
    try:
        for line in child.stdout:
            print(line, end='', flush=True)
            oom |= 'out of memory' in line.lower() or 'outofmemoryerror' in line.lower()
        return child.wait(), oom
    except BaseException:
        child.terminate()
        child.wait()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--skip-completed', choices=('0', '1'), default='1')
    parser.add_argument('--stop-on-error', choices=('0', '1'), default='0')
    parser.add_argument('--dry-run', action='store_true', help='List counts without inference or output creation')
    args = parser.parse_args()
    dataset, output_root = resolve(args.dataset), resolve(args.output_root)
    if not dataset.is_dir():
        parser.error(f'数据集目录不存在：{dataset}')
    if output_root == dataset or output_root.is_relative_to(dataset):
        parser.error('输出目录不能位于数据集目录内，避免把结果视频当成输入')
    videos = sorted((p.resolve() for p in dataset.rglob('*') if p.is_file() and p.suffix.lower() == '.mp4'),
                    key=lambda p: str(p))
    if not videos:
        parser.error(f'没有找到 MP4 视频：{dataset}')

    jobs = []
    for video in videos:
        try:
            relative = video.relative_to(dataset)
        except ValueError:
            parser.error(f'视频链接指向数据集目录之外：{video}')
        job_root = output_root / relative.parent / relative.name
        marker = job_root / 'batch-success.json'
        jobs.append((video, job_root, marker, args.skip_completed == '1' and completed(marker, video)))
    pending = sum(not job[3] for job in jobs)
    print(f'数据集：{dataset}\n输出目录：{output_root}\nGPU：{args.gpu}；模式：full\n'
          f'视频总数：{len(jobs)}；已完成跳过：{len(jobs) - pending}；本次待运行：{pending}', flush=True)
    if args.dry_run:
        for video, _, _, skip in jobs[:10]:
            print(f'{"跳过" if skip else "待运行"}：{video.relative_to(dataset)}')
        if len(jobs) > 10:
            print(f'……共 {len(jobs)} 条，仅显示前 10 条。')
        return 0
    if not pending:
        return 0

    first = next(job for job in jobs if not job[3])
    print('开始检查完整流程资源和 GPU（不运行模型）……', flush=True)
    code, _ = run_child(command(first[0], first[1], args.gpu, check=True))
    if code:
        print('启动检查失败，批处理未开始。', file=sys.stderr)
        return 1

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / 'batch.log'
    succeeded = failed = skipped = 0
    with summary_path.open('a', encoding='utf-8', buffering=1) as log:
        def report(message):
            line = f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}'
            print(line, flush=True)
            log.write(line + '\n')

        report(f'批次开始：总数 {len(jobs)}，待运行 {pending}，GPU {args.gpu}')
        for index, (video, job_root, marker, skip) in enumerate(jobs, 1):
            relative = video.relative_to(dataset)
            if skip:
                skipped += 1
                report(f'[{index}/{len(jobs)}] 跳过已完成：{relative}')
                continue
            job_root.mkdir(parents=True, exist_ok=True)
            before = {p.name for p in job_root.iterdir() if p.is_dir()}
            input_before = identity(video)
            annotation = video.with_suffix('.hdf5')
            annotation_before = identity(annotation) if annotation.is_file() else None
            report(f'[{index}/{len(jobs)}] 开始：{relative}；总体已遍历 {index - 1}/{len(jobs)}')
            code, oom = run_child(command(video, job_root, args.gpu))
            created = [p for p in job_root.iterdir() if p.is_dir() and p.name not in before]
            unchanged = (video.is_file() and identity(video) == input_before
                         and (identity(annotation) if annotation.is_file() else None) == annotation_before)
            if code == 0 and len(created) == 1 and unchanged:
                temp = marker.with_suffix('.tmp')
                temp.write_text(json.dumps({'input': input_before, 'annotation': annotation_before,
                    'output': str(created[0])}, ensure_ascii=False, indent=2))
                os.replace(temp, marker)
                if completed(marker, video):
                    succeeded += 1
                    report(f'[{index}/{len(jobs)}] 成功：{relative}；结果：{created[0]}')
                    continue
                marker.unlink(missing_ok=True)
            failed += 1
            report(f'[{index}/{len(jobs)}] 失败：{relative}；退出码 {code}；结果目录：{job_root}')
            if oom or args.stop_on_error == '1':
                report('检测到显存不足，停止批处理；释放显存或更换 GPU 后可重新运行。' if oom
                       else '按 STOP_ON_ERROR 设置停止批处理。')
                break
        report(f'本次结束：成功 {succeeded}，失败 {failed}，跳过 {skipped}；清单：{summary_path}')
    return 1 if failed else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\n批处理已中断；已成功的视频下次可跳过。', file=sys.stderr)
        raise SystemExit(130) from None
