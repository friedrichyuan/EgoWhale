#!/usr/bin/env python3
"""Run EgoWhale's synchronized WiLoR node on a directory, one worker per GPU."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
NODE = ROOT / 'tools/run_wilor_ego2robot.py'
PREFLIGHT = ROOT / 'egowhale/wilor/ego2robot_sync/scripts/run_hand_experiment.py'
CONFIG = ROOT / 'egowhale/wilor/cfgs/wilor_hand_pipeline.yaml'
PYTHON = ROOT.parent / '.conda-egohand/bin/python'


def natural_key(path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', str(path))]


def discover(directory):
    return sorted((p.resolve() for p in directory.rglob('*') if p.is_file() and p.suffix.lower() == '.mp4'), key=natural_key)


def safe_name(text):
    return re.sub(r'[^\w.-]+', '_', text).strip('._')[:60] or 'video'


def progress(path):
    try:
        data = json.loads(path.read_text())
        stage = data.get('stage', '处理中')
        current, total = data.get('current'), data.get('total')
        return f'{stage} {current}/{total}' if total is not None and current is not None else stage
    except (OSError, ValueError):
        return '模型加载或任务启动中'


def stream(command, env, log_path, label, progress_path, interval, active, lock, stop):
    with log_path.open('w', encoding='utf-8') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding='utf-8', errors='replace', bufsize=1, start_new_session=True)
        with lock:
            active.add(child)
        def drain():
            try:
                for line in child.stdout:
                    log.write(line)
                    log.flush()
                    print(f'[{label}] {line}', end='', flush=True)
            finally:
                child.stdout.close()
        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        started = time.monotonic()
        try:
            while True:
                try:
                    return child.wait(timeout=interval)
                except subprocess.TimeoutExpired:
                    elapsed = int(time.monotonic() - started)
                    print(f'[{label}] {progress(progress_path)} | 已耗时 {elapsed}s', flush=True)
                    if stop.is_set():
                        os.killpg(child.pid, signal.SIGTERM)
        finally:
            reader.join()
            with lock:
                active.discard(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--gpu-ids', default='0,1,2,3,4,5')
    parser.add_argument('--python', type=Path, default=PYTHON)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--camera-source', choices=('auto', 'egodex', 'da3'), default='auto')
    parser.add_argument('--max-videos', type=int, default=0)
    parser.add_argument('--heartbeat', type=float, default=10)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.camera_source not in ('auto', 'da3'):
        parser.error('批量 MP4 固定走 Path B 和 DA3，相机来源须为 auto 或 da3')
    args.camera_source = 'da3'
    args.input, args.output_root, args.python, args.config = (x.expanduser().resolve() for x in
                                                               (args.input, args.output_root, args.python, args.config))
    gpus = args.gpu_ids.split(',')
    if not gpus or any(not x.isdigit() for x in gpus) or len(set(gpus)) != len(gpus):
        parser.error('GPU 编号须为不重复的数字，例如 0,1,2')
    if args.max_videos < 0 or args.heartbeat <= 0:
        parser.error('max-videos 须非负，heartbeat 须为正数')
    if not args.input.is_dir():
        parser.error(f'输入目录不存在：{args.input}')
    if args.output_root.is_relative_to(args.input):
        parser.error('输出目录不能位于输入目录内')
    if not args.python.is_file() or not args.config.is_file():
        parser.error(f'Python 或配置文件不存在：{args.python} / {args.config}')
    videos = discover(args.input)
    if args.max_videos:
        videos = videos[:args.max_videos]
    if not videos:
        parser.error('输入目录中没有 MP4')
    stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
    run_root = args.output_root / f'batch_{stamp}_{os.getpid()}'
    jobs = [{'index': i + 1, 'video': p, 'gpu': gpus[i % len(gpus)],
             'name': f'{i + 1:05d}_{safe_name(p.parent.name)}_第{i + 1:03d}个_{safe_name(p.stem)}_{stamp}'}
            for i, p in enumerate(videos)]
    print(f'EgoWhale WiLoR | {len(jobs)} 个 MP4 | GPU {args.gpu_ids} | 输出：{run_root}', flush=True)
    if args.dry_run:
        for job in jobs:
            print(f"[GPU {job['gpu']} | {job['index']}/{len(jobs)}] Path B / DA3: {job['video']}")
        return 0
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / 'final_videos').mkdir()
    (run_root / 'batch_manifest.json').write_text(json.dumps({'created': stamp, 'config': str(args.config),
        'gpus': gpus, 'camera_source': args.camera_source,
        'jobs': [{**job, 'video': str(job['video'])} for job in jobs]}, ensure_ascii=False, indent=2))
    active, lock, stop, rows = set(), threading.Lock(), threading.Event(), []
    csv_path = run_root / 'results.csv'
    fields = ('index', 'gpu', 'source', 'status', 'left_visible_frames', 'right_visible_frames', 'video', 'log')
    csv_file = csv_path.open('w', newline='', encoding='utf-8-sig')
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    writer.writeheader()
    csv_file.flush()

    def save_row(row):
        with lock:
            rows.append(row)
            writer.writerow(row)
            csv_file.flush()

    def worker(gpu):
        assigned = [job for job in jobs if job['gpu'] == gpu]
        if not assigned:
            return
        env = os.environ.copy()
        env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED='1')
        env['PATH'] = str(args.python.parent) + os.pathsep + env.get('PATH', '')
        thirdparty = Path(env.get('EGOWHALE_WILOR_THIRDPARTY', ROOT / 'thirdparty')).resolve()
        env['PYTHONPATH'] = os.pathsep.join([str(thirdparty / 'da3/src'), str(thirdparty / 'sam3'),
                                             str(thirdparty / 'human_body_prior/src'), env.get('PYTHONPATH', '')])
        preflight_log = run_root / f'preflight_gpu_{gpu}.log'
        label = f'GPU {gpu} | 预检'
        command = [str(args.python), '-u', str(PREFLIGHT), '--video', str(assigned[0]['video']),
                   '--path', 'b', '--camera-source', 'da3', '--video-only', '--config', str(args.config), '--check']
        code = stream(command, env, preflight_log, label, run_root / f'preflight_gpu_{gpu}.json', args.heartbeat, active, lock, stop)
        if code or stop.is_set():
            for job in assigned:
                save_row({'index': job['index'], 'gpu': gpu, 'source': str(job['video']),
                          'status': '预检失败' if code else '已中断', 'log': str(preflight_log)})
            return
        for job in assigned:
            output = run_root / job['name']
            output.mkdir()
            log = output / 'process.log'
            progress_path = output / 'progress.json'
            label = f"GPU {gpu} | 第 {job['index']}/{len(jobs)} 个 | {job['video'].name}"
            if stop.is_set():
                save_row({'index': job['index'], 'gpu': gpu, 'source': str(job['video']), 'status': '已中断', 'log': str(log)})
                continue
            env['EGOHAND_VIDEO_LABEL'] = label
            env['EGOHAND_PROGRESS_FILE'] = str(progress_path)
            print(f'[{label}] 开始', flush=True)
            command = [str(args.python), '-u', str(NODE), '--video', str(job['video']), '--output', str(output),
                       '--python', str(args.python), '--config', str(args.config), '--camera-source', args.camera_source]
            code = stream(command, env, log, label, progress_path, args.heartbeat, active, lock, stop)
            work = output / 'action/wilor_ego2robot'
            video = work / 'optimized_hand_points.mp4'
            row = {'index': job['index'], 'gpu': gpu, 'source': str(job['video']), 'log': str(log)}
            try:
                counts = json.loads((work / 'annotation_counts.json').read_text())
                complete = (work / 'full_completed.json').is_file() and (output / 'action/hands_wilor.npz').is_file()
                if code or not complete or not video.is_file() or not video.stat().st_size:
                    raise ValueError(f'退出码 {code}，或输出不完整')
                left, right = counts['left'], counts['right']
                link = run_root / 'final_videos' / f"{job['name']}.mp4"
                link.symlink_to(os.path.relpath(video, link.parent))
                status = '成功' if left or right else '成功（无可见手部标注）'
                row.update(status=status, left_visible_frames=left, right_visible_frames=right, video=str(video))
                print(f'[{label}] {status} | 左手 {left} 帧、右手 {right} 帧 | {video}', flush=True)
            except (OSError, ValueError, KeyError, TypeError) as error:
                row['status'] = f'失败：{error}'
                print(f'[{label}] {row["status"]} | 日志：{log}', flush=True)
            save_row(row)

    previous = signal.getsignal(signal.SIGINT)
    def interrupt(signum, frame):
        stop.set()
        print('\n收到中断，正在停止当前任务…', flush=True)
        with lock:
            for child in list(active):
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
    signal.signal(signal.SIGINT, interrupt)
    try:
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(worker, gpu) for gpu in gpus]
            for future in futures:
                future.result()
    finally:
        signal.signal(signal.SIGINT, previous)
        csv_file.close()
    failed = sum(not row['status'].startswith('成功') for row in rows)
    empty = sum('无可见手部标注' in row['status'] for row in rows)
    print(f'批次结束：完成 {len(rows) - failed}/{len(jobs)}，失败或中断 {failed}，其中无可见点 {empty}。\n'
          f'标注视频：{run_root / "final_videos"}\n结果表：{csv_path}', flush=True)
    return 130 if stop.is_set() else 1 if failed or len(rows) != len(jobs) else 0


if __name__ == '__main__':
    raise SystemExit(main())
