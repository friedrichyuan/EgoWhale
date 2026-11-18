#!/usr/bin/env python3
"""DA3 and WiLoR on separate, persistent GPU pools with bounded hand queue."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time

from omegaconf import OmegaConf
from batch_wilor_ego2robot import ROOT, ResidentClient, discover, safe_name


def path_at_root(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def launch_parallel(document, args, parser):
    # The shell invokes this file directly from tools/.
    sys.path.insert(0, str(ROOT))
    from wilor_parallel_batch import run_parallel

    cfg = document.split
    input_dir = path_at_root(cfg.input_dir)
    output_root = path_at_root(cfg.output_root)
    python = args.python.expanduser().resolve()
    maximum = args.max_videos if args.max_videos is not None else int(cfg.get("max_videos", 0))
    if maximum < 0 or float(cfg.get("heartbeat_seconds", 10)) <= 0:
        parser.error("视频上限必须非负，心跳间隔必须为正")
    if not input_dir.is_dir() or not python.is_file() or output_root.is_relative_to(input_dir):
        parser.error("检查输入目录、输出目录和 Python 路径")
    videos = discover(input_dir)
    if maximum:
        videos = videos[:maximum]
    if not videos:
        parser.error("没有找到 MP4 视频")
    return run_parallel(document, args, videos, output_root, args.config.resolve(), python, parser)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--max-videos', type=int, help='覆盖 YAML 的视频数，0 表示全部')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume-run', type=Path, help='复用一个并行批次下已完成的阶段')
    args = parser.parse_args()
    document = OmegaConf.load(args.config.resolve())
    cfg = document.get('split', document)
    if "profiles" in cfg or args.resume_run:
        if "split" not in document:
            parser.error("并行批次需要统一 wilor.yaml 中的 split 配置")
        return launch_parallel(document, args, parser)

    camera_gpus = [str(x) for x in cfg.camera_gpus]
    hand_gpus = [str(x) for x in cfg.wilor_gpus]
    all_gpus = camera_gpus + hand_gpus
    if not camera_gpus or not hand_gpus or any(not x.isdigit() for x in all_gpus) or len(set(all_gpus)) != len(all_gpus):
        parser.error('camera_gpus / wilor_gpus 必须均非空，且数字 GPU 编号不能重复')
    input_dir, output_root = path_at_root(cfg.input_dir), path_at_root(cfg.output_root)
    pipeline_config = args.config.resolve() if 'split' in document else path_at_root(cfg.pipeline_config)
    python = args.python.expanduser().resolve()
    if not input_dir.is_dir() or not pipeline_config.is_file() or not python.is_file():
        parser.error('数据目录、pipeline_config 或 Python 不存在')
    if output_root.is_relative_to(input_dir):
        parser.error('输出目录不能在输入数据集内')
    if OmegaConf.load(pipeline_config).get('dynhamr_enabled', False):
        parser.error('分卡实验只支持关闭 DynHaMR 的配置')
    maximum = args.max_videos if args.max_videos is not None else int(cfg.get('max_videos', 10))
    capacity, interval = int(cfg.get('max_pending_camera', 4)), float(cfg.get('heartbeat_seconds', 10))
    if maximum < 0 or capacity < 1 or interval <= 0:
        parser.error('视频数必须非负，max_pending_camera 和 heartbeat_seconds 须为正数')
    videos = discover(input_dir)
    if maximum:
        videos = videos[:maximum]
    if not videos:
        parser.error('没有找到 MP4 视频')
    print(f'分卡配置：DA3={camera_gpus} | WiLoR={hand_gpus} | 视频={len(videos)} | 待处理容量={capacity}', flush=True)
    if args.dry_run:
        for index, video in enumerate(videos, 1):
            print(f'{index}/{len(videos)} {video}')
        return 0
    stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
    run_root = output_root / f'split_{stamp}_{os.getpid()}'
    run_root.mkdir(parents=True)
    (run_root / 'final_videos').mkdir()
    jobs = []
    for index, video in enumerate(videos, 1):
        name = f'{index:05d}_{safe_name(video.parent.name)}_{safe_name(video.stem)}_{stamp}'
        output = run_root / name
        output.mkdir()
        jobs.append({'index': index, 'video': video, 'output': output})
    (run_root / 'manifest.json').write_text(json.dumps({'split_config': OmegaConf.to_container(cfg, resolve=True),
        'pipeline_config': str(pipeline_config), 'videos': [str(x) for x in videos]}, ensure_ascii=False, indent=2))
    active, lock, stop = set(), threading.Lock(), threading.Event()
    camera_queue = queue.Queue()
    hand_queue = queue.Queue(maxsize=capacity)
    for job in jobs:
        camera_queue.put(job)
    rows = []
    fields = ('index', 'source', 'status', 'camera_gpu', 'wilor_gpu', 'frames', 'video', 'camera_log', 'wilor_log', 'error')
    results = (run_root / 'results.csv').open('w', encoding='utf-8-sig', newline='')
    writer = csv.DictWriter(results, fieldnames=fields)
    writer.writeheader(); results.flush()
    started = time.monotonic()

    def record(job, status, error='', video='', frames=''):
        row = {'index': job['index'], 'source': str(job['video']), 'status': status,
               'camera_gpu': job.get('camera_gpu', ''), 'wilor_gpu': job.get('wilor_gpu', ''),
               'frames': frames, 'video': str(video), 'camera_log': str(job['output'] / 'camera.log'),
               'wilor_log': str(job['output'] / 'wilor.log'), 'error': error}
        with lock:
            rows.append(row); writer.writerow(row); results.flush()
        print(f"[{job['index']}/{len(jobs)} | {job['video'].parent.name}/{job['video'].name}] {status} {error}", flush=True)

    def consume(stage, gpu):
        env = os.environ.copy()
        env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED='1')
        env['PATH'] = str(python.parent) + os.pathsep + env.get('PATH', '')
        thirdparty = Path(env.get('EGOWHALE_WILOR_THIRDPARTY', ROOT / 'thirdparty')).resolve()
        env['PYTHONPATH'] = os.pathsep.join([str(thirdparty / 'da3/src'), str(thirdparty / 'sam3'),
                                             str(thirdparty / 'human_body_prior/src'), env.get('PYTHONPATH', '')])
        client = ResidentClient(python, env, interval, active, lock, stop)
        tasks = camera_queue if stage == 'camera' else hand_queue
        try:
            while True:
                job = tasks.get()
                if job is None:
                    tasks.task_done()
                    break
                try:
                    if stop.is_set():
                        record(job, '中断', '用户中断')
                        continue
                    job['camera_gpu' if stage == 'camera' else 'wilor_gpu'] = gpu
                    tag = 'DA3' if stage == 'camera' else 'WiLoR'
                    label = f"{tag} GPU {gpu} | {job['index']}/{len(jobs)} | {job['video'].name}"
                    log = job['output'] / ('camera.log' if stage == 'camera' else 'wilor.log')
                    prog = job['output'] / ('camera_progress.json' if stage == 'camera' else 'wilor_progress.json')
                    code = client.request({'video': str(job['video']), 'output': str(job['output']),
                        'config': str(pipeline_config), 'stage': stage, 'label': label, 'progress_file': str(prog)},
                        log, label, prog)
                    if code:
                        record(job, f'{tag}失败' if not stop.is_set() else '中断', f'worker 退出码 {code}，查看日志')
                        continue
                    if stage == 'camera':
                        while not stop.is_set():
                            try:
                                hand_queue.put(job, timeout=1)
                                break
                            except queue.Full:
                                pass
                        else:
                            record(job, '中断', '未进入 WiLoR 队列')
                    else:
                        work = job['output'] / 'action/wilor_ego2robot'
                        video = work / 'optimized_hand_points.mp4'
                        if not video.is_file() or not video.stat().st_size or not (job['output'] / 'action/hands_wilor.npz').is_file():
                            raise ValueError('WiLoR 输出不完整')
                        info = json.loads((work / 'input.json').read_text())
                        counts = json.loads((work / 'annotation_counts.json').read_text())
                        link = run_root / 'final_videos' / f"{job['output'].name}.mp4"
                        link.symlink_to(os.path.relpath(video, link.parent))
                        record(job, '成功' if any(counts.values()) else '成功（无可见手部标注）', video=video, frames=info['frames'])
                except Exception as exc:
                    record(job, '失败', f'{type(exc).__name__}: {exc}')
                    client.close()
                finally:
                    tasks.task_done()
        finally:
            client.close()

    def interrupt(_sig, _frame):
        stop.set()
        print('收到中断，停止 worker 并记录未处理任务', flush=True)
        with lock:
            for process in list(active):
                if process.poll() is None:
                    try: os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError: pass

    previous = signal.signal(signal.SIGINT, interrupt)
    camera_threads = [threading.Thread(target=consume, args=('camera', gpu)) for gpu in camera_gpus]
    hand_threads = [threading.Thread(target=consume, args=('hand', gpu)) for gpu in hand_gpus]
    try:
        for t in hand_threads + camera_threads: t.start()
        camera_queue.join()
        for _ in camera_threads: camera_queue.put(None)
        for t in camera_threads: t.join()
        hand_queue.join()
        for _ in hand_threads: hand_queue.put(None)
        for t in hand_threads: t.join()
    finally:
        signal.signal(signal.SIGINT, previous)
        results.close()
    seconds = time.monotonic() - started
    success = [r for r in rows if r['status'].startswith('成功')]
    frames = sum(int(r['frames']) for r in success)
    summary = {'wall_seconds': seconds, 'videos': len(jobs), 'success': len(success),
               'failed_or_interrupted': len(rows) - len(success), 'successful_frames': frames,
               'aggregate_fps': frames / seconds if seconds else None,
               'camera_gpus': camera_gpus, 'wilor_gpus': hand_gpus}
    (run_root / 'throughput.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'实验目录：{run_root}\n' + json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 130 if stop.is_set() else 1 if len(success) != len(jobs) else 0


if __name__ == '__main__':
    raise SystemExit(main())
