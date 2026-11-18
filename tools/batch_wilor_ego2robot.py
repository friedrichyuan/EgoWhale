#!/usr/bin/env python3
"""Run EgoWhale's synchronized WiLoR node on a directory, one worker per GPU."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
import queue
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
RESIDENT_NODE = ROOT / 'egowhale/wilor/synced_worker.py'
RESPONSE = '@@EGOWHALE_WILOR_RESPONSE@@'
PREFLIGHT = ROOT / 'egowhale/wilor/ego2robot_sync/scripts/run_hand_experiment.py'
CONFIG = ROOT / 'egowhale/wilor/cfgs/wilor.yaml'
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


class ResidentClient:
    """One subprocess per GPU, receiving one video at a time."""
    def __init__(self, python, env, interval, active, lock, stop):
        self.python, self.env, self.interval = python, env, interval
        self.active, self.lock, self.stop = active, lock, stop
        self.process = None
        self.lines = None
        self.reader = None

    def start(self):
        if self.process is not None and self.process.poll() is None:
            return
        self.close()
        self.lines = queue.Queue()
        self.process = subprocess.Popen([str(self.python), '-u', str(RESIDENT_NODE)], cwd=ROOT,
            env=self.env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', bufsize=1, start_new_session=True)
        process, lines = self.process, self.lines
        with self.lock:
            self.active.add(process)
        def drain():
            try:
                for line in process.stdout:
                    lines.put(line)
            finally:
                process.stdout.close()
                lines.put(None)
        self.reader = threading.Thread(target=drain, daemon=True)
        self.reader.start()

    def request(self, request, log_path, label, progress_path):
        self.start()
        started = time.monotonic()
        process = self.process
        with log_path.open('w', encoding='utf-8') as log:
            try:
                process.stdin.write(json.dumps(request, ensure_ascii=False) + '\n')
                process.stdin.flush()
                while True:
                    try:
                        line = self.lines.get(timeout=self.interval)
                    except queue.Empty:
                        print(f'[{label}] {progress(progress_path)} | 已耗时 {int(time.monotonic()-started)}s', flush=True)
                        if self.stop.is_set():
                            self.close()
                            return 130
                        continue
                    if line is None:
                        raise RuntimeError(f'常驻 worker 异常退出，退出码 {process.poll()}')
                    if line.startswith(RESPONSE):
                        response = json.loads(line[len(RESPONSE):])
                        if not response.get('ok'):
                            print(f'[{label}] worker 处理失败：{response.get("error", "未知错误")}', flush=True)
                            self.close()
                            return 1
                        return 0
                    log.write(line)
                    log.flush()
                    print(f'[{label}] {line}', end='', flush=True)
            except (BrokenPipeError, OSError, ValueError, RuntimeError) as exc:
                print(f'[{label}] 常驻 worker 错误：{exc}', flush=True)
                log.write(f'常驻 worker 错误：{exc}\n')
                self.close()
                return 1

    def close(self):
        process, self.process = self.process, None
        if process is None:
            return
        if process.poll() is None:
            try:
                process.stdin.write('{"command":"stop"}\n')
                process.stdin.flush()
                process.stdin.close()
                process.wait(timeout=5)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        if process.stdin and not process.stdin.closed:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
        if self.reader:
            self.reader.join(timeout=5)
            self.reader = None
        with self.lock:
            self.active.discard(process)




if __name__ == '__main__':
    # Compatibility entry point: all synchronized batch jobs use split scheduling.
    from split_wilor_batch import main
    raise SystemExit(main())
