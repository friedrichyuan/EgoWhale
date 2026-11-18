"""Event-driven, bounded scheduling for the two standalone WiLoR profiles."""
from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
import time
import uuid

from batch_wilor_ego2robot import ROOT, ResidentClient, safe_name
from egowhale.wilor.retarget_export import RETARGET_HDF5, RETARGET_VIDEO, collect_retarget

ROLE = {"prepare": "prepare", "camera": "camera", "candidates": "wilor",
        "masks": "mask", "tracks": "postprocess", "optimizer_input": "postprocess",
        "smooth": "optimizer", "hmp": "optimizer",
        "basic": "postprocess", "merge": "postprocess", "raw_video": "write", "publish": "write"}


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def sampling_schedule(first, retry):
    return [first, *[n for n in retry if first == 0 or n < first]]


@dataclass
class Job:
    index: int
    video: Path
    output: Path
    identity: dict
    signature: str
    done: dict = field(default_factory=dict)
    submitted: set = field(default_factory=set)
    params: dict = field(default_factory=dict)
    windows: deque = field(default_factory=deque)
    window_results: list = field(default_factory=list)
    windows_active: int = 0
    publish_reused: bool = False
    fallback_reason: str | None = None
    error: str | None = None


@dataclass
class Task:
    job: Job
    stage: str
    key: str
    args: dict
    trial: int = 0


class Batch:
    def __init__(self, options, profile, cfg_path, python, root, jobs, refined, long_video):
        self.config = cfg_path
        self.config_bytes = cfg_path.read_bytes()
        self.options = options
        self.jobs = deque(jobs)
        self.total = len(jobs)
        self.root = root
        self.refined = refined
        self.stop = threading.Event()
        self.processes = set()
        self.lock = threading.Lock()
        self.free = defaultdict(deque)
        self.clients = {}
        self.ready = defaultdict(deque)
        self.running = {}
        self.inflight = {}
        self.rows = []
        self.events = None
        self.da3_budgets = sampling_schedule(int(long_video.get("da3_max_views", 128)),
                                             list(long_video.get("da3_retry_views", [64, 32])))
        self.mask_budgets = sampling_schedule(int(long_video.get("sam3_max_frames", 256)),
                                              list(long_video.get("sam3_retry_frames", [128, 64])))
        self.hmp_overlap = int(long_video.get("hmp_overlap", 16))
        self.window_limit = int(options.get("max_pending_hmp_per_video", 4))
        self.max_inflight = int(options.get("max_inflight_videos", 5))
        self.max_source_bytes = int(options.get("max_inflight_source_bytes", 0))
        self.required = profile["required"]
        thirdparty = Path(os.environ.get("EGOWHALE_WILOR_THIRDPARTY", ROOT / "thirdparty")).resolve()
        env = os.environ.copy()
        env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUNBUFFERED="1")
        env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
        env["PYTHONPATH"] = os.pathsep.join([str(thirdparty / p) for p in
                                            ("da3/src", "sam3", "human_body_prior/src")]
                                           + [env.get("PYTHONPATH", "")])
        env.setdefault("MPLCONFIGDIR", str(root / ".matplotlib"))
        for role, gpus in profile["cards"].items():
            for gpu in gpus:
                self.add_slot(role, f"{role}:{gpu}", str(gpu), python, env)
        for role in ("prepare", "postprocess", "write"):
            for i in range(int(options.get(role + "_workers", 2))):
                cpu_env = {**env, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
                self.add_slot(role, f"{role}:cpu{i}", "", python, cpu_env)

    def add_slot(self, role, label, gpu, python, env):
        client = ResidentClient(python, {**env, "CUDA_VISIBLE_DEVICES": gpu},
                                float(self.options.get("heartbeat_seconds", 10)),
                                self.processes, self.lock, self.stop)
        self.clients[label] = client
        self.free[role].append(label)

    def submit(self, job, stage, key=None, **args):
        key = key or stage
        if key in job.submitted or job.error:
            return
        job.submitted.add(key)
        self.ready[ROLE[stage]].append(Task(job, stage, key, args))

    def work(self, slot, task):
        job, stage = task.job, task.stage
        if self.config.read_bytes() != self.config_bytes:
            raise RuntimeError("WiLoR configuration changed during the batch")
        # Done markers are only reused when the stage and its dependencies use
        # the same inputs. A failed attempt never replaces a previous success.
        stage_dir = job.output / ".stages" / task.key
        stage_dir.mkdir(parents=True, exist_ok=True)
        done_path = stage_dir / "done.json"
        def fingerprint(args):
            task_input = {**args, "stage": stage, "run_signature": job.signature}
            return hashlib.sha256(json.dumps(task_input, sort_keys=True).encode()).hexdigest()
        if done_path.is_file():
            previous = json.loads(done_path.read_text())
            allowed = [task.args]
            if stage == "camera":
                allowed.extend({**task.args, "max_views": budget} for budget in self.da3_budgets)
            if stage == "masks":
                allowed.extend({**task.args, "max_mask_frames": budget} for budget in self.mask_budgets)
            if (previous["fingerprint"] in {fingerprint(args) for args in allowed}
                    and all(Path(p).is_file() and Path(p).stat().st_size for p in previous["files"])):
                return 0, Path(previous["attempt"]), 0., "reused"
            done_path.unlink()
        attempt = stage_dir / uuid.uuid4().hex[:12]
        attempt.mkdir()
        progress = attempt / "progress.json"
        label = f"{job.index}/{self.total} {job.video.name} {slot}/{task.key}"
        request = {**task.args, "parallel": True, "stage": stage, "video": str(job.video),
                   "output": str(job.output), "attempt_dir": str(attempt), "config": str(self.config),
                   "label": label, "progress_file": str(progress), "identity": job.identity,
                   "signature": job.signature}
        started = time.time()
        code = self.clients[slot].request(request, attempt / "worker.log", label, progress)
        if code == 0:
            work = job.output / "action/wilor_ego2robot"
            hands = work / "hand_processor"
            outputs = {
                "prepare": [work / "input.json", work / "video_L.mp4"],
                "camera": [work / name for name in ("cam_intrinsics.json", "cam_extrinsics.npz", "da3_samples.json")],
                "candidates": sorted((attempt / "candidates").glob("*.npz")),
                "masks": [attempt / "masks.npy"],
                "tracks": [hands / f"hand_data_raw_{side}.npz" for side in ("left", "right")],
                "basic": [hands / f"hand_data_{side}.npz" for side in ("left", "right")],
                "optimizer_input": ([attempt / "skipped.json"] if (attempt / "skipped.json").exists() else
                                    [attempt / "inputs/dynhamr/cameras/wilor/shot-0/cameras.npz",
                                     attempt / "inputs/images/wilor/000000.jpg",
                                     attempt / "inputs/video/wilor.mp4"]),
                "smooth": [attempt / "smooth/smooth_completed.json"],
                "hmp": [attempt / "window/result.npz"],
                "merge": [work / "optimization_status.json",
                          *[hands / f"hand_data_{side}.npz" for side in ("left", "right")]],
                "raw_video": [hands / "video_hand_overlay_anyhand.mp4"],
                "publish": [job.output / "action/hands_wilor.npz", job.output / "action/hands_wilor.json",
                            job.output / RETARGET_HDF5, job.output / RETARGET_VIDEO,
                            work / "full_completed.json", work / "optimized_hand_points.mp4", work / "annotation_counts.json"],
            }[stage]
            if not outputs or any(not p.is_file() or p.stat().st_size == 0 for p in outputs):
                raise ValueError(f"Missing stage output after {stage}")
            if stage == "smooth":
                info = json.loads(outputs[0].read_text())
                outputs.extend([Path(info["smooth_result"]), attempt / "smooth/observations.npz",
                                attempt / "smooth/hmp_config.yaml"])
            write_json(done_path, {"attempt": str(attempt), "fingerprint": fingerprint(task.args),
                                   "files": list(map(str, outputs))})
        return code, attempt, started, "completed" if code == 0 else "failed"

    def dispatch(self, pool):
        for role, tasks in self.ready.items():
            while self.free[role] and tasks:
                task = tasks.popleft()
                if task.job.error:
                    continue
                slot = self.free[role].popleft()
                future = pool.submit(self.work, slot, task)
                self.running[future] = (task, slot)

    def enqueue_windows(self, job):
        while job.windows and job.windows_active < self.window_limit:
            window = job.windows.popleft()
            self.submit(job, "hmp", f"hmp_{window['hand']}_{window['start']:06}", **window)
            job.windows_active += 1
        if not job.windows and not job.windows_active:
            self.submit(job, "merge", smooth=job.params.get("smooth"),
                        windows=sorted(job.window_results, key=lambda w: (w["hand"], w["start"])),
                        fallback_reason=job.fallback_reason, masks_available=bool(job.params.get("masks")),
                        tracks=str(job.done["tracks"]), camera=str(job.done["camera"]))

    def begin_prior(self, job, attempt):
        from egowhale.wilor.pipeline.hmp import windows

        smooth = attempt / "smooth"
        info = json.loads((smooth / "smooth_completed.json").read_text())
        count = info["frames"]
        job.params["smooth"] = str(smooth)
        spans = [(-1, 0, 128)] if count == 128 else [
            (hand, a, b) for hand in range(len(info["tracks"])) for a, b in windows(count, self.hmp_overlap)
        ]
        job.windows.extend({"smooth": str(smooth), "hand": hand, "start": a, "stop": b} for hand, a, b in spans)
        self.enqueue_windows(job)

    def advance(self, job):
        done = job.done
        if "prepare" not in done:
            return
        prepared = str(done["prepare"])
        self.submit(job, "camera", prepared=prepared, max_views=self.da3_budgets[0])
        self.submit(job, "candidates", prepared=prepared)
        if self.refined:
            self.submit(job, "masks", prepared=prepared, max_mask_frames=self.mask_budgets[0])
            if "candidates" in done and ("masks" in done or job.params.get("mask_failed")):
                self.submit(job, "tracks", candidates=str(done["candidates"] / "candidates"),
                            masks=job.params.get("masks"))
            if "tracks" in done:
                self.submit(job, "raw_video", tracks=str(done["tracks"]))
                if "camera" in done:
                    self.submit(job, "optimizer_input", tracks=str(done["tracks"]), camera=str(done["camera"]))
            if "optimizer_input" in done and not job.fallback_reason:
                self.submit(job, "smooth", inputs=str(done["optimizer_input"] / "inputs"))
            if "raw_video" in done and "merge" in done:
                self.submit(job, "publish", prediction=str(done["merge"]), raw_video=str(done["raw_video"]))
        elif "camera" in done and "candidates" in done:
            self.submit(job, "basic", candidates=str(done["candidates"] / "candidates"), camera=str(done["camera"]))
            if "basic" in done:
                self.submit(job, "publish", prediction=str(done["basic"]))

    def failed(self, task, attempt):
        job, stage = task.job, task.stage
        log_file = attempt / "worker.log"
        tail = ""
        if log_file.is_file():
            with log_file.open("rb") as log:
                log.seek(max(0, log_file.stat().st_size - 12000))
                tail = log.read().decode(errors="replace").lower()
        oom = any(token in tail for token in ("outofmemoryerror", "out of memory", "cuda_error_out_of_memory"))
        budgets, field_name = {"camera": (self.da3_budgets, "max_views"),
                               "masks": (self.mask_budgets, "max_mask_frames")}.get(stage, ([], None))
        retry = (oom and task.trial + 1 < len(budgets)) or (stage in {"hmp", "publish", "raw_video", "masks"} and task.trial == 0)
        if retry:
            task.trial += 1
            if oom and field_name and task.trial < len(budgets):
                task.args[field_name] = budgets[task.trial]
            self.ready[ROLE[stage]].append(task)
            return
        reason = f"{task.key} failed; {log_file}"
        if stage == "hmp":
            job.fallback_reason = reason
            job.windows.clear()
            job.windows_active -= 1
            self.enqueue_windows(job)
        elif stage in {"optimizer_input", "smooth"} and not self.required:
            job.fallback_reason = reason
            self.enqueue_windows(job)
        elif stage == "masks" and not self.required:
            job.params["mask_failed"] = True
        else:
            job.error = reason

    def complete(self, future, task, slot):
        self.free[ROLE[task.stage]].append(slot)
        job, stage = task.job, task.stage
        try:
            code, attempt, started, state = future.result()
            self.events.write(json.dumps({"video": str(job.video), "stage": task.key, "worker": slot,
                                          "start": started or None, "end": time.time(), "state": state,
                                          "attempt": str(attempt)}, ensure_ascii=False) + "\n")
            self.events.flush()
            if job.error or self.stop.is_set():
                return
            if code:
                self.failed(task, attempt)
            else:
                job.done[task.key] = attempt
                if stage == "publish":
                    job.publish_reused = state == "reused"
                elif stage == "masks":
                    job.params["masks"] = str(attempt / "masks.npy")
                elif stage == "optimizer_input" and (attempt / "skipped.json").exists():
                    job.fallback_reason = "no_eligible_tracks"
                    self.enqueue_windows(job)
                elif stage == "smooth":
                    self.begin_prior(job, attempt)
                elif stage == "hmp":
                    job.window_results.append({**task.args, "result": str(attempt / "window/result.npz")})
                    job.windows_active -= 1
                    self.enqueue_windows(job)
            self.advance(job)
        except Exception as exc:
            job.error = f"{task.key}: {type(exc).__name__}: {exc}"

    def record(self, job, writer):
        work = job.output / "action/wilor_ego2robot"
        video = work / "optimized_hand_points.mp4"
        row = {"index": job.index, "source": str(job.video), "frames": "", "video": "", "retarget_hdf5": "", "result_source": "",
               "error": job.error or "", "reused": job.publish_reused, "status": "中断" if self.stop.is_set() else "失败"}
        if "publish" in job.done and not job.error:
            counts = json.loads((work / "annotation_counts.json").read_text())
            result = json.loads((work / "full_completed.json").read_text())["optimization"]
            row.update(frames=json.loads((work / "input.json").read_text())["frames"], video=str(video),
                       status="成功" if any(counts.values()) else "成功（无可见手部标注）",
                       result_source=result["result_source"])
            row["retarget_hdf5"] = str(collect_retarget(self.root / "retarget_inputs", job.output.name, job.output))
            link = self.root / "final_videos" / f"{job.output.name}.mp4"
            if not link.is_symlink():
                link.symlink_to(os.path.relpath(video, link.parent))
        self.rows.append(row)
        writer.writerow(row)
        print(f"[{job.index}/{self.total}] {row['status']} {row['source']} {row['error']}", flush=True)

    def interrupt(self, *_):
        self.stop.set()
        with self.lock:
            for process in list(self.processes):
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def can_admit(self):
        if len(self.inflight) >= self.max_inflight:
            return False
        if not self.inflight or not self.max_source_bytes:
            return True
        active_bytes = sum(job.identity["size"] for job in self.inflight.values())
        return active_bytes + self.jobs[0].identity["size"] <= self.max_source_bytes

    def run(self):
        check = self.root / ".preflight"
        check.mkdir(exist_ok=True)
        slot = self.free["camera"][0]
        try:
            result = self.clients[slot].request(
                {"parallel": True, "stage": "check", "output": str(check), "video": str(self.jobs[0].video),
                 "attempt_dir": str(check), "config": str(self.config), "label": "预检",
                 "progress_file": str(check / "progress.json")},
                check / "worker.log", "预检", check / "progress.json")
        except BaseException:
            self.clients[slot].close()
            raise
        if result:
            self.clients[slot].close()
            return result
        started = time.monotonic()
        previous = signal.signal(signal.SIGINT, self.interrupt)
        fields = ("index", "source", "status", "frames", "video", "retarget_hdf5", "result_source", "reused", "error")
        try:
            with ThreadPoolExecutor(max_workers=len(self.clients)) as pool, (
                    self.root / "results.csv").open("w", encoding="utf-8-sig", newline="") as csv_file, (
                    self.root / "stages.jsonl").open("a") as self.events:
                writer = csv.DictWriter(csv_file, fieldnames=fields)
                writer.writeheader()
                while self.jobs or self.inflight or self.running:
                    while self.jobs and not self.stop.is_set() and self.can_admit():
                        job = self.jobs.popleft()
                        self.inflight[job.index] = job
                        self.submit(job, "prepare")
                    if not self.stop.is_set():
                        self.dispatch(pool)
                    if self.running:
                        finished, _ = wait(self.running, return_when=FIRST_COMPLETED)
                        for future in finished:
                            task, slot = self.running.pop(future)
                            self.complete(future, task, slot)
                    active_ids = {task.job.index for task, _ in self.running.values()}
                    for index, job in list(self.inflight.items()):
                        if index not in active_ids and (job.error or "publish" in job.done or self.stop.is_set()):
                            self.record(job, writer)
                            del self.inflight[index]
                    csv_file.flush()
                    if self.stop.is_set() and not self.running:
                        for job in self.jobs:
                            self.record(job, writer)
                        break
                    if self.inflight and not self.running and not any(
                            task.job.error is None for tasks in self.ready.values() for task in tasks):
                        raise RuntimeError("WiLoR scheduler has an unfinished job with no runnable task")
        finally:
            signal.signal(signal.SIGINT, previous)
            for client in self.clients.values():
                client.close()
        seconds = time.monotonic() - started
        success = [r for r in self.rows if r["status"].startswith("成功")]
        frames = sum(r["frames"] for r in success)
        new_frames = sum(r["frames"] for r in success if not r.get("reused"))
        write_json(self.root / "throughput.json", {
            "wall_seconds": seconds, "videos": self.total, "success": len(success),
            "failed_or_interrupted": self.total - len(success), "successful_frames": frames,
            "processed_frames_this_run": new_frames, "aggregate_fps": new_frames / seconds, "hmp_success": sum(r["result_source"] == "hmp" for r in success),
            "no_visible_hands": sum("无可见" in r["status"] for r in success),
            "result_sources": {key: sum(r["result_source"] == key for r in success)
                               for key in ("wilor", "wilor_filtered", "smooth_fit", "hmp")},
        })
        print(f"批次输出 {self.root}", flush=True)
        return 130 if self.stop.is_set() else int(len(success) != self.total)


def run_parallel(document, args, videos, output_root, config, python, parser):
    from egowhale.wilor.synced import identity, signature

    options = document.split
    refined = bool(document.get("dynhamr_enabled", False))
    profile_name = "refined" if refined else "basic"
    profiles = options.get("profiles", {})
    profile = profiles.get(profile_name, options if not refined else {})
    roles = ("camera", "wilor", "mask", "optimizer") if refined else ("camera", "wilor")
    cards = {role: list(profile.get(role + "_gpus", [])) for role in roles}
    all_cards = [str(gpu) for group in cards.values() for gpu in group]
    if not all(cards.values()) or any(not gpu.isdigit() for gpu in all_cards) or len({int(gpu) for gpu in all_cards}) != len(all_cards):
        parser.error(f"{profile_name} 需要为 {', '.join(roles)} 分别配置互不重复的物理 GPU")
    if min(int(options.get(key, default)) for key, default in
           (("max_inflight_videos", 5), ("max_pending_hmp_per_video", 4), ("prepare_workers", 2),
            ("postprocess_workers", 2), ("write_workers", 2))) < 1:
        parser.error("worker 数与在途容量必须为正")
    if int(options.get("max_inflight_source_bytes", 0)) < 0:
        parser.error("在途源文件字节上限须非负")
    if document.get("hand_backend", "anyhand") != "anyhand" or document.get("anyhand_backend", "wilor") != "wilor":
        parser.error("并行路线使用 AnyHand（WiLoR 子后端）")
    long_video = document.get("long_video", {})
    da3 = list(long_video.get("da3_retry_views", [64, 32]))
    masks = list(long_video.get("sam3_retry_frames", [128, 64]))
    if (int(long_video.get("da3_max_views", 128)) not in (0,) and int(long_video.get("da3_max_views", 128)) < 2
            or int(long_video.get("sam3_max_frames", 256)) < 0 or any(n < 2 for n in da3 + masks)
            or da3 != sorted(set(da3), reverse=True) or masks != sorted(set(masks), reverse=True)
            or not 0 <= int(long_video.get("hmp_overlap", 16)) < 128):
        parser.error("长视频预算或 HMP 重叠长度无效")
    print(f"WiLoR {profile_name} | GPUs {cards} | 视频 {len(videos)} | 长视频配置 {long_video}")
    if args.dry_run:
        for index, video in enumerate(videos, 1):
            print(f"{index}/{len(videos)} {video}")
        return 0

    root = args.resume_run.resolve() if args.resume_run else output_root / (
        datetime.now().astimezone().strftime("parallel_%Y%m%d_%H%M%S_%f") + f"_{os.getpid()}")
    jobs = [Job(i, video, root / f"{i:05d}_{safe_name(video.parent.name)}_{safe_name(video.stem)}",
                identity(video), signature(video, config, None, "da3", .3))
            for i, video in enumerate(videos, 1)]
    manifest = {"profile": profile_name, "videos": [job.identity for job in jobs]}
    if args.resume_run:
        if json.loads((root / "manifest.json").read_text()) != manifest:
            parser.error("恢复批次的模式、视频清单和源文件须与原运行一致")
    else:
        root.mkdir(parents=True)
        (root / "final_videos").mkdir()
        write_json(root / "manifest.json", manifest)

    batch = Batch(options, {"cards": cards, "required": bool(document.get("dynhamr_required", False))},
                  config, python, root, jobs, refined, long_video)
    import fcntl
    with (root / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"已有进程处理此批次：{root}")
        return batch.run()
