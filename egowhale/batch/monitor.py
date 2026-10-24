"""One-second physical resource samples, grouped by actor class rather than Ray calls.

CPU cores = process CPU seconds / elapsed seconds, including live descendants.
RSS includes shared mappings and must not be interpreted as unique host RAM.
GPU SM samples are per-process activity, not additive shares of GPU capacity.
Missing/unsupported NVML measurements are omitted, never replaced with zero.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from pathlib import Path

import psutil


class Resources:
    """Single-host sampler. No per-frame hooks or Ray RPCs during sampling."""

    def __init__(self, out: Path, identities: list[dict], origin: float, interval: float = 1.0):
        from torch.utils.tensorboard import SummaryWriter

        self.origin = origin
        self.interval = interval
        self.identities = identities
        self.roots = {item["pid"]: item["actor_class"] for item in identities}
        self.classes = sorted(set(self.roots.values()))
        self.previous = {}
        self.stop = threading.Event()
        self.writer = SummaryWriter(str(out / "tb" / "resources"), flush_secs=5)
        self.writer.add_custom_scalars({"Actor resources": {
            "CPU used cores": ["Multiline", ["resource/.*/cpu_cores"]],
            "CPU host percent": ["Multiline", ["resource/.*/cpu_host_pct"]],
            "RAM RSS GB (shared mappings counted repeatedly)": ["Multiline", ["resource/.*/ram_rss_gb"]],
            "GPU VRAM GB": ["Multiline", ["resource/.*/gpu_vram_gb"]],
            "GPU process SM activity mean percent": ["Multiline", ["resource/.*/gpu_sm_process_mean_pct"]],
            "GPU SM sample coverage": ["Multiline", ["resource/.*/gpu_sm_sample_coverage"]],
        }})
        self.file = (out / "resources.jsonl").open("w")
        self.nvml = None
        self.devices = []
        self.stamps = {}
        self.error = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.devices = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        except Exception as exc:
            self.error = str(exc)
        (out / "resources_meta.json").write_text(json.dumps({
            "actors": identities, "interval_seconds": interval, "nvml_error": self.error,
            "cpu_cores": "CPU seconds / wall seconds, including live child processes",
            "ram_rss_gb": "Sum of RSS; shared Ray mappings may be counted more than once",
            "gpu_sm_process_mean_pct": "Mean over process-device pairs with fresh driver samples; not a capacity share",
            "gpu_sm_sample_coverage": "Freshly sampled pairs / resident GPU process-device pairs; missing values omitted",
        }, indent=2))
        self.sample()  # Prime CPU deltas before the workload starts.
        self.thread = threading.Thread(target=self.run, name="actor-resources", daemon=True)

    def start(self):
        self.thread.start()

    def processes(self):
        owners = {}
        for pid, actor_class in self.roots.items():
            try:
                root = psutil.Process(pid)
                owners[pid] = (root, actor_class)
                for child in root.children(recursive=True):
                    owners.setdefault(child.pid, (child, actor_class))
            except psutil.Error:
                pass
        return owners

    def gpu(self, owners):
        n = self.nvml
        memory, samples, resident = defaultdict(int), defaultdict(list), defaultdict(int)
        devices = {}
        if n is None:
            return memory, samples, resident, devices
        for index, handle in enumerate(self.devices):
            processes = {}
            for query in (n.nvmlDeviceGetComputeRunningProcesses, n.nvmlDeviceGetGraphicsRunningProcesses):
                try:
                    for process in query(handle):
                        processes[process.pid] = process
                except n.NVMLError:
                    pass
            relevant = {pid: p for pid, p in processes.items() if pid in owners}
            if not relevant:
                continue
            try:
                devices[str(index)] = {"gpu_util_pct": n.nvmlDeviceGetUtilizationRates(handle).gpu,
                                       "vram_gb": n.nvmlDeviceGetMemoryInfo(handle).used / 2**30}
            except n.NVMLError:
                pass
            recent = defaultdict(list)
            try:
                values = n.nvmlDeviceGetProcessUtilization(handle, self.stamps.get(index, 0))
                for value in values:
                    # The first query can return historical samples: accept only the recent interval.
                    if value.timeStamp >= (time.time() - self.interval * 2) * 1e6 and 0 <= value.smUtil <= 100:
                        recent[value.pid].append(value.smUtil)
                if values:
                    self.stamps[index] = max(v.timeStamp for v in values)
            except n.NVMLError:
                pass
            for pid, process in relevant.items():
                actor_class = owners[pid][1]
                resident[actor_class] += 1
                used = process.usedGpuMemory
                if used is not None and used != getattr(n, "NVML_VALUE_NOT_AVAILABLE", 2**64 - 1):
                    memory[actor_class] += used
                if recent[pid]:
                    samples[actor_class].append(sum(recent[pid]) / len(recent[pid]))
        return memory, samples, resident, devices

    def sample(self):
        now = time.perf_counter()
        owners = self.processes()
        rows = {name: {"cpu_cores": 0.0, "ram_rss_gb": 0.0, "processes": 0} for name in self.classes}
        previous = {}
        for pid, (process, actor_class) in owners.items():
            try:
                with process.oneshot():
                    key = (pid, process.create_time())
                    cpu = process.cpu_times()
                    seconds = cpu.user + cpu.system
                    rss = process.memory_info().rss
                previous[key] = (now, seconds)
                row = rows[actor_class]
                if key in self.previous:
                    before, used = self.previous[key]
                    row["cpu_cores"] += max(0.0, seconds - used) / max(now - before, 1e-6)
                row["ram_rss_gb"] += rss / 2**30
                row["processes"] += 1
            except psutil.Error:
                pass
        self.previous = previous
        memory, samples, resident, devices = self.gpu(owners)
        for actor_class, row in rows.items():
            row["cpu_host_pct"] = 100 * row["cpu_cores"] / (psutil.cpu_count() or 1)
            row["gpu_processes"] = resident[actor_class]
            row["gpu_sm_sample_coverage"] = len(samples[actor_class]) / resident[actor_class] if resident[actor_class] else 0.0
            if actor_class in memory:
                row["gpu_vram_gb"] = memory[actor_class] / 2**30
            if samples[actor_class]:
                row["gpu_sm_process_mean_pct"] = sum(samples[actor_class]) / len(samples[actor_class])
                row["gpu_sm_process_max_pct"] = max(samples[actor_class])
        elapsed = now - self.origin
        self.file.write(json.dumps({"elapsed_seconds": elapsed, "timestamp": time.time(),
                                    "actors": rows, "devices": devices}) + "\n")
        self.file.flush()
        for actor_class, row in rows.items():
            for name, value in row.items():
                self.writer.add_scalar(f"resource/{actor_class}/{name}", value, int(elapsed))
        for device, row in devices.items():
            for name, value in row.items():
                self.writer.add_scalar(f"device/gpu{device}/{name}", value, int(elapsed))

    def run(self):
        while not self.stop.wait(self.interval):
            try:
                self.sample()
            except Exception as exc:
                self.error = str(exc)
                self.file.write(json.dumps({"timestamp": time.time(), "error": self.error}) + "\n")
                self.file.flush()

    def close(self):
        self.stop.set()
        self.thread.join()
        self.writer.close()
        self.file.close()
        if self.nvml is not None:
            self.nvml.nvmlShutdown()
