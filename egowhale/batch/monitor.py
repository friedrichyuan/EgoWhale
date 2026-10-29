"""Low-overhead host and physical GPU measurements in the existing system charts."""

from __future__ import annotations

import threading
import time
import warnings
from pathlib import Path

import psutil


class SystemMonitor:
    """Single-host sampler. No per-frame hooks or Ray RPCs during sampling."""

    def __init__(self, writer, origin: float, interval: float = 2):
        self.origin = origin
        self.interval = interval
        self.writer = writer
        self.cpus = psutil.cpu_count() or 1
        self.stop = threading.Event()
        self.nvml = None
        self.devices = []
        self.error = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.devices = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        except Exception as exc:
            self.error = str(exc)
            warnings.warn(f"System GPU monitoring unavailable: {self.error}", stacklevel=2)
        self.thread = threading.Thread(target=self.run, name="system-monitor", daemon=True)

    def start(self):
        self.thread.start()

    def gpu(self):
        n = self.nvml
        devices = {}
        if n is None:
            return devices
        for index, handle in enumerate(self.devices):
            try:
                devices[str(index)] = {"gpu_util_pct": n.nvmlDeviceGetUtilizationRates(handle).gpu,
                                       "vram_gb": n.nvmlDeviceGetMemoryInfo(handle).used / 2**30}
            except n.NVMLError:
                pass
        return devices

    def sample(self):
        step = max(0, int(time.perf_counter() - self.origin))
        cpu = psutil.cpu_times_percent()
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        busy = max(0.0, 100 - cpu.idle - getattr(cpu, "iowait", 0))
        values = {
            "cpu_used_cores": busy * self.cpus / 100,
            "cpu_utilization_pct": busy,
            "cpu_iowait_pct": getattr(cpu, "iowait", 0),
            "memory_used_gb": (memory.total - memory.available) / 2**30,
            "memory_used_pct": 100 * (1 - memory.available / memory.total),
            "memory_available_gb": memory.available / 2**30,
            "memory_shared_gb": getattr(memory, "shared", 0) / 2**30,
            "memory_cached_gb": getattr(memory, "cached", 0) / 2**30,
            "swap_used_gb": swap.used / 2**30,
            "swap_out_gb": swap.sout / 2**30,
        }
        for name in ("cpu", "memory", "io"):
            try:
                pressure = Path(f"/proc/pressure/{name}").read_text()
                values[f"{name}_pressure_pct"] = float(pressure.split("avg10=")[1].split()[0])
            except (OSError, ValueError, IndexError):
                pass
        for name, value in values.items():
            self.writer.add_scalar(f"system/{name}", value, step)
        for device, row in self.gpu().items():
            self.writer.add_scalar(f"system/gpu{device}/memory_used_gb", row["vram_gb"], step)
            self.writer.add_scalar(f"system/gpu{device}/utilization_pct", row["gpu_util_pct"], step)

    def run(self):
        psutil.cpu_times_percent()  # Prime CPU deltas in the sampling thread.
        while not self.stop.wait(self.interval):
            try:
                self.sample()
            except Exception as exc:
                self.error = str(exc)
                warnings.warn(f"System monitoring sample failed: {self.error}", stacklevel=2)

    def close(self):
        self.stop.set()
        self.thread.join()
        if self.nvml is not None:
            self.nvml.nvmlShutdown()
