"""Hardware measurements use physical host counters, not Ray resource reservations."""
from types import SimpleNamespace as Item

import psutil
import pytest

from egowhale.batch.monitor import SystemMonitor


class Writer:
    def __init__(self):
        self.values = {}

    def add_scalar(self, tag, value, step):
        self.values[tag] = (value, step)


class Nvml:
    class NVMLError(Exception):
        pass

    def nvmlDeviceGetMemoryInfo(self, device):
        return Item(used=(device + 1) * 10 * 2**30)

    def nvmlDeviceGetUtilizationRates(self, device):
        return Item(gpu=(device + 1) * 30)


def monitor(monkeypatch):
    import time

    value = SystemMonitor.__new__(SystemMonitor)
    value.writer, value.origin, value.cpus = Writer(), time.perf_counter(), 8
    value.nvml, value.devices = Nvml(), [0, 1]
    monkeypatch.setattr(psutil, 'cpu_times_percent', lambda: Item(idle=40, iowait=10))
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: Item(total=100*2**30, available=30*2**30,
                                                            shared=5*2**30, cached=10*2**30))
    monkeypatch.setattr(psutil, 'swap_memory', lambda: Item(used=2**30, sout=2*2**30))
    return value


def test_cpu_cores_and_memory_use_available_memory(monkeypatch):
    value = monitor(monkeypatch)
    value.sample()
    rows = value.writer.values
    assert rows['system/cpu_used_cores'][0] == 4
    assert rows['system/cpu_utilization_pct'][0] == 50
    assert rows['system/cpu_iowait_pct'][0] == 10
    assert rows['system/memory_used_gb'][0] == 70
    assert rows['system/memory_used_pct'][0] == 70
    assert rows['system/memory_available_gb'][0] == 30
    assert all(tag.startswith('system/') for tag in rows)


def test_gpu_counters_are_separate_physical_devices(monkeypatch):
    value = monitor(monkeypatch)
    value.sample()
    assert value.writer.values['system/gpu0/memory_used_gb'][0] == 10
    assert value.writer.values['system/gpu1/memory_used_gb'][0] == 20
    assert value.writer.values['system/gpu1/utilization_pct'][0] == 60


def test_unavailable_gpu_samples_are_omitted_not_zeroed(monkeypatch):
    value = monitor(monkeypatch)
    def unavailable(device):
        raise Nvml.NVMLError()
    value.nvml.nvmlDeviceGetMemoryInfo = unavailable
    value.sample()
    assert 'system/gpu0/memory_used_gb' not in value.writer.values
    assert 'system/cpu_used_cores' in value.writer.values


def test_nvml_initialization_failure_keeps_error_and_cpu_monitoring(monkeypatch):
    import sys
    import time

    def unavailable():
        raise RuntimeError("NVML unavailable")

    monkeypatch.setitem(sys.modules, "pynvml", Item(nvmlInit=unavailable))
    with pytest.warns(UserWarning, match="NVML unavailable"):
        value = SystemMonitor(Writer(), time.perf_counter())
    assert value.error == "NVML unavailable"
    assert value.nvml is None
    value.sample()
    assert "system/cpu_used_cores" in value.writer.values


def test_sampling_failure_keeps_error_and_continues(monkeypatch):
    import threading

    value = monitor(monkeypatch)
    value.stop = threading.Event()
    value.interval = 0
    attempts = []

    def sample():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("sample unavailable")
        value.stop.set()

    monkeypatch.setattr(value, "sample", sample)
    with pytest.warns(UserWarning, match="sample unavailable"):
        value.run()
    assert value.error == "sample unavailable"
    assert len(attempts) == 2


def test_report_records_live_system_metrics_without_resource_files(tmp_path, monkeypatch):
    import time

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from egowhale.batch import report

    original = report.SystemMonitor
    monkeypatch.setattr(report, 'SystemMonitor', lambda writer, origin: original(writer, origin, interval=0.05))
    recorded = report.Report(tmp_path, {})
    try:
        time.sleep(0.3)
    finally:
        recorded.close()
    board = EventAccumulator(str(tmp_path / 'tb')).Reload()
    assert 'system/cpu_used_cores' in board.Tags()['scalars']
    assert 'system/memory_used_gb' in board.Tags()['scalars']
    assert not any(tag.startswith(('resource/', 'device/')) for tag in board.Tags()['scalars'])
    assert not list(tmp_path.glob('resources*'))
    assert not (tmp_path / 'tb' / 'resources').exists()
