"""Resource attribution: shared GPU contexts must not double count, missing samples stay missing."""
import time
from types import SimpleNamespace as Item

from egowhale.batch.monitor import Resources


class Nvml:
    class NVMLError(Exception):
        pass

    NVML_VALUE_NOT_AVAILABLE = -1

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        return [Item(pid=10, usedGpuMemory=1024), Item(pid=20, usedGpuMemory=2048),
                Item(pid=999, usedGpuMemory=4096)]

    nvmlDeviceGetGraphicsRunningProcesses = nvmlDeviceGetComputeRunningProcesses

    def nvmlDeviceGetUtilizationRates(self, handle):
        return Item(gpu=80)

    def nvmlDeviceGetMemoryInfo(self, handle):
        return Item(used=10000)

    def nvmlDeviceGetProcessUtilization(self, handle, stamp):
        return [Item(pid=10, smUtil=70, timeStamp=int(time.time() * 1e6)),
                Item(pid=20, smUtil=20, timeStamp=1)]  # stale sample


def monitor():
    value = Resources.__new__(Resources)
    value.nvml = Nvml()
    value.devices = [0]
    value.stamps = {}
    value.interval = 1
    return value


def test_gpu_grouping_deduplicates_graphics_and_compute_and_excludes_other_jobs():
    value = monitor()
    memory, samples, resident, devices = value.gpu({10: (None, 'DepthActor'), 20: (None, 'DepthActor')})
    assert memory['DepthActor'] == 3072
    assert resident['DepthActor'] == 2
    assert samples['DepthActor'] == [70]
    assert devices['0']['gpu_util_pct'] == 80


def test_gpu_missing_samples_do_not_become_zero_utilization():
    value = monitor()
    def unavailable(*args):
        raise Nvml.NVMLError()
    value.nvml.nvmlDeviceGetProcessUtilization = unavailable
    memory, samples, resident, devices = value.gpu({10: (None, 'DepthActor')})
    assert memory['DepthActor'] == 1024
    assert not samples['DepthActor']
    assert resident['DepthActor'] == 1


def test_cpu_only_monitor_needs_no_nvml():
    value = monitor()
    value.nvml = None
    memory, samples, resident, devices = value.gpu({10: (None, 'FeedActor')})
    assert not memory and not samples and not resident and not devices
