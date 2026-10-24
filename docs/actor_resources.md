# Actor resource monitoring

`bash start.sh [limit] [episodes] [out]` enables one-second sampling by default.
For direct batch commands set `EGOWHALE_RESOURCE_MONITOR=1`; set it to `0` to disable.
Optional dependencies: `pip install -e '.[monitor]'` (psutil and nvidia-ml-py).

Open `tensorboard --logdir <out>/tb --port 6006`, forward port 6006 in VS Code,
then open http://localhost:6006. Select **Custom Scalars → Actor resources**.
Each metric chart combines one curve per actor class. The **Scalars** tab also
provides `resource/<ActorClass>/<metric>`; select the `resources` run. The x axis
is seconds since the same origin as the existing FPS metrics. Data is flushed
every five seconds and can be inspected both during and after the run.

Files:

- `resources.jsonl`: timestamp, elapsed seconds, actor-class aggregates and device fallback metrics.
- `resources_meta.json`: PID/class registration and measurement definitions.
- `tb/resources`: TensorBoard events and the custom dashboard layout.

Metric definitions:

| Metric | Meaning |
| --- | --- |
| cpu_cores | Sum of process CPU-time deltas divided by wall time. 2.0 means two busy logical CPU cores. |
| cpu_host_pct | cpu_cores divided by host logical CPU count, times 100. Ray CPU reservations are not hard limits. |
| ram_rss_gb | Sum of resident memory. Shared Ray object-store mappings can be counted repeatedly. |
| gpu_vram_gb | Sum of NVML per-process memory across devices, deduplicating compute/graphics contexts by PID on each device. |
| gpu_sm_process_mean_pct | Mean of fresh NVML process-device SM activity samples in the class. Not an additive share of GPU capacity. |
| gpu_sm_process_max_pct | Maximum of those process-device sample averages. |
| gpu_sm_sample_coverage | Fraction of resident GPU process-device pairs that supplied fresh SM samples. Read this beside mean/max. |
| gpu_processes | Number of GPU process-device pairs in this class. |
| processes | Number of live registered processes and descendants sampled. |

CPU samples include live descendants such as ffmpeg; very short children that
exit between samples may be missed. The initial CPU sample primes the counters.
NVML values can be unavailable, especially for EGL graphics contexts; missing
values are omitted rather than written as zero. The mean may be biased toward
active processes when coverage is low. Device-level utilization is provided as
a fallback under `device/gpuN`; do not sum process percentages to infer a card's
utilization. Idle GPU memory can remain allocated even when SM activity is low.

Sampling starts after actor/model initialization. It covers steady-state work
and tail drain, not model-loading peaks. The implementation targets this
single-host batch deployment. Sampling is outside the per-frame path, with no
Ray RPCs after initial PID registration. Full-run sampling overhead has not yet
been benchmarked.

The old custom method trace and `ray_timeline.json` export have been removed.
Actor task-event reporting is disabled, and start.sh disables RAY_PROFILING.
The old EGOWHALE_RAY_TIMELINE switch no longer enables recording.

Existing runs with only gpu.csv and Ray events cannot reconstruct per-class
CPU or GPU memory history. These resource metrics require a new run.
