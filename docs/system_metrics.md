# System hardware metrics

Run `bash start.sh` normally. The existing TensorBoard run records physical host
and GPU measurements every two seconds under `system/`, with the same elapsed
seconds as the FPS charts. Open **Scalars** and filter `system/`. GPU indices are
physical NVML indices; all installed GPUs are measured, including other users'
cards. The monitor has one background thread and no separate CLI or output file.
TensorBoard flushes at least every five seconds.

| Tag | Meaning |
| --- | --- |
| system/cpu_used_cores | Busy logical core equivalents across the host, excluding I/O wait. |
| system/cpu_utilization_pct | Busy CPU time / total host CPU time. |
| system/cpu_iowait_pct | Host CPU time accounted as waiting for I/O. |
| system/memory_used_gb | (MemTotal - MemAvailable) in GiB; keeps the previous tag's meaning. |
| system/memory_used_pct | The same amount as a percentage of total RAM. |
| system/memory_available_gb | RAM the OS estimates available for allocation. |
| system/memory_shared_gb | Shared memory, including Ray plasma/tmpfs pages. |
| system/memory_cached_gb | OS-reported file cache. |
| system/swap_used_gb, system/swap_out_gb | Swap occupied, and cumulative bytes swapped out in GiB. |
| system/cpu_pressure_pct, system/memory_pressure_pct, system/io_pressure_pct | Linux PSI `some avg10`, when available. Higher values indicate more time with stalled work. |
| system/gpuN/memory_used_gb | Physical GPU N's used VRAM in GiB. |
| system/gpuN/utilization_pct | Physical GPU N's kernel-active time percentage from NVML. |

High memory use alone does not demonstrate memory pressure. Check available RAM,
memory PSI, swap activity, I/O wait and I/O PSI together with FPS. Host statistics
include all processes; these curves cannot by themselves attribute memory to a
specific pipeline stage. Sampling starts after actors have initialized.

New runs produce no `resources.jsonl`, `resources_meta.json`, `tb/resources`,
Actor resource curves or Perfetto export. Previous run files are kept unchanged.
TensorBoard normally flushes periodically; it can be viewed during the run or later.


