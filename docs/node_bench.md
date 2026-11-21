# Single-node batch benchmarks

Run from the project root with the ego environment activated. One persistent node processes episodes sequentially. No Ray pool and no automatic upstream execution.

```bash
CUDA_VISIBLE_DEVICES=9 bash scripts/node_bench.sh base_ik --robot arx \
  --episodes datasets/egodex/test_episodes.txt --limit 20 \
  --inputs-root /path/to/existing/run --out outputs/tmp/base_ik_test
```

Select exactly one physical GPU via CUDA_VISIBLE_DEVICES. Render and curate bind EGL to the same physical card. The runner does not select or reserve free GPUs: check availability before starting. Composite can run without a GPU.

Supported stages: retarget, segment, inpaint, depth, base_ik, approach, render, composite, curate.
--episodes accepts HDF5 files, text lists, directories, or multiple such entries. --limit 0 means all.
--robot defaults to arx; piper/dual_piper select two PiPer arms.
--inputs-root uses <task>/<episode>/ layout, identical to pipeline run outputs.
For stages with no upstream files (retarget, segment, inpaint, depth), omit --inputs-root.
Upstream piece boundaries are preserved. Otherwise --piece-frames defaults to 600.
--window sets Depth views per window; --inpaint-model selects sttn or propainter.

Only known existing pipeline artifacts are symlinked from the input root; the selected node's previous outputs are excluded. Outputs are recomputed into a NEW --out directory, preserving source results. Known upstream artifacts are carried forward so this output root can feed the next isolated node benchmark. Missing inputs fail that episode explicitly, and other episodes continue. Any failures produce exit code 1. Existing --out is rejected; no overwrite/skip flag is provided.

## Metrics
results.jsonl records status, frame count/basis, timing and error for every episode.
summary.json records aggregate all_successful and after_warmup statistics.
run_config.json records arguments, resolved episodes and selected CUDA device.

- node_fps = sum(successfully processed frames) / sum(successful node_s). Includes node input reading, computation and output writing. Explicit step.load time is reported separately; initialization performed lazily inside a node remains in node_s.
- after_warmup excludes the first --warmup successful episodes (default 1). They are still run and saved. With no remaining samples, FPS is null.
- compute_fps is available only for action nodes exposing read/compute/save. It includes the entire compute method, including transfers, synchronization inherent in returning CPU data and solver setup. It is not GPU kernel-only FPS. No extra CUDA synchronization was inserted.
- overall_fps = all successful frames / benchmark wall time, including explicit loading, input preparation, failures and cleanup. Interpreter/import startup is outside this timer.
- Approach counts generated prefix frames; Render and Composite count output frames including prefix; other nodes count episode frames. Frame basis is recorded per row. Do not compare these denominators blindly.
- Failed episodes contribute no successful frames. No prior outputs are skipped. Aggregates use summed frames/times, never arithmetic means of per-episode FPS.

This is serial isolated-node throughput. It is not full-pipeline system/fps and does not emulate concurrent Ray actors or prefetched I/O.

## Verification (2026-11-09)
CPU tests cover persistent-node reuse, input preservation, missing input failures, warmup exclusion, weighted metrics and prefix frame counting. Actual Composite execution on two existing episodes completed using CPU only; current GPUs were occupied, so no new GPU-stage benchmark was run. Existing node implementations are reused without modifying their inference or timing internals.

Project changes: egowhale/node_bench.py, scripts/node_bench.sh, tests/test_node_bench.py, this document.

## Optional Approach (2026-11-09)
Render requires IK, base and gripper metadata; action/prefix.npz is optional. Without it, all IK frames including frame zero are rendered. Composite and Curate use zero prefix offset. With a P-frame prefix, Render retains prefix + IK[1:] (N+P-1 frames). Single-node commands need no new flag.
Full bench can set top-level approach: false or pass --no-approach; serial python -m egowhale also accepts --no-approach. Defaults preserve Approach generation. These switches do not delete or ignore an existing prefix; use fresh output directories for a no-Approach run.
See docs/changes/20261109_optional_approach.md for project changes and CPU validation.
