<p align="center">
  <img src="assets/logo.jpg" alt="EgoWhale" width="200">
</p>

<h1 align="center">EgoWhale</h1>

<p align="center">
  A framework for large-scale synthesis of robot data from egocentric video.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12+-3776AB?logo=python&logoColor=white" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/MuJoCo-3-111111" alt="MuJoCo 3">
</p>

EgoWhale takes in egocentric manipulation video and writes out dual-arm robot demonstrations: retargeted actions, a person-free scene, a depth-composited robot, and a quality report. Stages are independent and skip work that is already on disk. One episode runs in one process; `egowhale.batch` runs a corpus on Ray, with one heavy model held on each GPU.

## Pipeline

| Stage | What it does | Output |
| --- | --- | --- |
| Retarget | Camera-frame parallel gripper, then smooth | `action/gripper.npz` |
| Segment | Person mask (SAM 3) | `visual/masks.npz` |
| Inpaint | Remove the person (STTN; ProPainter optional) | `visual/inpaint.mp4` |
| Depth | Depth of the original video (Depth Anything 3), scaled to meters | `visual/depth.npz` |
| Base + IK | Base pose and arm trajectory (cuRobo) | `action/base.json`, `action/ik.npz` |
| Approach | TrajOpt from the zero configuration to the first end-effector pose | `action/prefix.npz` |
| Render | MuJoCo render of the arm from the action outputs | `visual/render.npz` |
| Composite | Paste the render; gripper hidden where the scene is closer | `visual/composite.mp4` |
| Curate | Frame checks, action statistics, then a VLM audit | `action/quality.json` |

A finished stage is skipped when every file in its output column exists. Delete those files to rerun it. Each stage is one `egowhale.step.Step` subclass with `needs`, `makes`, and a GPU share; its docstring describes the methods batch mode adds.

## Quick start

An episode is an EgoDex HDF5 file with a sibling `.mp4` of the same length. Python 3.12 is required. Rendering uses MuJoCo EGL, which the package sets before MuJoCo is imported.

```bash
python -m egowhale path/to/episode.hdf5 [out_dir] [--steps NAME ...] [--inpaint-model sttn|propainter]
```

When `out_dir` is omitted, files are written to `outputs/<parent>/<stem>/`. `--steps` runs only the named stages.

The independent WiLoR hand experiment (`scripts/run_wilor_video.sh`, default `full` mode) attempts DynHaMR root/smooth and HMP refinement. If HMP fails or its worker exits during that stage, it continues exporting `prediction.mp4` from the existing filtered WiLoR tracks. The hand metadata and `SUCCESS.json` record `optimizer_fallback`, `prior_executed: false`, and the fallback reason. Camera, detection, and root/smooth failures still stop the task. HMP resource checks are optional so missing prior weights can follow this fallback path.

The `WiLoR` node uses the synchronized Ego2Robot hand implementation in `egowhale/wilor/ego2robot_sync`, configured by `egowhale/wilor/cfgs/wilor.yaml`. WiLoR accepts only MP4 input (Path B), using DA3 camera estimation and AnyHand/WiLoR hand prediction. Send annotated EgoDex HDF5 episodes directly to Retarget (`bash scripts/retarget.sh <episode.hdf5> [output_dir]`); the WiLoR node does not read hand annotations. DynHaMR is disabled by default. Set `dynhamr_enabled: true` to enable reference optimization and its fallback on Path B. The 480-high overlay and reference output live in `action/wilor_ego2robot/`, while `action/hands_wilor.npz` exposes EgoWhale's wrist/thumb/index/middle/ring/little coordinate order. This node defaults to `EgoWhale/thirdparty` and the shared `../.conda-egohand` environment; override `EGOWHALE_WILOR_THIRDPARTY` for a different installation.

WiLoR also writes `action/hands_wilor.hdf5` and a sibling `hands_wilor.mp4` link pointing to the source video. Pass this HDF5 to `bash scripts/retarget.sh <hands_wilor.hdf5> [output_dir]`. The HDF5 stores the camera-frame joint positions with identity camera transforms, the source-resolution intrinsics and the source FPS. Missing hands are represented by NaN translations. The DA3 camera trajectory remains in `cam_extrinsics.npz`. Multi-GPU batches gather the HDF5/video pairs in `retarget_inputs/` and list the HDF5 path in `results.csv`. For an earlier job or batch, use `python tools/export_wilor_retarget_hdf5.py <output_dir>` to export from the saved NPZ.

When enabled, the synchronized node runs root/smooth before HMP. The single-video entry point runs HMP only on an optimized sequence of exactly 128 frames. The split-GPU entry point also handles other lengths using overlapping 128-frame windows. It pads the final window as needed and trims the padding when merging. If HMP fails or is skipped, the node checks the completed `smooth_fit` result before publishing it. By default, each previously detected hand must retain at least 80% of its valid frames, all retained 3D points must be finite with positive depth, and at least 50% of the retained 2D keypoints must lie inside the image. Median and 95th-percentile offsets from WiLoR must be at most 5% and 15% of the image diagonal, respectively. The configurable `dynhamr_smooth_*` checks compare against WiLoR. Rejected or missing smooth results fall back to filtered WiLoR tracks. The output metadata records the selected result as `hmp`, `smooth_fit`, or `wilor_filtered`. Root/smooth errors follow the `dynhamr_required` policy. EgoWhale supplies the static-camera loader with `cameras.npz` intrinsics and keeps the extrinsics static during optimization.

For a standalone test, use `bash scripts/wilor.sh <video.mp4> [output_dir]`; edit its `GPU_ID` or set it in the environment. HDF5 input is rejected with instructions to use Retarget. Alternatively edit `scripts/run_wilor_ego2robot.sh`. After changing code/configuration, use a new output directory because completed runs are identified by their code/config signature. The older `scripts/run_wilor_video.sh` and `egowhale.wilor.pipeline` remain a legacy experimental single-video implementation.

For a split-GPU experiment, edit `egowhale/wilor/cfgs/wilor.yaml` and run `bash scripts/run_wilor_split_multigpu.sh`. `dynhamr_enabled` selects `split.profiles.basic` or `split.profiles.refined`. Each physical GPU belongs to one role. After CPU video preparation, DA3 and WiLoR can run concurrently on the same clip. The refined profile also runs SAM3 concurrently and assigns root/smooth and HMP to the optimizer pool. The `long_video` settings control DA3 view limits, SAM3 window lengths, out-of-memory retry limits, and HMP overlap. The default `max_videos: 0` processes all MP4s. Use `--max-videos 5` for a small run or `--resume-run <batch-dir>` to reuse completed stages. Each batch writes a manifest, per-stage logs, annotated videos, `results.csv`, and `throughput.json`.

All WiLoR batch shell entry points now use split scheduling and read the `split` section of `wilor.yaml`. The former same-GPU DA3 + WiLoR resident batch mode is removed. Former environment variable overrides such as `GPU_IDS` are rejected; edit the YAML instead. Single-video debugging remains available separately.

Run `bash scripts/compare_wilor_smooth.sh` to compare one MP4 before and after root/smooth on a single idle GPU. Edit `GPU_ID`, `VIDEO`, and `OUTPUT_DIR` in that script; the default is the 124-frame `paint_clean_brush/14.mp4`, with DA3 and Path B. It runs WiLoR once and reuses the same detections/tracks for both views. Outputs include `wilor_only.mp4`, `smooth_candidate.mp4`, a labeled `comparison.mp4`, the actually selected `selected_result.mp4`, and `comparison.json`. Rejected smooth candidates are labeled as diagnostic and are never presented as accepted results. The comparison excludes differences between the default non-optimization filters and the full branch's tracking/mask filters. Existing completed work can be rendered without rerunning GPU models using `python tools/compare_wilor_smooth.py --video <mp4> --work <run/work> --output <comparison-dir>`.

Model weights stay local:

| Model | Path |
| --- | --- |
| SAM 3 | `thirdparty/sam3` |
| Cutie (from the ProPainter web demo) | `thirdparty/cutie`: `tracker/` and `weights/cutie-base-mega.pth` |
| STTN (default inpainter) | `thirdparty/sttn`, weights in `checkpoints/sttn.pth` |
| ProPainter (optional inpainter) | `thirdparty/propainter` |
| Depth Anything 3 (DA3-GIANT) | `thirdparty/da3` |

The robot assets go in `assets/curobo_arx/` (cuRobo URDF and YAML) and `assets/mujoco_arx_scene/` (MuJoCo scene).

Inpaint uses STTN by default. ProPainter is sharper but about ten times slower; select it with
`--inpaint-model propainter`, or `nodes.inpaint.model: propainter` in a batch config.

## Run one stage

```bash
scripts/<stage>.sh path/to/episode.hdf5 [out_dir]
```

| Script | Reads | Writes |
| --- | --- | --- |
| `scripts/retarget.sh` | episode | `action/gripper.npz` |
| `scripts/segment.sh` | sibling `.mp4` | `visual/masks.npz` |
| `scripts/inpaint.sh` | masks | `visual/inpaint.mp4` |
| `scripts/depth.sh` | sibling `.mp4` | `visual/depth.npz` |
| `scripts/base_ik.sh` | gripper | `action/base.json`, `action/ik.npz` |
| `scripts/approach.sh` | IK | `action/prefix.npz` |
| `scripts/composite.sh` | gripper, inpaint, depth, masks, base, IK, approach | `visual/render.npz`, `visual/composite.mp4` |
| `scripts/curate.sh` | gripper, base, IK, approach, composite | `action/quality.json` |

Each script is `python -m egowhale ... --steps <stage>`. Upstream stages are not rerun. A missing input raises `FileNotFoundError`.

## Run a corpus

```bash
scripts/batch.sh cfgs/bench8_len1000.yaml [--name NAME] [--limit N] [--episodes PATH ...] [--out DIR]
```

A config in `cfgs/` lists only what it changes from `DEFAULTS` in `egowhale/config.py`: episodes, the cards and actors of each node, IO queues, and memory limits. GPU pools must add up to the visible devices. Results go to `<out>/<name>/`, with `batch.jsonl` per episode and TensorBoard in `tb/`. `scripts/bench.sh <out_dir>` runs the 8-GPU throughput benchmark on `outputs/len1000.txt`, the designed mix of 1000 test episodes. `scripts/server_test.sh <out_dir>` runs the whole server acceptance test on that list and packs the logs for review; see its header for options.

## Configuration

The VLM audit in Curate (`egowhale/quality/vlm.py`) calls a Qwen model when a key is present. Without one, or when the call fails, the audit is reported as skipped or errored and the episode is not failed for it.

| Variable | Role | Default |
| --- | --- | --- |
| `EGOWHALE_VLM_API_KEY` | Bearer token. Unset skips the audit. | |
| `EGOWHALE_VLM_BASE_URL` | OpenAI-compatible chat endpoint | DashScope compatible API |
| `EGOWHALE_VLM_MODEL` | Model name | `qwen3.8-flash` |

## Layout

```
egowhale/step.py      Step base class, serial runner, shared batch helpers
egowhale/paths.py     repository paths and each episode's file names
egowhale/pieces.py    splitting long episodes into pieces
egowhale/media.py     video, mask, and depth files; working scale and frame rate
egowhale/robot.py     ARX assets, joint names, MuJoCo scene helpers
egowhale/thirdparty.py  imports from thirdparty/
egowhale/action/      retarget, base search, approach
egowhale/visual/      segmentation, inpaint (STTN, ProPainter), depth, render, composite
egowhale/quality/     curation: frame checks, action statistics, VLM audit
egowhale/batch/       Ray launcher
cfgs/                 batch configs
scripts/              one shell entry per stage, batch and benchmark launchers
tests/                pytest; no GPU needed except the retarget check
tools/                inpaint benchmarks and output comparison
assets/               robot scene and logo
thirdparty/           SAM 3, Cutie, STTN, ProPainter, Depth Anything 3, cuRobo
```

## Acknowledgments

The conversion of egocentric manipulation video into robot demonstrations — action retargeting, arm synthesis, and multi-level quality curation — follows [Ego2Robot](https://arxiv.org/abs/2608.02580) (Wang et al.).

EgoWhale also relies on [SAM 3](https://github.com/facebookresearch/sam3), [Cutie](https://github.com/hkchengrex/Cutie), [STTN](https://github.com/researchmm/STTN), [ProPainter](https://github.com/sczhou/ProPainter), [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3), [cuRobo](https://curobo.org/), and [MuJoCo](https://mujoco.org/).
