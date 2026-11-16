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

The `WiLoR` node uses the synchronized Ego2Robot hand implementation in `egowhale/wilor/ego2robot_sync`, configured by `egowhale/wilor/cfgs/wilor_hand_pipeline.yaml`. MP4 input uses Path B with DA3 camera estimation and AnyHand/WiLoR hand prediction; HDF5 input uses Path A hand annotations and its sibling MP4. DynHaMR is disabled by default. Set `dynhamr_enabled: true` to enable reference optimization and its fallback on Path B. The 480-high overlay and reference output live in `action/wilor_ego2robot/`, while `action/hands_wilor.npz` exposes EgoWhale's wrist/thumb/index/middle/ring/little coordinate order. This node defaults to `EgoWhale/thirdparty` and the shared `../.conda-egohand` environment; override `EGOWHALE_WILOR_THIRDPARTY` for a different installation.

For a standalone test, use `bash scripts/wilor.sh <video.mp4|episode.hdf5> [output_dir]`; edit its `GPU_ID` or set it in the environment. HDF5 input requires a same-name MP4. Alternatively edit `scripts/run_wilor_ego2robot.sh`. The older `scripts/run_wilor_video.sh`, `scripts/run_wilor_batch.sh`, and `egowhale.wilor.pipeline` are retained as a legacy experimental implementation and do not run this synchronized node.

For an MP4 directory, edit `GPU_IDS`, `INPUT_DIR`, and `OUTPUT_DIR` in `scripts/run_wilor_ego2robot_batch.sh`, then run it. The default processes all `add_remove_lid` videos with Path B and DA3, one video per GPU at a time. `MAX_VIDEOS=1` tests one clip; `--dry-run` lists the plan. Results are gathered in the batch's `final_videos/`; `results.csv` reports visible frames and failures, and each video directory contains a `process.log` and the EgoWhale hand output.

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
