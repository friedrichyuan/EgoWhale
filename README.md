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

EgoWhale takes in egocentric manipulation video and writes out dual-arm robot demonstrations: retargeted actions, a person-free scene, a depth-composited robot, and a quality report. Stages are independent, skip work that is already on disk, and are shaped so a later launcher can run them as Ray actors across a large corpus.

## Pipeline

| Stage | What it does | Output |
| --- | --- | --- |
| Retarget | Camera-frame parallel gripper, then smooth | `action/gripper.npz` |
| Segment | Person mask (SAM 3) | `visual/masks.npz` |
| Inpaint | Remove the person (ProPainter) | `visual/inpaint.mp4` |
| Depth | Depth of the original video (Depth Anything 3), scaled to meters | `visual/depth.npz` |
| Base + IK | Base pose and arm trajectory (cuRobo) | `action/base.json`, `action/ik.npz` |
| Approach | TrajOpt from the zero configuration to the first end-effector pose | `action/prefix.npz` |
| Render | MuJoCo render of the arm from the action outputs | `visual/render.npz` |
| Composite | Paste the render; gripper hidden where the scene is closer | `visual/composite.mp4` |
| Curate | Frame checks, action statistics, then a VLM audit | `action/quality.json` |

A finished stage is skipped when every file in its output column exists. Delete those files to rerun it. Each stage is one class with `needs`, `makes`, and a GPU hint.

## Quick start

An episode is an EgoDex HDF5 file with a sibling `.mp4`. Python 3.12 is required. Rendering uses MuJoCo EGL, which the entry point sets before MuJoCo is imported.

```bash
python -m egowhale path/to/episode.hdf5 [out_dir]
```

When `out_dir` is omitted, files are written to `outputs/<parent>/<stem>/`.

Model weights stay local:

| Model | Path |
| --- | --- |
| SAM 3 | `thirdparty/sam3` |
| Cutie (from the ProPainter web demo) | `thirdparty/cutie` |
| ProPainter | `thirdparty/propainter` |
| Depth Anything 3 (DA3-GIANT) | `thirdparty/da3` |

## Run one stage

```bash
scripts/<stage>.sh path/to/episode.hdf5 [out_dir]
```

| Script | Reads | Writes |
| --- | --- | --- |
| `scripts/retarget.sh` | episode | `action/gripper.npz` |
| `scripts/segment.sh` | sibling `.mp4` | `visual/masks.npz` |
| `scripts/inpaint.sh` | masks | `visual/inpaint.mp4` |
| `scripts/depth.sh` | inpaint | `visual/depth.npz` |
| `scripts/base_ik.sh` | gripper | `action/base.json`, `action/ik.npz` |
| `scripts/approach.sh` | IK | `action/prefix.npz` |
| `scripts/composite.sh` | gripper, inpaint, depth, masks, base, IK, approach | `visual/composite.mp4` |
| `scripts/curate.sh` | gripper, base, IK, approach, composite | `action/quality.json` |

Upstream stages are not rerun. A missing input raises `FileNotFoundError`.

## Configuration

The VLM audit in Curate calls Qwen3.5 when a key is present. Without one, that audit is skipped and the episode is not failed for it.

| Variable | Role | Default |
| --- | --- | --- |
| `EGOWHALE_VLM_API_KEY` | Bearer token. Unset skips the audit. | |
| `EGOWHALE_VLM_BASE_URL` | OpenAI-compatible chat endpoint | DashScope compatible API |
| `EGOWHALE_VLM_MODEL` | Model name | `qwen3.5-plus` |

## Layout

```
egowhale/action/    retarget, base search, approach, curation
egowhale/visual/    segmentation, inpaint, depth, composite
scripts/            one shell entry per stage
assets/             robot scene and logo
thirdparty/         SAM 3, Cutie, ProPainter, Depth Anything 3
```

## Acknowledgments

The conversion of egocentric manipulation video into robot demonstrations — action retargeting, arm synthesis, and multi-level quality curation — follows [Ego2Robot](https://arxiv.org/abs/2608.02580) (Wang et al.).

EgoWhale also relies on [SAM 3](https://github.com/facebookresearch/sam3), [Cutie](https://github.com/hkchengrex/Cutie), [ProPainter](https://github.com/sczhou/ProPainter), [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3), [cuRobo](https://curobo.org/), and [MuJoCo](https://mujoco.org/).
