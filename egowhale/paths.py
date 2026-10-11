"""Where the pipeline reads and writes: the repository, its local models and assets, and each episode's files."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Third-party code and weights, kept out of git; see the README for what goes where.
THIRDPARTY = ROOT / "thirdparty"
ASSETS = ROOT / "assets"
OUTPUTS = ROOT / "outputs"

# One episode's files, relative to its output directory. Visual files in PER_PIECE live per piece.
GRIPPER = "action/gripper.npz"
MASKS = "visual/masks.npz"
INPAINT = "visual/inpaint.mp4"
DEPTH = "visual/depth.npz"
BASE = "action/base.json"
IK = "action/ik.npz"
PREFIX = "action/prefix.npz"
RENDER = "visual/render.npz"
COMPOSITE = "visual/composite.mp4"
QUALITY = "action/quality.json"
PIECES = "pieces.json"
PER_PIECE = (MASKS, INPAINT, DEPTH, RENDER)


def default_out(src: Path) -> Path:
    """outputs/<task>/<episode> for an EgoDex episode at <task>/<episode>.hdf5."""
    return OUTPUTS / src.parent.name / src.stem
