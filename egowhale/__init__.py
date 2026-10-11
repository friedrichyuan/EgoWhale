"""EgoDex video to dual-arm robot demonstrations: retarget, segment, inpaint, depth, base+IK, approach, render,
composite, and curate."""

import os

# MuJoCo picks its GL backend when it is first imported, and every stage that renders imports this package first.
os.environ.setdefault("MUJOCO_GL", "egl")

from .step import Result, Step, run  # noqa: E402

__all__ = ["Result", "Step", "run"]
