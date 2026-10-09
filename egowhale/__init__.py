"""EgoDex video to robot frames: retarget, segment, inpaint, base+IK, composite."""

from .step import run, run_remote

__all__ = ["run", "run_remote"]
