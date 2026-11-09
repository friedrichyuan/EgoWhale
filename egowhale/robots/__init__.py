"""Explicit robot selection shared by serial nodes and Ray actors."""
from .base import Robot


def load_robot(robot="arx"):
    if isinstance(robot, Robot):
        return robot
    if robot == "arx":
        from .arx import ARXRobot
        return ARXRobot()
    if robot in ("piper", "dual_piper"):
        from .piper import PiperRobot
        return PiperRobot()
    raise ValueError(f"Unknown robot {robot!r}; choose arx or piper")
