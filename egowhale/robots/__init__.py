"""Explicit robot selection shared by serial nodes and Ray actors."""
from .base import Robot

ROBOT_NAMES = ("arx", "piper", "dual_piper", "franka", "ur5e")


def load_robot(robot="arx"):
    if isinstance(robot, Robot):
        return robot
    if robot == "arx":
        from .arx import ARXRobot
        return ARXRobot()
    if robot in ("piper", "dual_piper"):
        from .piper import PiperRobot
        return PiperRobot()
    if robot in ("franka", "ur5e"):
        from .twin import FrankaRobot, UR5eRobot
        return FrankaRobot() if robot == "franka" else UR5eRobot()
    raise ValueError(f"Unknown robot {robot!r}; choose from {', '.join(ROBOT_NAMES)}")
