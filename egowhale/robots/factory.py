"""One constructor policy for CLI stages and all Ray actor roles."""
ROBOT_STAGES = frozenset(("base_ik", "approach", "render", "curate"))


def make_step(cls, robot="arx"):
    return cls(robot=robot) if cls.name in ROBOT_STAGES else cls()
