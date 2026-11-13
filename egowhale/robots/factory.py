"""One constructor policy for CLI stages and all Ray actor roles."""
ROBOT_STAGES = frozenset(("base_ik", "approach", "render", "curate"))


def make_step(cls, robot="arx", *, ik_method="base_ik", batch_size=0):
    if cls.name == "base_ik":
        return cls(robot=robot, ik_method=ik_method, batch_size=batch_size)
    return cls(robot=robot) if cls.name in ROBOT_STAGES else cls()
