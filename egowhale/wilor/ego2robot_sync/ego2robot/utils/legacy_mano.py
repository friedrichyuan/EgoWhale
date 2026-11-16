"""Compatibility for the chumpy objects inside the released MANO pickle."""


def prepare_mano_loading():
    import inspect
    import numpy as np

    # chumpy's released source predates Python 3.11 and NumPy 1.24.
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    aliases = {"bool": bool, "int": int, "float": float, "complex": complex,
               "object": object, "str": str, "unicode": str}
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)
