"""Compatibility scoped to model subprocesses, without editing third-party code."""
from __future__ import annotations

import inspect


def legacy_mano():
    """Older licensed MANO pickles import chumpy's removed NumPy/inspect names."""
    import numpy as np

    for name, value in {'bool': bool, 'int': int, 'float': float, 'complex': complex,
                        'object': object, 'str': str, 'unicode': str}.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec


def load_detector(checkpoint, device):
    """Old Ultralytics checkpoints need the legacy loader in PyTorch >=2.6.

    This override is limited to the explicitly supplied local detector checkpoint.
    Other files retain torch.load's default behavior.
    """
    from pathlib import Path
    from unittest.mock import patch

    import torch
    from ultralytics import YOLO

    allowed = Path(checkpoint).resolve()
    original = torch.load

    def trusted_load(source, *args, **kwargs):
        if isinstance(source, (str, Path)) and Path(source).resolve() == allowed:
            kwargs.setdefault('weights_only', False)
        return original(source, *args, **kwargs)

    with patch.object(torch, 'load', trusted_load):
        return YOLO(str(allowed)).to(device)
