"""Third-party code under thirdparty/, which stays unmodified. Each model's import fixes live here, named."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from egowhale.paths import THIRDPARTY


def use(name: str, *inner: str) -> Path:
    """Put thirdparty/<name>/<inner...> on sys.path once, so its top-level packages import. Returns that path."""
    path = THIRDPARTY.joinpath(name, *inner)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def stub_evo() -> None:
    """DA3 imports evo while building its Gaussian head, which the pipeline does not use."""
    if "evo.core.trajectory" in sys.modules:
        return
    evo = types.ModuleType("evo")
    core = types.ModuleType("evo.core")
    trajectory = types.ModuleType("evo.core.trajectory")
    trajectory.PosePath3D = type("PosePath3D", (), {})
    sys.modules["evo"] = evo
    sys.modules["evo.core"] = core
    sys.modules["evo.core.trajectory"] = trajectory


def use_da3() -> None:
    use("da3", "src")
    stub_evo()


def use_sam3() -> None:
    use("sam3")
    # SAM 3 imports psutil. Where only Ray's vendored copy is installed, that copy is used.
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    vendored = Path(sys.prefix, "lib", version, "site-packages", "ray", "thirdparty_files")
    if vendored.is_dir() and str(vendored) not in sys.path:
        sys.path.insert(0, str(vendored))
