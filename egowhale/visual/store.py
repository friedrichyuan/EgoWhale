"""CPU-side writes for visual compute nodes."""

from __future__ import annotations

from pathlib import Path

from egowhale.media import save_masks, write_rgb
from egowhale.visual.depth import _store_depth


def commit(saves: list[dict]) -> None:
    for item in saves:
        path = Path(item["path"])
        kind = item["kind"]
        if kind == "masks":
            save_masks(path, item["data"])
        elif kind == "video":
            from egowhale.visual.paint import paste

            write_rgb(path, paste(item["background"], item["data"], item["box"]), float(item["fps"]))
        elif kind == "depth":
            _store_depth(path, item["data"])
        else:
            raise ValueError(f"unknown save {kind}")
