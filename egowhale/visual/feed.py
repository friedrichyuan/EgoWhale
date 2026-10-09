"""CPU dataloader. Decodes the next visual batch while a GPU is busy."""

from __future__ import annotations

from pathlib import Path

from egowhale.media import load_masks, read_rgb
from egowhale.step import DEPTH, INPAINT, MASKS


def prepare(stage: str, units: list[tuple[str, str, int, int]]) -> dict:
    """`units` are (episode, piece dir, start, stop). Only the piece's frames are decoded."""
    items = []
    for src, dst, start, stop in units:
        try:
            items.append(_one(stage, Path(src), Path(dst), int(start), int(stop)))
        except Exception as exc:
            items.append({"ok": False, "error": str(exc).splitlines()[-1]})
    return {"items": items}


def _one(stage: str, src: Path, dst: Path, start: int, stop: int) -> dict:
    frames, fps = read_rgb(src.with_suffix(".mp4"), start, stop)
    if stage == "segment":
        return {"ok": True, "frames": frames, "dst": str(dst / MASKS)}
    if stage == "inpaint":
        from egowhale.visual.paint import prepare_crop

        masks = load_masks(dst / MASKS)
        if len(frames) != len(masks):
            raise ValueError(f"{len(frames)} frames vs {len(masks)} masks")
        clip = prepare_crop(frames, masks)
        clip["ok"] = True
        clip["fps"] = float(fps)
        clip["dst"] = str(dst / INPAINT)
        return clip
    if stage == "depth":
        from egowhale.visual.depth import prepare_view

        view = prepare_view(src, frames, start)
        view["ok"] = True
        view["dst"] = str(dst / DEPTH)
        return view
    raise ValueError(f"no feed for {stage}")
