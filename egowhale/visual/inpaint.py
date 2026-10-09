"""Remove the person with ProPainter. The weights stay on the inpaint actor."""

from __future__ import annotations

from pathlib import Path

from egowhale.media import load_masks, read_rgb, write_rgb
from egowhale.step import INPAINT, MASKS, ROOT, Step, compute_lock, piece_dir, pieces
from egowhale.visual.paint import load_models, paint_clips, paste, prepare_crop

_ROOT = ROOT / "thirdparty" / "propainter"


class Inpaint(Step):
    name = "inpaint"
    needs = (MASKS,)
    makes = (INPAINT,)
    gpus = 1

    def load(self):
        self._held = load_models(str(_ROOT / "weights"))

    def run(self, src: Path, dst: Path) -> None:
        video = Path(src).with_suffix(".mp4")
        if getattr(self, "_held", None) is None:
            self.load()
        spans = pieces(dst)
        for index, (start, stop) in enumerate(spans):
            where = piece_dir(dst, index, len(spans))
            if (where / INPAINT).is_file():
                continue
            frames, fps = read_rgb(video, start, stop)
            masks = load_masks(where / MASKS)
            if len(frames) != len(masks):
                raise ValueError(f"{len(frames)} frames vs {len(masks)} masks")
            clip = prepare_crop(frames, masks)
            painted = paint_clips(self._held, [clip])[0]
            write_rgb(where / INPAINT, paste(clip["background"], painted, clip["box"]), fps)

    def consume(self, payload: dict, budget: int) -> dict:
        """Paint a dataloader batch. Same size and length share one forward. OOM halves the budget."""
        import time

        import ray
        import torch

        items = payload["items"]
        results = [None] * len(items)
        saves = []
        todo = [index for index, item in enumerate(items) if item.get("ok")]
        for index, item in enumerate(items):
            if not item.get("ok"):
                results[index] = (False, 0.0, item.get("error") or "feed failed")
        limit = max(int(budget), 1)
        if getattr(self, "_held", None) is None:
            self.load()

        def run(indexes: list[int]) -> None:
            nonlocal limit
            if not indexes:
                return
            started = time.perf_counter()
            try:
                painted = paint_clips(self._held, [items[index] for index in indexes])
                for index, frames in zip(indexes, painted):
                    saves.append({
                        "kind": "video",
                        "path": items[index]["dst"],
                        "data": frames,
                        "fps": items[index]["fps"],
                        "background": items[index]["background"],
                        "box": items[index]["box"],
                    })
            except Exception as exc:
                oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
                if oom:
                    torch.cuda.empty_cache()
                if oom and len(indexes) > 1:
                    limit = max(1, limit // 2)
                    mid = max(1, len(indexes) // 2)
                    run(indexes[:mid])
                    run(indexes[mid:])
                    return
                message = str(exc).splitlines()[-1]
                for index in indexes:
                    results[index] = (False, 0.0, message)
                return
            share = (time.perf_counter() - started) / len(indexes)
            for index in indexes:
                results[index] = (True, share, "")

        pending = list(todo)
        sizes = {index: int(items[index]["frames"].shape[1]) for index in todo}
        with compute_lock(self):
            while pending:
                group = _slices(pending, [sizes[index] for index in pending], limit)[0]
                run(group)
                pending = pending[len(group) :]
        return {"results": results, "budget": limit, "save_ref": ray.put(saves) if saves else None}


def _slices(indexes: list[int], sizes: list[int], limit: int) -> list[list[int]]:
    """Groups that fit in the frame budget. One episode longer than the budget stays whole."""
    groups: list[list[int]] = []
    current: list[int] = []
    total = 0
    for index, size in zip(indexes, sizes):
        if current and total + size > limit:
            groups.append(current)
            current = []
            total = 0
        current.append(index)
        total += size
    if current:
        groups.append(current)
    return groups
