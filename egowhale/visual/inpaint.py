"""Remove the person. STTN by default; ProPainter is kept as the slower, higher-quality choice.

The weights stay on the inpaint actor. Only one model is loaded per process.
"""

from __future__ import annotations

import time
from pathlib import Path

from egowhale.media import load_masks
from egowhale.paths import INPAINT, MASKS, THIRDPARTY
from egowhale.step import Result, Step, failed_items, run_grouped

MODELS = ("sttn", "propainter")
DEFAULT_MODEL = "sttn"
LABELS = {"sttn": "STTN", "propainter": "ProPainter"}


class Inpaint(Step):
    name = "inpaint"
    needs = (MASKS,)
    makes = (INPAINT,)
    gpus = 1
    per_piece = True

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        super().__init__()
        if model not in MODELS:
            raise ValueError(f"inpaint model must be one of {MODELS}, not {model}")
        self.model = model
        self.held = None
        self._paint = None

    def load(self) -> None:
        if self.held is not None:
            return
        if self.model == "sttn":
            from egowhale.visual import sttn

            self.held, self._paint = sttn.load_model(), sttn.paint_clips
        else:
            from egowhale.visual import paint

            self.held, self._paint = paint.load_models(str(THIRDPARTY / "propainter" / "weights")), paint.paint_clips

    def release(self) -> None:
        self.held = None

    def prepare(self, src: Path, where: Path, start: int, stop: int) -> dict:
        """CPU side of one piece, decoded a chunk at a time. ProPainter paints the mask box; STTN the whole frame."""
        video = Path(src).with_suffix(".mp4")
        masks = load_masks(Path(where) / MASKS)
        if self.model == "sttn":
            from egowhale.visual.sttn import prepare_frames

            clip = prepare_frames(video, start, stop, masks)
        else:
            from egowhale.visual.paint import prepare_crop

            clip = prepare_crop(video, start, stop, masks)
        clip.update(dst=str(Path(where) / INPAINT), source=str(video), span=(start, stop))
        return clip

    def run_piece(self, src: Path, dst: Path, index: int) -> None:
        clip = self.prepare(src, *self.piece(dst, index))
        self.load()
        self.store(self._save(clip, self._paint(self.held, [clip])[0]))

    def consume(self, payload: dict, budget: int | None = None) -> dict:
        """Paint a dataloader batch. ProPainter gives same-size, same-length pieces one forward. OOM halves the budget."""
        import ray

        items = payload["items"]
        results = failed_items(items)
        saves = []
        self.load()

        def forward(group: list[int]) -> None:
            started = time.perf_counter()
            painted = self._paint(self.held, [items[index] for index in group])
            share = (time.perf_counter() - started) / len(group)
            for index, frames in zip(group, painted):
                saves.append(self._save(items[index], frames))
                results[index] = Result(True, share)

        todo = [index for index, result in enumerate(results) if result is None]
        sizes = {index: int(items[index]["frames"].shape[1]) for index in todo}
        with self.lock:
            errors, limit = run_grouped(todo, sizes, budget, forward)
        for index, message in errors.items():
            results[index] = Result(False, 0.0, message)
        return {"results": results, "budget": limit, "save_ref": ray.put(saves) if saves else None}

    def _save(self, clip: dict, painted) -> dict:
        # The writer decodes the source again for ProPainter's background, so only the painted box travels.
        return {"stage": self.name, "dst": clip["dst"], "painted": painted, "box": clip["box"], "source": clip["source"], "span": clip["span"]}

    def store(self, save: dict) -> None:
        from egowhale.visual.paint import write_painted

        write_painted(Path(save["dst"]), save["source"], *save["span"], save["painted"], save["box"])
