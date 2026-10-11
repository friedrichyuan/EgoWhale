"""The Step base class, the serial runner, and the helpers every stage's batch path shares."""

from __future__ import annotations

import contextlib
import gc
import io
import threading
import time
from pathlib import Path
from typing import ClassVar, NamedTuple

from egowhale.pieces import missing, piece_dir, pieces, plan


class Result(NamedTuple):
    """One unit of work in batch mode: whether it computed (a skip did not), its seconds, and why it failed."""

    ran: bool
    seconds: float = 0.0
    error: str = ""


class Step:
    """One stage of the episode DAG.

    A subclass sets `name`, the files it `needs` and `makes` (names from egowhale.paths), and `gpus`, the share of
    a card it holds in batch mode. `run(src, dst)` does the whole episode in one process. A stage with
    `per_piece` set implements `run_piece` instead, and `run` paints the pieces still missing.

    Weights are held from `load()` to `release()`; both may be called any number of times, and every compute
    path calls `load()` first. Batch mode adds one of two shapes:

    - Visual GPU stages: `prepare(src, dst, start, stop)` builds one piece's input on a CPU reader,
      `consume(payload, budget)` computes a batch of them under `lock` and returns {"results", "budget",
      "save_ref"}, and `store(save)` writes one result on a CPU writer.
    - Action stages: `read(src, dst)` on a CPU reader, `compute(data)` on the GPU, `save(dst, out)` on a writer.
    """

    name: ClassVar[str] = ""
    needs: ClassVar[tuple[str, ...]] = ()
    makes: ClassVar[tuple[str, ...]] = ()
    gpus: ClassVar[float] = 0
    per_piece: ClassVar[bool] = False

    def __init__(self) -> None:
        # One GPU section at a time. A batch actor copies results out after releasing it.
        self.lock = threading.Lock()

    def load(self) -> None:
        """Acquire weights. A no-op when they are held, or when the stage has none."""

    def release(self) -> None:
        """Drop weights so the next stage of a serial run gets the card."""

    def run(self, src: Path, dst: Path) -> None:
        if not self.per_piece:
            raise NotImplementedError(f"{type(self).__name__}.run")
        for index in self.todo(dst):
            self.run_piece(Path(src), Path(dst), index)

    def run_piece(self, src: Path, dst: Path, index: int) -> None:
        raise NotImplementedError(f"{type(self).__name__}.run_piece")

    def piece(self, dst: Path, index: int) -> tuple[Path, int, int]:
        """Where piece `index` keeps its visual files, and its [start, stop) frames."""
        spans = pieces(dst)
        start, stop = spans[index]
        return piece_dir(dst, index, len(spans)), start, stop

    def todo(self, dst: Path) -> list[int]:
        """Pieces with an output still missing."""
        spans = pieces(dst)
        done = [all((piece_dir(dst, index, len(spans)) / name).is_file() for name in self.makes) for index in range(len(spans))]
        return [index for index, finished in enumerate(done) if not finished]


def run(src: Path | str, dst: Path | str, steps: tuple[type[Step] | Step, ...]) -> None:
    """Single process, one GPU stage at a time. A step is a class, or an instance built with its options."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    plan(src, dst)
    for given in steps:
        step = given if isinstance(given, Step) else given()
        if not missing(dst, step.makes):
            print(f"skip {step.name}")
            continue
        absent = missing(dst, step.needs)
        if absent:
            raise FileNotFoundError(f"{step.name} missing {absent}")
        print(f"run {step.name}")
        try:
            step.run(src, dst)
        finally:
            step.release()
            if step.gpus:
                release_gpu()
        absent = missing(dst, step.makes)
        if absent:
            raise FileNotFoundError(f"{step.name} did not write {absent}")


def execute(step: Step, src: Path | str, dst: Path | str, index: int | None = None) -> Result:
    """Run one stage, or one piece of it, with its prints kept quiet. A finished unit is skipped, not run."""
    src, dst = Path(src), Path(dst)
    where = dst if index is None else piece_dir(dst, index, len(pieces(dst)))
    outputs = [where / name for name in step.makes]
    if all(path.is_file() for path in outputs):
        return Result(False)
    absent = missing(dst, step.needs)
    if absent:
        raise FileNotFoundError(f"{step.name} missing {absent}")
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        if index is None:
            step.run(src, dst)
        else:
            step.run_piece(src, dst, index)
    absent = [str(path) for path in outputs if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"{step.name} did not write {absent}")
    return Result(True, time.perf_counter() - started)


def release_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.set_default_dtype(torch.float32)
    except ImportError:
        pass


def short_error(exc: BaseException) -> str:
    """The last line of an exception's message, or its type when the message is empty."""
    lines = str(exc).strip().splitlines()
    return lines[-1] if lines else type(exc).__name__


def is_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    return "out of memory" in str(exc).lower()


def failed_items(items: list[dict]) -> list[Result | None]:
    """A batch's results so far: each item the reader could not prepare has failed; the rest are open."""
    return [None if item.get("ok") else Result(False, 0.0, item.get("error") or "feed failed") for item in items]


def run_grouped(positions: list[int], sizes: dict[int, int], budget: int | None, forward) -> tuple[dict[int, str], int]:
    """Call `forward(group)` on consecutive groups whose sizes fit in `budget`; one oversized item runs alone.

    No budget puts every position in one group. A group that runs out of GPU memory is halved and retried, and
    the budget stays halved for the groups after it. The retry starts only after the exception is gone: its
    traceback holds the failed forward's tensors, so retrying from inside the `except` block would run without
    that memory. Returns the error per failed position, and the budget that is left.
    """
    errors: dict[int, str] = {}
    limit = max(int(budget if budget is not None else sum(sizes[position] for position in positions)), 1)

    def attempt(group: list[int]) -> None:
        nonlocal limit
        try:
            forward(group)
            return
        except Exception as exc:
            oom, message = is_oom(exc), short_error(exc)
        if oom:
            import torch

            torch.cuda.empty_cache()
            if len(group) > 1:
                limit = max(1, limit // 2)
                middle = len(group) // 2
                attempt(group[:middle])
                attempt(group[middle:])
                return
        for position in group:
            errors[position] = message

    pending = list(positions)
    while pending:
        group = groups_within(pending, [sizes[position] for position in pending], limit)[0]
        attempt(group)
        pending = pending[len(group) :]
    return errors, limit


def groups_within(indexes: list[int], sizes: list[int], limit: int) -> list[list[int]]:
    """Consecutive groups whose sizes add up to at most `limit`. One item larger than the limit stays alone."""
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
