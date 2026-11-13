"""Transport-independent streaming episode IK; finished episodes release their slots."""

from egowhale.action import base_ik as core
from egowhale.robots import load_robot

# Frames per actor RPC; stop sooner whenever an episode completes. Not a wait timeout.
FRAME_QUANTUM = 8


def episode_capacity(robot, batch_size):
    robot = load_robot(robot) if isinstance(robot, str) else robot
    cost = sum(len(core._offset_grid(g.bounds)) for g in robot.groups)
    if type(batch_size) is not int or batch_size < cost or batch_size % cost:
        raise ValueError(f"batch_size must be a positive multiple of {cost} for {robot.name}")
    return batch_size // cost


class EpisodeStream:
    """Bounded slots keyed by caller tokens. Knows neither Ray nor paths nor writers."""

    def __init__(self, robot, batch_size, backend=None):
        self.capacity = episode_capacity(robot, batch_size)
        if backend is None:
            from egowhale.action.sequential_ik import CuroboBackend

            backend = CuroboBackend(robot, self.capacity)
        self.backend = backend
        self.active = {}

    def advance(self, items=(), steps=FRAME_QUANTUM):
        items = list(items)
        tokens = [token for token, _ in items]
        if steps < 1 or len(items) > self.capacity - len(self.active):
            raise ValueError("invalid step count or stream capacity exceeded")
        if len(set(tokens)) != len(tokens) or any(token in self.active for token in tokens):
            raise ValueError("duplicate live episode token")
        completed = []
        for token, data in items:
            try:
                self.active[token] = self.backend.prepare(data)
            except Exception as exc:
                completed.append(dict(token=token, ok=False, error=str(exc)))
        for _ in range(steps):
            if not self.active:
                break
            try:
                self.backend.step(list(self.active.values()))
            except Exception as exc:
                # A shared solver failure invalidates all its live states. Never silently restart them.
                completed.extend(dict(token=t, ok=False, error=str(exc)) for t in self.active)
                self.active.clear()
                self.backend.reset()
                break
            finished = [t for t, s in self.active.items() if s["frame"] == s["length"]]
            for token in finished:
                state = self.active.pop(token)
                try:
                    completed.append(dict(token=token, ok=True, out=self.backend.finish(state)))
                except Exception as exc:
                    completed.append(dict(token=token, ok=False, error=str(exc)))
            if completed:
                break  # Let caller write results/refill immediately, never wait for longer episodes.
        return completed

    def close(self):
        self.active.clear()
        self.backend.reset()


def iter_results(robot, items, batch_size):
    """Finite standalone input stream, yields each completion before reading its replacement."""
    stream = EpisodeStream(robot, batch_size)
    source = iter(items)
    exhausted = False
    try:
        while not exhausted or stream.active:
            arrivals = []
            for _ in range(stream.capacity - len(stream.active)):
                try:
                    arrivals.append(next(source))
                except StopIteration:
                    exhausted = True
                    break
            yield from stream.advance(arrivals)
    finally:
        stream.close()


def compute_batch(robot, items, batch_size):
    # Compatibility API returns input order; streaming callers use EpisodeStream/iter_results.
    rows = {}
    for row in iter_results(robot, enumerate(items), batch_size):
        if not row["ok"]:
            raise RuntimeError(row["error"])
        rows[row["token"]] = row["out"]
    return [rows[i] for i in range(len(items))]


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="piper", choices=("arx", "piper", "dual_piper"))
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument(
        "--inputs", type=Path, nargs="+", required=True, help="episode output directories containing action/gripper.npz"
    )
    parser.add_argument("--out", type=Path, required=True, help="new output root, grouped by task/episode")
    args = parser.parse_args()
    robot = load_robot(args.robot)
    episode_capacity(robot, args.batch_size)
    keys = [Path(p.parent.name) / p.name for p in args.inputs]
    if len(set(keys)) != len(keys):
        parser.error("duplicate task/episode names")
    args.out.mkdir(parents=True, exist_ok=False)

    def inputs():
        for index, path in enumerate(args.inputs):
            yield index, core.read_gripper(path)

    for row in iter_results(robot, inputs(), args.batch_size):
        if not row["ok"]:
            raise RuntimeError(f"{args.inputs[row['token']]}: {row['error']}")
        src, key, result = args.inputs[row["token"]], keys[row["token"]], row["out"]
        dst = args.out / key
        core.save_base(dst, result)
        # Carry forward prerequisites for independent Approach/Render, preserving originals.
        import shutil

        shutil.copy2(src / core.GRIPPER, dst / core.GRIPPER)
        if (src / "pieces.json").is_file():
            shutil.copy2(src / "pieces.json", dst / "pieces.json")
        print(f"wrote {dst}", flush=True)


if __name__ == "__main__":
    main()
