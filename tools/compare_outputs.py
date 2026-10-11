"""Compare two pipeline output trees file by file: python tools/compare_outputs.py <before> <after>.

Arrays and videos are compared numerically; JSON field by field. Prints one line per file, worst first.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from egowhale.media import load_masks, read_rgb  # noqa: E402


def _arrays(path: Path) -> dict[str, np.ndarray]:
    if path.name.endswith("masks.npz"):
        return {"masks": load_masks(path)}
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _diff(a: np.ndarray, b: np.ndarray) -> str:
    if a.shape != b.shape:
        return f"shape {a.shape} vs {b.shape}"
    if a.dtype == bool:
        return f"{int((a != b).sum())} of {a.size} differ"
    a, b = a.astype(np.float64), b.astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    if (np.isfinite(a) != np.isfinite(b)).any():
        return f"non-finite pattern differs ({int((np.isfinite(a) != np.isfinite(b)).sum())})"
    if not finite.any():
        return "equal"
    delta = np.abs(a - b)[finite]
    return "equal" if delta.max() == 0 else f"max {delta.max():.3g} mean {delta.mean():.3g}"


def _json(a, b, where: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append(f"{where}{key}: only in {'after' if key not in a else 'before'}")
            else:
                out.extend(_json(a[key], b[key], f"{where}{key}."))
        return out
    if isinstance(a, list) and isinstance(b, list) and a and all(isinstance(v, (int, float)) for v in a + b):
        if len(a) != len(b):
            return [f"{where[:-1]}: length {len(a)} vs {len(b)}"]
        flips = sum(x != y for x, y in zip(a, b))
        return [f"{where[:-1]}: {flips} of {len(a)} differ"] if flips else []
    if isinstance(a, float) and isinstance(b, float):
        return [] if abs(a - b) <= 1e-9 * max(1.0, abs(a)) else [f"{where[:-1]}: {a:.6g} vs {b:.6g}"]
    return [] if a == b else [f"{where[:-1]}: {a!r} vs {b!r}"[:200]]


def compare(before: Path, after: Path) -> list[tuple[str, str]]:
    names = sorted({p.relative_to(before) for p in before.rglob("*") if p.is_file()} | {p.relative_to(after) for p in after.rglob("*") if p.is_file()})
    rows = []
    for name in names:
        a, b = before / name, after / name
        if not a.is_file() or not b.is_file():
            rows.append((str(name), f"only in {'before' if a.is_file() else 'after'}"))
            continue
        if name.suffix == ".npz":
            left, right = _arrays(a), _arrays(b)
            parts = [f"{key}: {_diff(left[key], right[key])}" for key in sorted(set(left) & set(right))]
            parts += [f"{key}: only one side" for key in sorted(set(left) ^ set(right))]
            rows.append((str(name), "; ".join(parts)))
        elif name.suffix == ".mp4":
            rows.append((str(name), _diff(read_rgb(a)[0], read_rgb(b)[0])))
        elif name.suffix == ".json":
            found = _json(json.loads(a.read_text()), json.loads(b.read_text()))
            rows.append((str(name), "; ".join(found) if found else "equal"))
        else:
            rows.append((str(name), "equal" if a.read_bytes() == b.read_bytes() else "bytes differ"))
    return rows


def main() -> None:
    before, after = Path(sys.argv[1]), Path(sys.argv[2])
    for name, verdict in compare(before, after):
        print(f"{name:<52} {verdict}")


if __name__ == "__main__":
    main()
