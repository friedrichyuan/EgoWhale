"""Human-readable, truthful frame/window counters on model-worker stderr."""
import sys


def report(label: str, completed: int, total: int) -> None:
    if total > 0 and (completed <= 1 or completed == total or completed % max(1, total // 20) == 0):
        print(f'[{label}] {completed}/{total} ({completed / total:.1%})', file=sys.stderr, flush=True)
