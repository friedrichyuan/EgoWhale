#!/usr/bin/env python3
"""Export Retarget inputs from a completed WiLoR job or batch."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from egowhale.wilor.retarget_export import collect_retarget, export_existing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    if (root / "action/hands_wilor.npz").is_file():
        print(export_existing(root))
        return 0
    jobs = sorted(path.parent.parent for path in root.glob("*/action/hands_wilor.npz"))
    if not jobs:
        parser.error(f"No completed WiLoR jobs in {root}")

    def export(job):
        try:
            export_existing(job)
            hdf5 = collect_retarget(root / "retarget_inputs", job.name, job)
            return str(hdf5), ""
        except (KeyError, OSError, ValueError) as error:
            return "", str(error)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(export, jobs))
    with (root / "retarget_exports.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("job", "retarget_hdf5", "error"))
        writer.writerows((job.name, hdf5, error) for job, (hdf5, error) in zip(jobs, results, strict=True))
    failed = sum(bool(error) for _, error in results)
    print(f"{len(jobs) - failed}/{len(jobs)} exported to {root / 'retarget_inputs'}")
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
