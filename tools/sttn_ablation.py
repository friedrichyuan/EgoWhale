"""Summarise the STTN precision / slicing ablation from outputs/inpaint_bench/results/sttn_*/metrics.json.

Seam error: temporal error on frame pairs that straddle a chunk boundary (t = k*chunk - 1), against the rest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "outputs" / "inpaint_bench" / "results"
VARIANTS = ["sttn_fp16", "sttn_fp32", "sttn_fp32_c120", "sttn_fp32_c60", "sttn_fp32_c30", "sttn_fp32_c60_o10", "sttn_fp32_allrefs"]


def seams(row: dict) -> tuple[float, float]:
    error = np.array(row["syn_pair_error"])
    pixels = np.array(row["syn_pair_pixels"])
    chunk = row.get("chunk")
    mask = np.zeros(len(error), dtype=bool)
    if chunk:
        mask[chunk - 1 :: chunk] = True
    seam = error[mask].sum() / max(pixels[mask].sum(), 1) if mask.any() else float("nan")
    rest = error[~mask].sum() / max(pixels[~mask].sum(), 1)
    return seam, rest


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("variants", nargs="*")
    parser.add_argument("--base", default="sttn_fp32", help="variant the per-episode PSNR change is measured against")
    args = parser.parse_args()
    variants = args.variants or VARIANTS
    base = {r["episode"]: r for r in json.loads((RESULTS / args.base / "metrics.json").read_text())}
    print("| Variant | fps real | fps syn | Peak GB | PSNR | SSIM | LPIPS | Temporal | Seam / non-seam temporal | ΔPSNR vs base (worst ep) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name in variants:
        path = RESULTS / name / "metrics.json"
        if not path.is_file():
            continue
        rows = json.loads(path.read_text())
        weight = np.array([r["frames"] for r in rows], float)

        def avg(key: str, rows=rows, weight=weight) -> float:
            return float(np.average([r[key] for r in rows], weights=weight))

        frames = weight.sum()
        real = frames / sum(r["real_seconds"] for r in rows) if all("real_seconds" in r for r in rows) else float("nan")
        syn = frames / sum(r["syn_seconds"] for r in rows)
        peak = max(r["syn_peak_gb"] for r in rows)
        seam_e = seam_p = rest_e = rest_p = 0.0
        for r in rows:
            error, pixels, chunk = np.array(r["syn_pair_error"]), np.array(r["syn_pair_pixels"]), r.get("chunk")
            mask = np.zeros(len(error), bool)
            if chunk:
                mask[chunk - 1 :: chunk] = True
            seam_e += error[mask].sum()
            seam_p += pixels[mask].sum()
            rest_e += error[~mask].sum()
            rest_p += pixels[~mask].sum()
        seam_text = f"{seam_e / seam_p:.2f} / {rest_e / rest_p:.2f}" if seam_p else f"– / {rest_e / rest_p:.2f}"
        deltas = [r["syn_psnr"] - base[r["episode"]]["syn_psnr"] for r in rows]
        worst = min(range(len(deltas)), key=lambda i: deltas[i])
        print(
            f"| {name} | {real:.1f} | {syn:.1f} | {peak:.1f} | {avg('syn_psnr'):.2f} | {avg('syn_ssim'):.4f} | {avg('syn_lpips'):.4f} | "
            f"{avg('syn_temporal'):.2f} | {seam_text} | {deltas[worst]:+.2f} ({rows[worst]['episode'].split('/')[0]}) |"
        )


if __name__ == "__main__":
    main()
