"""Consolidate the threshold sweep across RANSAC draws.

Two effects are confounded in a single-draw sweep: the threshold, which is the knob
under study, and the inlier draw, which is not reproducible (see ransac_stability.py).
This reads every sweep JSON in the InlierSweep directory and separates them --- spread
across thresholds at a fixed draw, against spread across draws at a fixed threshold.

    python scripts/sweep_summary.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

SWEEP = Path("data/Output_Repro/1903_Stage3/InlierSweep")
KEY = "F1@50m"


def load():
    full, fixed = {}, []
    for f in sorted(SWEEP.glob("inlier_*.json")):
        rows = json.loads(f.read_text())
        m = re.search(r"_s(\d+)\.json$", f.name)
        seed = int(m.group(1)) if m else 250
        if f.name.startswith("inlier_fixed"):
            fixed.extend((seed, r) for r in rows)
        elif len(rows) >= 4:
            full[seed] = rows
    return full, fixed


def main() -> None:
    full, fixed = load()
    thresholds = sorted({r["threshold"] for rows in full.values() for r in rows})

    print(f"Threshold sweep, {KEY}, one column per RANSAC draw")
    print(f"{'thr':>8}" + "".join(f"{'seed '+str(s):>14}" for s in sorted(full))
          + f"{'median':>10}{'range':>14}")
    for thr in thresholds:
        vals, cells = [], ""
        for s in sorted(full):
            r = next((x for x in full[s] if x["threshold"] == thr), None)
            if r and r.get(KEY) is not None:
                vals.append(r[KEY])
                cells += f"{r[KEY]:>9.3f} ({r['inliers']:>2})"
            else:
                cells += f"{'--':>14}"
        if vals:
            print(f"{thr:>8}{cells}{np.median(vals):>10.3f}"
                  f"{max(vals)-min(vals):>14.3f}")

    print("\nSpread across thresholds, within one draw:")
    for s in sorted(full):
        v = [r[KEY] for r in full[s] if r.get(KEY) is not None]
        print(f"  seed {s:<5} {min(v):.3f}-{max(v):.3f}   span {max(v)-min(v):.3f}")

    # every draw at the configured threshold, from both study designs
    at_conf = [r[KEY] for rows in full.values() for r in rows
               if r["threshold"] == 0.01 and r.get(KEY) is not None]
    at_conf += [r[KEY] for _, r in fixed if r.get(KEY) is not None]
    n_conf = [r["inliers"] for rows in full.values() for r in rows
              if r["threshold"] == 0.01] + [r["inliers"] for _, r in fixed]
    a = np.array(at_conf, float)
    print(f"\nSpread across draws, at the configured threshold 0.01 ({len(a)} draws):")
    print(f"  {KEY} median {np.median(a):.3f}   range {a.min():.3f}-{a.max():.3f}"
          f"   span {a.max()-a.min():.3f}")
    print(f"  inliers {min(n_conf)}-{max(n_conf)}")
    same = [v for v, n in zip(at_conf, n_conf) if n == 18]
    if len(same) > 1:
        print(f"  among the {len(same)} draws that returned exactly 18 inliers: "
              f"{min(same):.3f}-{max(same):.3f}, span {max(same)-min(same):.3f}")


if __name__ == "__main__":
    main()
