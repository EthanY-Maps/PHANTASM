"""Affine versus TPS registration accuracy, across RANSAC draws.

`affine_only_baseline.py` pins one anchor set and compares the two transforms on it.
That is the right thing for the end-to-end arm, where both must see identical anchors,
but a single draw is exactly what `ransac_stability.py` shows cannot be trusted: the
inlier set for the 1903 sheet varies from 13 to 20 points between identical calls.

So the registration comparison is repeated over many draws. For each, both models are
fitted on the same inliers and scored two ways:

  in-sample      residual at the anchors the model was fitted on. This is what
                 Section 7.4 reports, and it flatters the more flexible model by
                 construction: TPS can bend to reach its own control points, an
                 affine cannot.
  leave-one-out  refit on n-1, predict the held-out anchor. Neither model has seen
                 the target, so this is the comparison that carries information.

    python scripts/warp_residual_sweep.py --seeds 25
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "modules" / "Geolocation", ROOT / "scripts"):
    sys.path.insert(0, str(p))

PHD = Path("data")
OUT = PHD / "Output_Repro/AffineOnly"
THRESHOLD = 0.01


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=25)
    a = ap.parse_args()

    from ransac_stability import anchors, seeded_ransac
    from affine_only_baseline import residuals

    old, new = anchors()
    rows = []
    for seed in range(a.seeds):
        mask = seeded_ransac(old, new, THRESHOLD, seed)
        if mask.sum() < 5:
            continue
        r = residuals(old[mask], new[mask])
        rows.append({"seed": seed, "inliers": int(mask.sum()),
                     "affine": r["affine"], "tps": r["tps"]})
        print(f"  seed {seed:>2}  n={int(mask.sum()):>3}  "
              f"affine {r['affine']['in_sample_median_m']:>7.1f}/{r['affine']['loo_median_m']:>7.1f}   "
              f"tps {r['tps']['in_sample_median_m']:>7.1f}/{r['tps']['loo_median_m']:>7.1f}  (in/LOO m)",
              flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "warp_residual_sweep.json").write_text(json.dumps(rows, indent=2))

    def col(model, key):
        return np.array([r[model][key] for r in rows if r[model][key] is not None], float)

    print(f"\n{'='*78}\nOver {len(rows)} RANSAC draws at threshold {THRESHOLD}\n{'='*78}")
    print(f"{'':<10}{'in-sample median (m)':>26}{'leave-one-out median (m)':>28}")
    for m in ("affine", "tps"):
        i, l = col(m, "in_sample_median_m"), col(m, "loo_median_m")
        print(f"{m:<10}{np.median(i):>14.1f}  [{i.min():.1f}-{i.max():.1f}]"
              f"{np.median(l):>16.1f}  [{l.min():.1f}-{l.max():.1f}]")

    print(f"\n{'':<10}{'in-sample mean (m)':>26}{'leave-one-out mean (m)':>28}")
    for m in ("affine", "tps"):
        i, l = col(m, "in_sample_mean_m"), col(m, "loo_mean_m")
        print(f"{m:<10}{np.median(i):>14.1f}  [{i.min():.1f}-{i.max():.1f}]"
              f"{np.median(l):>16.1f}  [{l.min():.1f}-{l.max():.1f}]")

    ai, ti = col("affine", "in_sample_median_m"), col("tps", "in_sample_median_m")
    al, tl = col("affine", "loo_median_m"), col("tps", "loo_median_m")
    print(f"\nTPS better in-sample:      {int((ti < ai).sum())}/{len(ai)} draws"
          f"   (median gain {np.median(ai - ti):+.1f} m)")
    print(f"TPS better leave-one-out:  {int((tl < al).sum())}/{len(al)} draws"
          f"   (median gain {np.median(al - tl):+.1f} m)")
    print(f"\nwritten -> {OUT/'warp_residual_sweep.json'}")


if __name__ == "__main__":
    main()
