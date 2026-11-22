"""How reproducible is the Stage 4 inlier set?

`filter_outliers_ransac` wraps its RANSAC call in `_scoped_numpy_seed(250)`, whose
docstring says the seed exists so the filter is reproducible. It is not. scikit-image
0.24's `ransac` draws from an `rng` argument that defaults to a fresh
`np.random.default_rng()`, which ignores numpy's legacy global state entirely, so
seeding `np.random` has no effect on it. Every call therefore returns a different
inlier set.

That matters beyond tidiness. The Stage 4 tier decision is a threshold on the inlier
count -- fewer than 3 falls back to a bounds fit, fewer than 10 to a global affine, 10
or more to a TPS warp -- so a configuration sitting near a boundary can take a
different registration path on different runs of the same input.

This script measures the spread directly: N draws per threshold, reporting the median,
the range, and how often the tier decision flips. It also checks that passing an
explicit generator restores determinism, which is the one-line fix.

    python scripts/ransac_stability.py --draws 40
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
for p in (ROOT, ROOT / "modules" / "Geolocation"):
    sys.path.insert(0, str(p))

PHD = Path("data")
NET = PHD / "Output_Repro/1903_Stage3/Geolocation/graph_network_combined.geojson"
OSM = PHD / "Input/Model/sydney_intersections.geojson"
OUT = PHD / "Output_Repro/RansacStability"
BBOX = ("151.045,-34.020", "151.300,-33.717")
THRESHOLDS = [0.0025, 0.005, 0.01, 0.02, 0.04]


def tier(n: int) -> str:
    return "BoundsFit" if n < 3 else ("Affine" if n < 10 else "TPS")


def anchors():
    import geopandas as gpd
    from Anchor_Nodes import create_rectangle_from_points, pre_scale_gdf
    from utils import parse_bounding_coordinates

    gdf = gpd.read_file(NET)
    modern = gpd.read_file(OSM)
    if modern.crs is None:
        modern.set_crs("EPSG:4326", inplace=True)
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    gdf = gdf.to_crs(modern.crs)

    rect = create_rectangle_from_points(parse_bounding_coordinates(*BBOX))
    sub = modern[modern.geometry.intersects(rect)]
    mm = {str(r.get("intersection_name", "")).strip().lower(): r.geometry.coords[0]
          for _, r in sub.iterrows() if r.get("intersection_name")}

    a_old, a_new = [], []
    for _, r in gdf.iterrows():
        n = str(r.get("canonical_name", "")).strip().lower()
        if n in mm and r.get("confidence", 0) > 0.7:
            a_old.append(r.geometry.coords[0])
            a_new.append(mm[n])
    a_old, a_new = np.array(a_old), np.array(a_new)
    _, scaled_old = pre_scale_gdf(gdf.copy(), a_old, a_new)
    return scaled_old, a_new


def seeded_ransac(old_pts, new_pts, thr, seed):
    """`filter_outliers_ransac` with the generator actually pinned."""
    from skimage.measure import ransac
    from skimage.transform import AffineTransform
    from Anchor_Nodes import get_robust_bounds

    o_min, o_max = get_robust_bounds(old_pts)
    n_min, n_max = get_robust_bounds(new_pts)
    no = (np.clip(old_pts, o_min, o_max) - o_min) / (o_max - o_min + 1e-9)
    nn = (np.clip(new_pts, n_min, n_max) - n_min) / (n_max - n_min + 1e-9)
    _, inliers = ransac((no, nn), AffineTransform, min_samples=3,
                        residual_threshold=thr, max_trials=5000,
                        rng=np.random.default_rng(seed))
    return np.zeros(len(old_pts), bool) if inliers is None else inliers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=40)
    a = ap.parse_args()

    from Anchor_Nodes import filter_outliers_ransac

    old, new = anchors()
    print(f"raw name matches: {len(old)}\n")
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for thr in THRESHOLDS:
        counts = np.array([int(filter_outliers_ransac(old, new, thr).sum())
                           for _ in range(a.draws)])
        tiers = [tier(c) for c in counts]
        modal = max(set(tiers), key=tiers.count)
        rows.append({
            "threshold": thr, "draws": a.draws,
            "median": int(np.median(counts)), "min": int(counts.min()),
            "max": int(counts.max()), "iqr": [int(np.percentile(counts, 25)),
                                              int(np.percentile(counts, 75))],
            "modal_tier": modal,
            "tier_flip_rate": round(1 - tiers.count(modal) / len(tiers), 3),
            # determinism check with the generator pinned
            "seeded_repeat": [int(seeded_ransac(old, new, thr, 250).sum())
                              for _ in range(3)],
        })
        r = rows[-1]
        print(f"  thr={thr:<7} median={r['median']:>3}  range {r['min']}-{r['max']}"
              f"  IQR {r['iqr'][0]}-{r['iqr'][1]}   tier={r['modal_tier']:<10}"
              f" flips={r['tier_flip_rate']:.0%}   seeded={r['seeded_repeat']}")

    (OUT / "ransac_stability.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwritten -> {OUT/'ransac_stability.json'}")


if __name__ == "__main__":
    main()
