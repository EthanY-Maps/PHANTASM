"""Affine-only registration carried through Stage 5.

Section 7.4 reports the affine-to-TPS improvement as a registration result, measured
at the anchor points. That is not a baseline: it never
reaches the matching stage, so it says nothing about what the non-rigid warp is worth
end to end. There is a second problem. TPS with a smoothing
of 0.02 very nearly interpolates its own control points, so its in-sample residual is
small by construction. Comparing it against an affine fit's in-sample residual compares
a model that reproduces the anchors against one that cannot, which is a property of the
model families rather than evidence about the map.

This script reports three things on the 1903 sheet, the only sheet whose automated
anchors survive Stage 4:

  in-sample      residual at the anchors, as the manuscript reports it
  leave-one-out  refit on n-1 anchors, predict the held-out one; the honest
                 registration comparison, since neither model has seen the target
  end-to-end     both geolocations carried through Stage 5 and scored against the
                 same reference

Everything upstream of the tier decision is shared: the same network file, the same
OCR-derived anchors, the same pre-scaling and the same RANSAC inlier set. Only the
transform applied to the graph changes, so the difference is attributable.

    python scripts/affine_only_baseline.py            # register + Stage 5
    python scripts/affine_only_baseline.py --fit-only
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
for p in (ROOT, ROOT / "modules" / "Similarity", ROOT / "_superceded" / "Similarity",
          ROOT / "modules" / "Geolocation"):
    sys.path.insert(0, str(p))
sys.path.insert(0, str(ROOT / "scripts"))

PHD = Path("data")
RUN = PHD / "Output_Repro/1903_Stage3"
NET = RUN / "Geolocation/graph_network_combined.geojson"
OSM_INTERSECTIONS = PHD / "Input/Model/sydney_intersections.geojson"
OSM_ROADS = PHD / "Input/Model/Sydney/sydney_drive.graphml"
OUT = PHD / "Output_Repro/AffineOnly"
BBOX = ("151.045,-34.020", "151.300,-33.717")

RANSAC_THRESHOLD = 0.01      # normalised, as configured
RANSAC_SEED = 250            # the seed Anchor_Nodes intends but fails to apply
TPS_SMOOTHING = 0.02
MATCH_THRESHOLD = 0.7
METRIC = "EPSG:7856"


def to_metres(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Residual magnitudes in metres, both clouds being lon/lat."""
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", METRIC, always_xy=True)
    px, py = tr.transform(pred[:, 0], pred[:, 1])
    tx, ty = tr.transform(target[:, 0], target[:, 1])
    return np.hypot(np.asarray(px) - np.asarray(tx), np.asarray(py) - np.asarray(ty))


def affine_predict(src, dst, query):
    from skimage.transform import AffineTransform
    m = AffineTransform()
    if not m.estimate(src, dst):
        return None
    q = np.column_stack([query, np.ones(len(query))])
    out = (m.params @ q.T).T
    return out[:, :2]


def tps_predict(src, dst, query, smoothing=TPS_SMOOTHING):
    from scipy.interpolate import RBFInterpolator
    _, keep = np.unique(src, axis=0, return_index=True)
    interp = RBFInterpolator(src[keep], dst[keep], kernel="thin_plate_spline",
                            smoothing=smoothing)
    return interp(query)


def residuals(c_old, c_new) -> dict:
    """In-sample and leave-one-out residuals for both transform families."""
    out = {}
    for name, fn in (("affine", affine_predict), ("tps", tps_predict)):
        pred = fn(c_old, c_new, c_old)
        r = to_metres(pred, c_new) if pred is not None else np.array([np.nan])
        loo = []
        for i in range(len(c_old)):
            keep = np.arange(len(c_old)) != i
            p = fn(c_old[keep], c_new[keep], c_old[i:i + 1])
            if p is None:
                continue
            loo.append(to_metres(p, c_new[i:i + 1])[0])
        loo = np.array(loo, float)
        out[name] = {
            "n_anchors": int(len(c_old)),
            "in_sample_median_m": round(float(np.median(r)), 2),
            "in_sample_p90_m": round(float(np.percentile(r, 90)), 2),
            # mean and max as well, because §7.4.2 reports the affine-to-TPS
            # improvement in those terms (2,634 m -> 120 m mean, 4,143 m -> 252 m max)
            # and they behave very differently from the median on this tail.
            "in_sample_mean_m": round(float(np.mean(r)), 2),
            "in_sample_max_m": round(float(np.max(r)), 2),
            "loo_median_m": round(float(np.median(loo)), 2) if len(loo) else None,
            "loo_p90_m": round(float(np.percentile(loo, 90)), 2) if len(loo) else None,
            "loo_mean_m": round(float(np.mean(loo)), 2) if len(loo) else None,
            "loo_max_m": round(float(np.max(loo)), 2) if len(loo) else None,
        }
    return out


def register() -> dict:
    """Rebuild the anchor stage, then emit the affine and TPS geolocations."""
    import geopandas as gpd
    from Anchor_Nodes import (create_rectangle_from_points, pre_scale_gdf,
                              apply_affine_transform, apply_tps_transform)
    from ransac_stability import seeded_ransac
    from utils import parse_bounding_coordinates

    OUT.mkdir(parents=True, exist_ok=True)
    original = gpd.read_file(NET)
    gdf = original.copy()
    modern = gpd.read_file(OSM_INTERSECTIONS)
    if modern.crs is None:
        modern.set_crs("EPSG:4326", inplace=True)
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    gdf = gdf.to_crs(modern.crs)

    rect = create_rectangle_from_points(parse_bounding_coordinates(*BBOX))
    subset = modern[modern.geometry.intersects(rect)].copy()
    modern_map = {str(r.get("intersection_name", "")).strip().lower(): r.geometry.coords[0]
                  for _, r in subset.iterrows() if r.get("intersection_name")}

    a_old, a_new = [], []
    for _, r in gdf.iterrows():
        name = str(r.get("canonical_name", "")).strip().lower()
        if name in modern_map and r.get("confidence", 0) > MATCH_THRESHOLD:
            a_old.append(r.geometry.coords[0])
            a_new.append(modern_map[name])
    a_old, a_new = np.array(a_old), np.array(a_new)
    print(f"raw name matches: {len(a_old)}")

    scaled, a_old_scaled = pre_scale_gdf(gdf.copy(), a_old, a_new)
    # `filter_outliers_ransac` is advertised as seeded but is not: scikit-image 0.24
    # ignores numpy's legacy global RNG, so the inlier count wanders between 13 and 20
    # on this input (scripts/ransac_stability.py). Both arms have to see the SAME
    # anchor set or the comparison measures the draw rather than the transform, so the
    # generator is pinned here.
    mask = seeded_ransac(a_old_scaled, a_new, RANSAC_THRESHOLD, RANSAC_SEED)
    c_old, c_new = a_old_scaled[mask], a_new[mask]
    print(f"RANSAC inliers: {int(mask.sum())} of {len(mask)} (seed {RANSAC_SEED})")

    res = residuals(c_old, c_new)
    for k, v in res.items():
        print(f"  {k:<7} in-sample {v['in_sample_median_m']:>7.2f} m   "
              f"LOO {v['loo_median_m']:>8.2f} m")

    paths = {}
    aff = apply_affine_transform(scaled.copy(), c_old, c_new)
    if aff is None:
        raise SystemExit("affine fit failed on the inlier set")
    paths["affine"] = OUT / "1903_affine_geolocated.geojson"
    aff.to_file(paths["affine"], driver="GeoJSON")

    tps = apply_tps_transform(scaled.copy(), c_old, c_new, TPS_SMOOTHING)
    paths["tps"] = OUT / "1903_tps_geolocated.geojson"
    tps.to_file(paths["tps"], driver="GeoJSON")

    out = {"inliers": int(mask.sum()), "raw_matches": int(len(a_old)),
           "ransac_seed": RANSAC_SEED, "residuals": res,
           "paths": {k: str(v) for k, v in paths.items()}}
    (OUT / "registration.json").write_text(json.dumps(out, indent=2))
    return out


def stage5(arm: str, hist_path: Path, truth_path: Path) -> dict:
    """Stage 5 on one geolocation, scored against the shared reference."""
    from HistoricMapMatcher import HistoricMapMatcher
    from load_osmnx import load_osmnx
    from Comparison import evaluate_network
    from run_stage5_bypass import build_config

    print(f"\n{'='*68}\n[1903 / {arm}] Stage 5\n{'='*68}", flush=True)
    G_hist, G_osm = load_osmnx(str(hist_path), str(OSM_ROADS))
    matcher = HistoricMapMatcher(G_hist, G_osm, build_config(tuned=True))
    results = matcher.run_pipeline(n_jobs=1)

    run_dir = OUT / arm
    run_dir.mkdir(parents=True, exist_ok=True)
    matcher.export_results(str(run_dir), results)

    row = {"arm": arm}
    summary = run_dir / "match_summary.json"
    if summary.exists():
        row.update(json.loads(summary.read_text()))

    pred = run_dir / "combined_high_medium.geojson"
    if pred.exists() and pred.stat().st_size > 100:
        df = evaluate_network(str(truth_path), str(pred), output_folder=None)
        if not df.empty:
            g = df.loc["Global_Network"]
            row.update(recall=round(float(g["Recall"]), 4),
                       precision=round(float(g["Precision"]), 4),
                       f1=round(float(g["F1_Score"]), 4))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-only", action="store_true")
    a = ap.parse_args()

    reg = register()
    if a.fit_only:
        return

    # The reference is the one the semi-automatic 1903 arm was scored against:
    # Turner et al. clipped to the hull of the GCP-registered graph. Both arms
    # are scored against that same clip, so the recall denominator is the map's
    # true footprint rather than wherever each transform happened to put it.
    truth = PHD / "Output_Repro/Stage5/Sydney_1903/truth_clipped_fullhull.geojson"
    if not truth.exists():
        raise SystemExit(f"missing shared reference: {truth}")

    rows = []
    for arm in ("affine", "tps"):
        try:
            rows.append(stage5(arm, Path(reg["paths"][arm]), truth))
        except Exception as e:
            import traceback
            print(f"[FAIL] {arm}: {type(e).__name__}: {e}")
            traceback.print_exc()
        (OUT / "affine_only_results.json").write_text(
            json.dumps({"registration": reg, "stage5": rows}, indent=2))

    print(f"\n{'='*70}\n{'arm':<10}{'total':>8}{'acc':>7}{'recall':>9}{'prec':>9}{'F1':>9}")
    for r in rows:
        f = lambda k: f"{r[k]:9.3f}" if isinstance(r.get(k), float) else f"{'--':>9}"
        print(f"{r['arm']:<10}{r.get('total',0):>8}{r.get('accepted',0):>7}"
              f"{f('recall')}{f('precision')}{f('f1')}")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
