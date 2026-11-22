"""Sweep the Stage 4 RANSAC inlier threshold, carried through to matching.

Section 7.4.2 discusses the threshold qualitatively ("relaxing to 150 m increases the
inlier count to about 60...") and Table 1 lists it as delta_inlier = 100 m. Neither
matches the implementation. `filter_outliers_ransac` min-max normalises both point
clouds to [0,1] over their robust bounds and then applies `residual_threshold` in that
space, so the configured 0.01 is **1% of the anchor cloud's extent**, not a distance in
metres. For the 1903 sheet that is roughly 230-330 m depending on axis.

So the sweep varies the parameter as implemented (normalised units) and reports the
metric equivalent alongside, rather than pretending the metre figure is the knob.

Only the anchor step is re-run per threshold: text detection, OCR, name propagation
and intersection processing happen once, upstream. Each setting is then carried
through to a buffer evaluation against the reference, because the downstream
consequence matters more than the inlier count.

Two corrections since the first pass. The RANSAC inside `filter_outliers_ransac` is
not reproducible -- its `_scoped_numpy_seed(250)` seeds numpy's legacy global RNG,
which scikit-image 0.24 ignores -- so an unpinned sweep reports one draw out of a
distribution spanning 13 to 20 inliers at the configured threshold. The generator is
therefore pinned here by monkeypatching the seeded version over it, which is the
one-line fix the module should carry. And the inlier count is taken from the returned
mask rather than parsed out of the log, where a regex over the first parenthesised
number occasionally picked up the wrong message.

    python scripts/inlier_sweep.py --run-dir /path/to/Output_Repro/1903_Stage3
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "modules" / "Similarity", ROOT / "_superceded" / "Similarity",
          ROOT / "modules" / "Geolocation", ROOT / "scripts"):
    sys.path.insert(0, str(p))

PHD = Path("data")
OSM = PHD / "Input/Model/sydney_intersections.geojson"
TURNER = Path("data/GreaterSydneyStreetsEvolution/DATA/Sydney_roads.shp")
BBOX = ("151.045,-34.020", "151.300,-33.717")

# configured value is 0.01; bracket it by a factor of four either way
THRESHOLDS = [0.0025, 0.005, 0.01, 0.02, 0.04]
RANSAC_SEED = 250          # the seed Anchor_Nodes intends but fails to apply
TPS_SMOOTHING = 0.02
MATCH_THRESHOLD = 0.7


def find_network_file(run_dir: Path) -> Path:
    """The intersection-processed graph that the anchor stage consumes."""
    cands = sorted(run_dir.rglob("*.geojson"))
    for pat in ("intersection", "network", "_combined", "propagat"):
        hits = [c for c in cands if pat in c.name.lower()]
        if hits:
            return max(hits, key=lambda f: f.stat().st_size)
    if not cands:
        raise SystemExit(f"no geojson under {run_dir}; has the run finished?")
    return max(cands, key=lambda f: f.stat().st_size)


def extent_metres(run_dir: Path) -> tuple[float, float]:
    """Metric extent of the anchor cloud.

    Taken from the geolocated output, not from the network file: the latter is in
    pixel coordinates, so reprojecting it yields NaN. RANSAC normalises over the
    anchors' robust bounds, which are tighter than this, so treat the metre figures
    as an upper-bound gloss on the normalised threshold rather than an exact
    conversion.
    """
    import geopandas as gpd

    cands = list((run_dir / "Geolocation").glob("*geolocated*.geojson"))
    if not cands:
        return float("nan"), float("nan")
    g = gpd.read_file(max(cands, key=lambda f: f.stat().st_size))
    g = g.set_crs("EPSG:4326", allow_override=True).to_crs("EPSG:7856")
    x0, y0, x1, y1 = g.total_bounds
    return x1 - x0, y1 - y0


def clipped_truth(pred: Path, out: Path) -> Path:
    import geopandas as gpd
    if out.exists():
        return out
    p = gpd.read_file(pred).to_crs("EPSG:7856")
    t = gpd.read_file(TURNER).to_crs("EPSG:7856")
    hull = p.union_all().convex_hull.buffer(100.0)
    sub = t[t.intersects(hull)].copy()
    sub["geometry"] = sub.intersection(hull)
    sub = sub[~sub.is_empty]
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_crs("EPSG:4326").to_file(out, driver="GeoJSON")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path,
                    default=PHD / "Output_Repro/1903_Stage3")
    ap.add_argument("--out-name", default="inlier_sweep.json")
    ap.add_argument("--thresholds", help="comma-separated, overrides the default sweep")
    ap.add_argument("--seed", type=int, default=RANSAC_SEED,
                    help="RANSAC generator seed; vary it to check the ordering is "
                         "not an artefact of one draw")
    a = ap.parse_args()

    import Anchor_Nodes
    from Anchor_Nodes import anchor_pipeline
    from utils import parse_bounding_coordinates   # lives at the repo root
    from Comparison import evaluate_network
    from ransac_stability import seeded_ransac

    # Pin the generator so the table is reproducible, and capture the mask on the
    # way past so the inlier count is read rather than parsed.
    captured = {}
    def _pinned(old_pts, new_pts, thr):
        m = seeded_ransac(old_pts, new_pts, thr, a.seed)
        captured["n"] = int(m.sum())
        captured["total"] = int(len(m))
        return m
    Anchor_Nodes.filter_outliers_ransac = _pinned

    net = find_network_file(a.run_dir)
    print(f"network file: {net}")
    ew, eh = extent_metres(a.run_dir)
    span = max(ew, eh)
    if span != span:   # NaN
        span = 0.0
        print("anchor-cloud extent: unavailable; metre column omitted")
    else:
        print(f"anchor-cloud extent: {ew:,.0f} m x {eh:,.0f} m")
    print("so a normalised threshold t corresponds to roughly t x extent in metres\n")

    out_dir = a.run_dir / "InlierSweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    bpts = parse_bounding_coordinates(*BBOX)

    thresholds = ([float(t) for t in a.thresholds.split(",")] if a.thresholds
                  else THRESHOLDS)
    rows = []
    for thr in thresholds:
        out = out_dir / f"geolocated_thr{thr}_s{a.seed}.geojson"
        buf = io.StringIO()
        # Anchor_Nodes reports the inlier count and tier through `logging`, not
        # print, so redirecting stdout alone captures nothing.
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.DEBUG)
        root = logging.getLogger()
        prev_level = root.level
        root.setLevel(logging.DEBUG)
        root.addHandler(handler)
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                anchor_pipeline(str(net), str(OSM), str(out), bpts, thr,
                                TPS_SMOOTHING, match_threshold=MATCH_THRESHOLD)
        except Exception as e:
            print(f"  thr={thr}: FAILED {type(e).__name__}: {e}")
            continue
        finally:
            root.removeHandler(handler)
            root.setLevel(prev_level)

        log = buf.getvalue()
        inliers = captured.get("n", "?")
        tier = ("TPS" if "TPS Warp" in log else
                "Affine" if "Affine" in log else
                "BoundsFit" if "Bounds Fit" in log else "?")

        row = {"threshold": thr, "approx_m": (round(thr * span) if span else None),
               "inliers": inliers, "method": tier}
        if out.exists():
            truth = clipped_truth(out, out_dir / f"truth_thr{thr}_s{a.seed}.geojson")
            for tol in (25, 50):
                try:
                    df = evaluate_network(str(truth), str(out), output_folder=None,
                                          tolerance_meters=tol)
                    row[f"F1@{tol}m"] = round(float(df.loc["Global_Network"]["F1_Score"]), 3)
                except Exception:
                    row[f"F1@{tol}m"] = None
        rows.append(row)
        am = f"~{row['approx_m']:>5} m" if row['approx_m'] else "     --"
        print(f"  thr={thr:<7} {am}   inliers={row['inliers']:>4}  "
              f"{row['method']:<10} F1@25m={row.get('F1@25m')}  F1@50m={row.get('F1@50m')}")

    (out_dir / a.out_name).write_text(json.dumps(rows, indent=2))
    print(f"\nwritten -> {out_dir/a.out_name}")


if __name__ == "__main__":
    main()
