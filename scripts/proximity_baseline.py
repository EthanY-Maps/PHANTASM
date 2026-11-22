"""Proximity-only matching: what do the geometric terms buy over 'nearest link'?

The composite score needs a baseline to be measured against. The
obvious one is proximity: drop the shape, bearing, sinuosity, topology and flow
terms, drop the A* stitching and the confidence tiers, and assign every historic
link to the OSM link nearest to it. Whatever the full matcher scores above that
is what the scoring machinery is actually worth.

The comparison is like-for-like by construction. The OSM edge set, the motorway
filter, the intersection consolidation, the metric CRS and the historic edge set
all come from `HistoricMapMatcher.__init__`, so the baseline sees exactly the
network the matcher sees. Only the choice of which edge to take changes.

Two knobs are swept rather than fixed, because either choice on its own would be
arguable:

  registration  The matcher's Pass 0 fits an affine warp from KDTree node probes
                before scoring. Running the baseline with and without it splits
                "what the warp buys" from "what the score buys".
  rule          "Nearest" is ambiguous, and the naive reading is a straw man.
                Minimum distance from the link to the edge picks the crossing
                street whenever a link ends on a junction, so a second rule is
                run alongside it: among all edges within the radius, take the one
                minimising the mean distance from points sampled along the link,
                i.e. the edge the link most nearly lies along. That still uses
                nothing but proximity, but it is the rule a competent baseline
                would use.
  geometry      The matcher trims its chosen path to the projections of the
                source link's endpoints. Trimming is the like-for-like
                convention; the untrimmed whole edge has the opposite bias,
                better recall and worse precision.

All four combinations are evaluated and the strongest F1 is reported, so the
baseline is stated at its best rather than at its most convenient.

A radius cap is applied at the matcher's own `initial_search_radius`; links whose
nearest edge is further away are left unmatched, exactly as the matcher would
find no candidate.

    python scripts/proximity_baseline.py --all
    python scripts/proximity_baseline.py --sheet Annandale --no-warp
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "modules" / "Similarity", ROOT / "_superceded" / "Similarity"):
    sys.path.insert(0, str(p))
sys.path.insert(0, str(ROOT / "scripts"))

from run_stage5_bypass import (BYPASS, OSM_PATH, OVERRIDES, RESULTS, SHEETS,
                               TRUTH, build_config, clip_truth)

OUT = Path("data/Output_Repro/Proximity")

# Fraction of the source link's length below which a trimmed match is treated as
# degenerate. Reported, not silently dropped.
DEGENERATE = 0.25


def _sample(geom, n=10):
    """Points evenly spaced along a link, used to score how well an edge follows it."""
    import numpy as np
    return [geom.interpolate(d, normalized=True) for d in np.linspace(0, 1, n)]


def assign(matcher, edge_data, radius: float):
    """Proximity assignment under both rules, with and without endpoint trimming.

    Returns a dict of convention -> list of geometries, plus per-link records.
    """
    import numpy as np
    from shapely.geometry import LineString
    from shapely.ops import substring

    idx = matcher.edge_index
    geoms = [it["geom"] for it in edge_data]
    nearest = np.atleast_1d(idx.tree.nearest(geoms))

    # Candidate sets for the collinear rule: everything within the radius of
    # either endpoint, which is the same gate the matcher's Pass 1 applies.
    s_cands = idx.bulk_query([it["start_pt"] for it in edge_data], radius)
    e_cands = idx.bulk_query([it["end_pt"] for it in edge_data], radius)

    out = {k: [] for k in ("nearest_trim", "nearest_full",
                           "collinear_trim", "collinear_full")}
    rows = []
    n_deg = 0
    n_differ = 0

    def trim(eg, it):
        a, b = eg.project(it["start_pt"]), eg.project(it["end_pt"])
        seg = substring(eg, min(a, b), max(a, b))
        if not isinstance(seg, LineString) or seg.length < DEGENERATE * it["len"]:
            return seg if isinstance(seg, LineString) and seg.length > 0 else eg, True
        return seg, False

    for i, it in enumerate(edge_data):
        j_near = int(nearest[i])
        eg_near = idx.geometries[j_near]
        d_near = it["geom"].distance(eg_near)
        if d_near > radius:
            continue

        # collinear rule: minimise mean distance from sampled points on the link
        cand = set(int(c) for c in s_cands[i]) | set(int(c) for c in e_cands[i])
        cand.add(j_near)
        pts = _sample(it["geom"])
        j_col, d_col = j_near, float("inf")
        for c in cand:
            g = idx.geometries[c]
            m = float(np.mean([g.distance(p) for p in pts]))
            if m < d_col:
                j_col, d_col = c, m
        eg_col = idx.geometries[j_col]
        if j_col != j_near:
            n_differ += 1

        g_nt, deg = trim(eg_near, it)
        g_ct, _ = trim(eg_col, it)
        n_deg += deg
        out["nearest_trim"].append(g_nt)
        out["nearest_full"].append(eg_near)
        out["collinear_trim"].append(g_ct)
        out["collinear_full"].append(eg_col)

        rows.append({"source_id": str(it["edge"]),
                     "nearest_edge": str(idx.edge_keys[j_near]),
                     "collinear_edge": str(idx.edge_keys[j_col]),
                     "min_dist_m": round(float(d_near), 2),
                     "mean_dist_m": round(float(d_col), 2),
                     "src_len_m": round(float(it["len"]), 2)})

    return out, rows, n_deg, n_differ


def evaluate(geoms, crs, truth_path, out_path):
    import geopandas as gpd
    from Comparison import evaluate_network

    if not geoms:
        return {}
    gpd.GeoDataFrame(geometry=geoms, crs=crs).to_file(out_path, driver="GeoJSON")
    df = evaluate_network(str(truth_path), str(out_path), output_folder=None)
    if df.empty:
        return {}
    g = df.loc["Global_Network"]
    return {"recall": round(float(g["Recall"]), 4),
            "precision": round(float(g["Precision"]), 4),
            "f1": round(float(g["F1_Score"]), 4)}


def run_sheet(sheet: str, warp: bool, tuned: bool = True) -> dict | None:
    from HistoricMapMatcher import HistoricMapMatcher
    from load_osmnx import load_osmnx

    hist_path = BYPASS / f"{sheet}_geolocated.geojson"
    if not hist_path.exists():
        print(f"[SKIP] {sheet}: no geolocated graph")
        return None

    print(f"\n{'='*68}\n[{sheet}] proximity-only, warp={warp}\n{'='*68}", flush=True)
    t0 = time.time()
    ov = OVERRIDES.get(sheet, {})
    G_hist, G_osm = load_osmnx(str(hist_path), str(ov.get("osm", OSM_PATH)))

    cfg = build_config(tuned)
    matcher = HistoricMapMatcher(G_hist, G_osm, cfg)

    edge_data = matcher._prepare_edge_data()
    n_links = len(edge_data)
    if warp:
        # Same Pass 0 the full pipeline runs, so the only difference downstream
        # is the absence of the composite score.
        probes = [matcher._probe_node_fast(it) for it in edge_data]
        matcher._apply_warp(probes)
        edge_data = matcher._prepare_edge_data()

    radius = float(cfg.initial_search_radius)
    variants, rows, n_deg, n_differ = assign(matcher, edge_data, radius)

    run_dir = OUT / sheet / ("warped" if warp else "raw")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "assignments.json").write_text(json.dumps(rows, indent=2))

    # Same reference the full pipeline was scored against: clipped to the hull of
    # the full historic graph, so the recall denominator does not move.
    truth = clip_truth(sheet, RESULTS / sheet, ov.get("truth", TRUTH))

    row = {"sheet": sheet, "warp": warp, "links": n_links,
           "assigned": len(rows), "degenerate_trims": n_deg,
           "rules_disagree": n_differ, "radius_m": radius}
    for name, geoms in variants.items():
        row[name] = evaluate(geoms, matcher.crs_meters, truth,
                             run_dir / f"proximity_{name}.geojson")

    best_name = max(variants, key=lambda n: row[n].get("f1", -1) if row[n] else -1)
    row["best"] = row[best_name]
    row["best_convention"] = best_name
    row["seconds"] = round(time.time() - t0, 1)

    del matcher, G_hist, G_osm
    gc.collect()
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", choices=SHEETS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-warp", action="store_true",
                    help="skip Pass 0 registration (default runs both arms)")
    ap.add_argument("--warp-only", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    targets = SHEETS if a.all else ([a.sheet] if a.sheet else [])
    if not targets:
        raise SystemExit("give --sheet or --all")
    arms = [True, False]
    if a.no_warp:
        arms = [False]
    elif a.warp_only:
        arms = [True]

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in targets:
        for warp in arms:
            try:
                r = run_sheet(s, warp)
                if r:
                    rows.append(r)
                    (OUT / f"proximity_results{a.tag}.json").write_text(json.dumps(rows, indent=2))
            except Exception as e:
                import traceback
                print(f"[FAIL] {s} warp={warp}: {type(e).__name__}: {e}")
                traceback.print_exc()

    if rows:
        print(f"\n{'='*94}\n{'sheet':<17}{'warp':>6}{'links':>7}{'assign':>8}"
              f"{'recall':>9}{'prec':>9}{'F1':>9}{'conv':>12}")
        for r in rows:
            b = r["best"]
            f = lambda k: f"{b[k]:9.3f}" if b.get(k) is not None else f"{'--':>9}"
            print(f"{r['sheet']:<17}{str(r['warp']):>6}{r['links']:>7}{r['assigned']:>8}"
                  f"{f('recall')}{f('precision')}{f('f1')}{r['best_convention']:>12}")
        print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
