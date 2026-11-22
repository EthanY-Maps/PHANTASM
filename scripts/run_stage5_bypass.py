"""Stage 5 on the bypassed sheets, evaluated against Turner et al. (2023).

In the paper-era run these sheets matched almost nothing, because Stage 4 had
starved the matcher: Annandale accepted 6 of 405 links, Macdonaldtown 0 of 283,
Waterloo 0 of 302 (Output/Atlas_1866/.../Similarity/match_summary.json). That is
what the dashes in Tables 6-7 record.

Here the geolocation comes from bypass_geoloc.py instead -- a transform fitted
truth->truth at 1-13 m residual, against the ~120 m automated Stage 4 leaves on
1903 -- applied to the paper-era predicted graphs, which are untouched.

Weights are the shipped V5.3 defaults, NOT the paper's Trial 202: the Optuna
study (optuna_pareto.db) no longer exists on disk, so Trial 202's parameters are
unrecoverable without re-running the search. Any number out of this is therefore
"the pipeline at its default operating point", not "the paper's operating point".

    python scripts/run_stage5_bypass.py --sheet Waterloo
    python scripts/run_stage5_bypass.py --all
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "modules" / "Similarity", ROOT / "_superceded" / "Similarity"):
    sys.path.insert(0, str(p))

PHD = Path("data")
BYPASS = PHD / "Output_Repro/Bypass"
RESULTS = PHD / "Output_Repro/Stage5"
OSM_PATH = PHD / "Input/Model/Sydney/sydney_drive.graphml"
TRUTH = Path("data/GreaterSydneyStreetsEvolution/DATA/Sydney_roads.shp")

# Every sheet with a control-point file and a paper-era predicted graph. The atlas
# GCPs arrived 2026-08-28 and supersede the earlier truth->truth ICP fits, so "Glebe"
# (ICP) is replaced by "The_Glebe" (GCP) and Macdonaldtown joins for the first time.
SHEETS = ["Annandale", "Waterloo", "Alexandria", "Camperdown", "The_Glebe", "Paddington",
          "Macdonaldtown", "Redfern", "St_Peters", "Drummoyne", "Enfield",
          "1889_Strathfield", "Melbourne_1942", "Sydney_1903"]

# Melbourne is out of Sydney, so it needs its own OSM extract, and Turner et al.
# does not cover it. Its reference is the manual pixel truth pushed through the
# same transform -- a different question from the Sydney sheets: how much of the
# 1942 drawn network the pipeline recovers, not how much of it survives today.
OVERRIDES = {
    "Melbourne_1942": {
        "osm": PHD / "Input/Model/melbourne_drive.graphml",
        "truth": BYPASS / "Melbourne_1942_truth_geolocated.geojson",
    },
}


# Trial 151 of the re-run NSGA-II search: the F1 argmax over the recomputed
# non-dominated set (recall 0.576, precision 0.739, F1 0.648). Weights are the raw
# samples normalised to sum to 1, exactly as the objective does.
TUNED = dict(
    shape_weight=0.1428, bearing_weight=0.6869, displacement_weight=0.0606,
    sinuosity_weight=0.0920, topology_weight=0.0177,
    initial_search_radius=50.0, relaxation_search_radius=125.0,   # 2.5x, per the objective
    min_len_ratio=0.17151, max_len_ratio=7.63728,
    relaxation_length_max=7.63728,
    high_confidence_threshold=0.21423, medium_confidence_threshold=0.5279,
    low_confidence_threshold=0.6779, very_low_confidence_threshold=0.9,
    max_candidates_edges=19, max_paths_to_solve=5,
)

# Absolute source-to-path displacement penalty, off by default so the tuned results
# stay reproducible. Set via --offset-cap / --offset-weight.



# Median registration residual per sheet, from the control-point fits. The offset cap
# has to scale with this: d_fwd is a MAXIMUM over vertices, so on a sheet registered to
# 30 m a fixed 40 m cap rejects correct matches wholesale.
RESIDUAL_M = {"Annandale": 7.8, "Alexandria": 9.6, "Waterloo": 8.7, "Paddington": 14.7,
              "Redfern": 4.5, "Camperdown": 8.7, "Macdonaldtown": 4.3, "The_Glebe": 7.6,
              "St_Peters": 8.3, "Drummoyne": 7.7, "Enfield": 12.2, "1889_Strathfield": 7.6,
              "Melbourne_1942": 17.2, "Sydney_1903": 30.0}


def build_config(tuned: bool, offset_cap: float = 0.0, offset_weight: float = 0.0):
    from model_parameters import SimilarityParameters

    cfg = SimilarityParameters()
    if tuned:
        for k, v in TUNED.items():
            setattr(cfg, k, v)
    cfg.source_offset_cap_m = offset_cap
    cfg.source_offset_weight = offset_weight
    return cfg


def clip_truth(sheet: str, run_dir: Path, truth_path: Path,
               margin_m: float = 100.0) -> Path:
    """Reference clipped to the sheet footprint.

    Clipped to the hull of the FULL historic graph, not the accepted matches. Using
    the accepted set makes the recall denominator depend on the configuration being
    evaluated -- a stricter setting accepts fewer links, shrinks the hull, and changes
    the reference it is being scored against, so two configurations are no longer
    comparable. The map's footprint is a property of the map.
    """
    import geopandas as gpd

    out = run_dir / "truth_clipped_fullhull.geojson"
    if out.exists():
        return out
    hist = gpd.read_file(BYPASS / f"{sheet}_geolocated.geojson")
    hist = hist[hist.geom_type == "LineString"]
    hist = hist.set_crs("EPSG:4326", allow_override=True).to_crs("EPSG:7856")
    truth = gpd.read_file(truth_path).to_crs("EPSG:7856")
    hull = hist.union_all().convex_hull.buffer(margin_m)
    sub = truth[truth.intersects(hull)].copy()
    sub["geometry"] = sub.intersection(hull)
    sub = sub[~sub.is_empty]
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_crs("EPSG:4326").to_file(out, driver="GeoJSON")
    print(f"  reference clipped to map footprint: {len(truth)} -> {len(sub)} lines")
    return out


def run_sheet(sheet: str, osm_cache: dict, tuned: bool = False,
              offset_cap: float = 0.0, offset_weight: float = 0.0,
              offset_mult: float = 0.0, out_root: Path | None = None) -> dict | None:
    from HistoricMapMatcher import HistoricMapMatcher
    from load_osmnx import load_osmnx
    from Comparison import evaluate_network
    from model_parameters import SimilarityParameters

    hist_path = BYPASS / f"{sheet}_geolocated.geojson"
    if not hist_path.exists():
        print(f"[SKIP] {sheet}: no bypassed graph")
        return None

    print(f"\n{'='*68}\n[{sheet}]\n{'='*68}", flush=True)
    t0 = time.time()
    if offset_mult:
        offset_cap = RESIDUAL_M.get(sheet, 10.0) * offset_mult
        print(f"  offset cap {offset_cap:.0f} m ({offset_mult}x residual)")
    ov = OVERRIDES.get(sheet, {})
    G_hist, G_osm = load_osmnx(str(hist_path), str(ov.get("osm", OSM_PATH)))

    cfg = build_config(tuned, offset_cap, offset_weight)
    matcher = HistoricMapMatcher(G_hist, G_osm, cfg)
    results = matcher.run_pipeline(n_jobs=1)

    run_dir = (out_root or RESULTS) / sheet
    run_dir.mkdir(parents=True, exist_ok=True)
    matcher.export_results(str(run_dir), results)

    row = {"sheet": sheet, "weights": "trial151" if tuned else "default",
           "seconds": round(time.time() - t0, 1)}
    summary = run_dir / "match_summary.json"
    if summary.exists():
        row.update(json.loads(summary.read_text()))

    pred = run_dir / "combined_high_medium.geojson"
    if pred.exists() and pred.stat().st_size > 100:
        # Turner et al. covers all of Sydney (21,103 lines). Scoring one suburb
        # against it makes recall meaningless -- a sheet can only ever cover its
        # own footprint, so unclipped recall just reports the sheet's share of
        # Sydney (Waterloo: 0.007). Clip the reference to the sheet extent.
        # The reference is a property of the map, not of the arm, so every arm
        # reads the one cached clip under RESULTS rather than making its own.
        clipped = clip_truth(sheet, RESULTS / sheet, ov.get("truth", TRUTH))
        df = evaluate_network(str(clipped), str(pred), output_folder=None)
        if not df.empty:
            g = df.loc["Global_Network"]
            row.update(recall=float(g["Recall"]), precision=float(g["Precision"]),
                       f1=float(g["F1_Score"]))
    else:
        print(f"[{sheet}] no accepted matches to evaluate")

    del matcher, results, G_hist
    gc.collect()
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", choices=SHEETS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--tuned", action="store_true",
                    help="use trial-151 weights instead of the shipped defaults")
    ap.add_argument("--offset-cap", type=float, default=0.0,
                    help="metres of source-to-path deviation tolerated before penalty")
    ap.add_argument("--offset-weight", type=float, default=0.0)
    ap.add_argument("--offset-mult", type=float, default=0.0,
                    help="set the cap to this multiple of the sheet's registration residual")
    ap.add_argument("--tag", default="", help="suffix for the results file")
    ap.add_argument("--out-subdir", default="",
                    help="write per-sheet outputs under Stage5/<name>/ instead of "
                         "Stage5/. Without it, arms overwrite each other's geojson "
                         "and anything computed from them acquires mixed provenance.")
    a = ap.parse_args()

    targets = SHEETS if a.all else ([a.sheet] if a.sheet else [])
    if not targets:
        raise SystemExit("give --sheet or --all")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_root = (RESULTS / a.out_subdir) if a.out_subdir else RESULTS
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in targets:
        try:
            r = run_sheet(s, {}, tuned=a.tuned,
                          offset_cap=a.offset_cap, offset_weight=a.offset_weight,
                          offset_mult=a.offset_mult, out_root=out_root)
            if r:
                rows.append(r)
        except Exception as e:
            print(f"[FAIL] {s}: {type(e).__name__}: {e}", flush=True)

    if rows:
        (out_root / (f"bypass_results{a.tag or ('_tuned' if a.tuned else '')}.json")).write_text(json.dumps(rows, indent=2))
        print(f"\n{'='*68}\n{'sheet':<13}{'total':>7}{'acc':>7}{'recall':>9}{'prec':>9}{'F1':>9}")
        for r in rows:
            rc, pr, f1 = r.get("recall"), r.get("precision"), r.get("f1")
            fmt = lambda v: f"{v:9.3f}" if isinstance(v, float) else f"{'--':>9}"
            print(f"{r['sheet']:<13}{r.get('total',0):>7}{r.get('accepted',0):>7}"
                  f"{fmt(rc)}{fmt(pr)}{fmt(f1)}")
        print(f"\nwritten -> {RESULTS}")


if __name__ == "__main__":
    main()
