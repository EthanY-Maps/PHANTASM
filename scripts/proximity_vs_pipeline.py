"""Does the composite score choose a nearer road than proximity does, or a further one?

Both arms are measured identically: mean distance from ten points sampled along the
historic link to the geometry that arm selected. Using a minimum distance for one and a
mean for the other would decide the comparison by the metric.

The proximity baseline scores better than the full matcher on the buffer metric, which
raises the obvious question of what the composite score is doing differently. Buffer F1
cannot answer it: both arms predict OSM geometry, and OSM geometry lies on the reference
by construction, so the metric rewards covering roads rather than covering the right ones.

This compares the two choices per historic link instead. For every link that both arms
matched, it measures the distance from the link to the geometry each arm selected. If the
matcher's path is systematically further from the source than the nearest available
candidate, the composite score is reaching past closer geometry -- the same signature
Section 8.5 identifies in cell C, now measured against an explicit alternative rather than
against the nearest road in the abstract.

Reads existing artefacts only; nothing is re-matched.

    python scripts/proximity_vs_pipeline.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

PHD = Path("data")
BYPASS = PHD / "Output_Repro/Bypass"
# Per-arm output directory. Arms used to share one path and overwrite each other,
# so anything computed from the per-sheet geojson inherited whichever arm ran last.
# Everything downstream now reads the one canonical trial-151 run.
STAGE5 = PHD / "Output_Repro/Stage5/trial151"
PROX = PHD / "Output_Repro/Proximity"
METRIC = "EPSG:7856"

SHEETS = ["St_Peters", "Drummoyne", "Enfield", "1889_Strathfield", "Paddington",
          "The_Glebe", "Alexandria", "Waterloo", "Annandale", "Macdonaldtown",
          "Redfern", "Camperdown", "Melbourne_1942", "Sydney_1903"]


def analyse(sheet: str) -> dict | None:
    import geopandas as gpd

    assign = PROX / sheet / "raw" / "assignments.json"
    acc_path = STAGE5 / sheet / "combined_high_medium.geojson"
    hist_path = BYPASS / f"{sheet}_geolocated.geojson"
    if not (assign.exists() and acc_path.exists() and hist_path.exists()):
        print(f"  {sheet}: missing inputs")
        return None

    rows = {r["source_id"]: r for r in json.loads(assign.read_text())}

    acc = gpd.read_file(acc_path)
    acc["source_id"] = acc["source_id"].astype(str)
    # Stage 5 exports in the UTM zone it worked in, not WGS84.
    acc = acc.to_crs(METRIC)

    hist = gpd.read_file(hist_path)
    hist = hist[hist.geom_type == "LineString"]
    hist = hist.set_crs("EPSG:4326", allow_override=True).to_crs(METRIC)
    # rebuild the matcher's edge key from endpoints, as match_breakdown.py does,
    # but on the WGS84 coordinates the key was built from
    wgs = gpd.read_file(hist_path)
    wgs = wgs[wgs.geom_type == "LineString"]
    keys = []
    for g in wgs.geometry:
        c = list(g.coords)
        (x0, y0), (x1, y1) = c[0][:2], c[-1][:2]
        keys.append(f"('coord_{x0:.7f}_{y0:.7f}', 'coord_{x1:.7f}_{y1:.7f}', 0)")
    geom_by_key = dict(zip(keys, hist.geometry))

    # Both arms must be measured the same way or the comparison is meaningless.
    # The proximity record stores the MEAN distance from ten points sampled along
    # the link, so the pipeline's path is measured the same way rather than by the
    # minimum distance, which would flatter it wherever the two geometries touch.
    def mean_dist(src, geom, n=10):
        pts = [src.interpolate(t, normalized=True) for t in np.linspace(0, 1, n)]
        return float(np.mean([geom.distance(q) for q in pts]))

    d_pipe, d_prox = [], []
    for _, r in acc.iterrows():
        k = r["source_id"]
        src = geom_by_key.get(k)
        rec = rows.get(k)
        if src is None or rec is None or r.geometry is None:
            continue
        d_pipe.append(mean_dist(src, r.geometry))
        d_prox.append(rec["mean_dist_m"])

    if not d_pipe:
        print(f"  {sheet}: no overlapping source ids")
        return None

    d_pipe, d_prox = np.array(d_pipe), np.array(d_prox)
    excess = d_pipe - d_prox
    out = {"sheet": sheet, "n": int(len(d_pipe)),
           "pipeline_median_m": round(float(np.median(d_pipe)), 1),
           "proximity_median_m": round(float(np.median(d_prox)), 1),
           "excess_median_m": round(float(np.median(excess)), 1),
           "excess_p90_m": round(float(np.percentile(excess, 90)), 1),
           "share_pipeline_further": round(float((excess > 0).mean()), 3),
           "share_further_by_25m": round(float((excess > 25).mean()), 3)}
    print(f"  {sheet} done (n={out['n']})", flush=True)
    return out


def main() -> None:
    rows = [r for s in SHEETS if (r := analyse(s))]
    out = PROX / "proximity_vs_pipeline.json"
    out.write_text(json.dumps(rows, indent=2))

    print(f"\n{'='*104}\nDistance from each historic link to the geometry each arm chose\n{'='*104}")
    print(f"{'sheet':<17}{'n':>7}{'pipeline':>11}{'proximity':>11}"
          f"{'excess med':>12}{'excess p90':>12}{'pipe further':>14}{'by >25 m':>10}")
    for r in rows:
        print(f"{r['sheet']:<17}{r['n']:>7}{r['pipeline_median_m']:>11.1f}"
              f"{r['proximity_median_m']:>11.1f}{r['excess_median_m']:>12.1f}"
              f"{r['excess_p90_m']:>12.1f}{r['share_pipeline_further']:>13.1%}"
              f"{r['share_further_by_25m']:>10.1%}")
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
