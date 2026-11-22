"""Classify every historic link by whether OSM has it and whether we matched it.

Precision and recall against an OSM-aligned reference conflate two very different
failures: a road the pipeline should have matched and didn't, and a road that simply
has no modern counterpart to match to. A sheet can be near-perfect on everything
recoverable and still score badly because most of its 1903 streets no longer exist.

Each historic link falls into one of four cells:

  A  in OSM,     matched      -> recovered correctly
  B  in OSM,     not matched  -> a genuine pipeline miss
  C  not in OSM, matched      -> matched with no modern counterpart (suspect)
  D  not in OSM, not matched  -> correctly left alone; the road is gone

The number that actually measures the matcher is **conditional recall**, A/(A+B):
of the historic roads that OSM could have matched, how many did we get. Headline
recall mixes that with D, which is a fact about the city, not about the pipeline.

"In OSM" is a geometric test: the fraction of a link's length lying within `tol` of
any drivable OSM edge. Reported at two tolerances so the reader can see it is not an
artefact of one threshold.

    python scripts/match_breakdown.py                 # all sheets
    python scripts/match_breakdown.py --sheet Waterloo
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PHD = Path("data")
BYPASS = PHD / "Output_Repro/Bypass"
# Per-arm output directory. Arms used to share one path and overwrite each other,
# so anything computed from the per-sheet geojson inherited whichever arm ran last.
# Everything downstream now reads the one canonical trial-151 run.
STAGE5 = PHD / "Output_Repro/Stage5/trial151"
OSM_SYD = PHD / "Input/Model/Sydney_Australia_drive_combined.geojson"
OSM_MEL = PHD / "Input/Model/melbourne_drive.graphml"
METRIC = "EPSG:7856"          # GDA2020 / MGA zone 56, metres
COVER = 0.5                   # a link "is in OSM" if this share of it lies within tol

SHEETS = ["St_Peters", "Drummoyne", "Enfield", "1889_Strathfield", "Paddington",
          "The_Glebe", "Alexandria", "Waterloo", "Annandale", "Macdonaldtown",
          "Redfern", "Camperdown", "Melbourne_1942", "Sydney_1903"]


def source_key(geom) -> str:
    """Rebuild the matcher's edge key from a link's endpoints.

    Stage 5 exports `source_id` as a stringified (node, node, 0) tuple built from
    7-decimal coordinate names, so the historic geometry regenerates it exactly.
    """
    c = list(geom.coords)
    (x0, y0), (x1, y1) = c[0][:2], c[-1][:2]
    return f"('coord_{x0:.7f}_{y0:.7f}', 'coord_{x1:.7f}_{y1:.7f}', 0)"


def load_osm(sheet: str, bounds, pad: float = 0.01):
    import geopandas as gpd

    if sheet == "Melbourne_1942":
        import osmnx as ox
        g = ox.graph_to_gdfs(ox.load_graphml(OSM_MEL), nodes=False)
        return g.set_crs("EPSG:4326", allow_override=True).to_crs(METRIC)

    x0, y0, x1, y1 = bounds
    g = gpd.read_file(OSM_SYD, bbox=(x0 - pad, y0 - pad, x1 + pad, y1 + pad))
    return g.to_crs(METRIC)


def analyse(sheet: str, tols=(15.0, 30.0), links_dir: Path | None = None) -> dict | None:
    import geopandas as gpd

    hist_path = BYPASS / f"{sheet}_geolocated.geojson"
    acc_path = STAGE5 / sheet / "combined_high_medium.geojson"
    if not hist_path.exists() or not acc_path.exists():
        print(f"  {sheet}: missing inputs, skipped")
        return None

    h = gpd.read_file(hist_path)
    h = h[h.geom_type == "LineString"].copy()
    if h.empty:
        return None
    accepted = set(gpd.read_file(acc_path)["source_id"].astype(str))
    h["source_key"] = [source_key(g) for g in h.geometry]
    h["matched"] = h["source_key"].isin(accepted)

    wgs_bounds = h.total_bounds
    h = h.set_crs("EPSG:4326", allow_override=True).to_crs(METRIC)
    h["len_m"] = h.length

    osm = load_osm(sheet, wgs_bounds)
    row = {"sheet": sheet, "links": len(h), "length_km": round(h["len_m"].sum() / 1000, 1)}

    for tol in tols:
        blob = osm.geometry.buffer(tol).union_all()
        inside = h.geometry.intersection(blob).length
        in_osm = (inside / h["len_m"].clip(lower=1e-9)) >= COVER

        cells = {}
        for name, mask in (("A_in_matched",    in_osm & h["matched"]),
                           ("B_in_missed",     in_osm & ~h["matched"]),
                           ("C_notin_matched", ~in_osm & h["matched"]),
                           ("D_notin_missed",  ~in_osm & ~h["matched"])):
            cells[name] = {"n": int(mask.sum()),
                           "km": round(float(h.loc[mask, "len_m"].sum()) / 1000, 2)}

        a, b = cells["A_in_matched"]["n"], cells["B_in_missed"]["n"]
        c, d = cells["C_notin_matched"]["n"], cells["D_notin_missed"]["n"]
        cells["conditional_recall"] = round(a / (a + b), 3) if (a + b) else None
        cells["osm_coverage"] = round((a + b) / len(h), 3)
        cells["match_rate"] = round((a + c) / len(h), 3)
        row[f"tol{int(tol)}m"] = cells

        h[f"in_osm_{int(tol)}m"] = in_osm
        h[f"cell_{int(tol)}m"] = [("A" if m else "B") if i else ("C" if m else "D")
                                  for i, m in zip(in_osm, h["matched"])]

    if links_dir is not None:
        links_dir.mkdir(parents=True, exist_ok=True)
        cols = ["source_key", "matched", "len_m"]
        for tol in tols:
            cols += [f"in_osm_{int(tol)}m", f"cell_{int(tol)}m"]
        links = h[cols + ["geometry"]].copy()
        links["len_m"] = links["len_m"].round(1)
        links.to_crs("EPSG:4326").to_file(links_dir / f"{sheet}_links.geojson", driver="GeoJSON")

    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", choices=SHEETS)
    ap.add_argument("--tol", type=float, default=15.0,
                    help="tolerance to print in the table (both are stored)")
    ap.add_argument("--links-out", type=Path,
                    help="also write every link's cell as GeoJSON, one file per sheet, here")
    a = ap.parse_args()

    targets = [a.sheet] if a.sheet else SHEETS
    rows = []
    for s in targets:
        try:
            r = analyse(s, links_dir=a.links_out)
        except Exception as e:
            print(f"  {s}: FAILED {type(e).__name__}: {e}")
            continue
        if r:
            rows.append(r)
            print(f"  {s} done", flush=True)

    out = STAGE5 / "match_breakdown.json"
    out.write_text(json.dumps(rows, indent=2))

    k = f"tol{int(a.tol)}m"
    print(f"\n{'='*104}\nBreakdown at {a.tol:.0f} m, by link count "
          f"(A/B/C/D; conditional recall = A/(A+B))\n{'='*104}")
    print(f"{'sheet':<17}{'links':>6}{'A in+mat':>10}{'B in+miss':>11}"
          f"{'C out+mat':>11}{'D out+miss':>12}{'OSM cov':>9}{'cond.rec':>10}")
    for r in rows:
        c = r[k]
        print(f"{r['sheet']:<17}{r['links']:>6}{c['A_in_matched']['n']:>10}"
              f"{c['B_in_missed']['n']:>11}{c['C_notin_matched']['n']:>11}"
              f"{c['D_notin_missed']['n']:>12}{c['osm_coverage']:>9.3f}"
              f"{(c['conditional_recall'] if c['conditional_recall'] is not None else float('nan')):>10.3f}")
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
