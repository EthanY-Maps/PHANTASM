"""Interrogate cell C: historic links matched despite no OSM road along them.

Cell C is where the false-positive question actually lives. A link lands there for one
of two very different reasons:

  displaced  the road does exist in OSM, but registration offset pushes it past the
             coverage tolerance -- the match is right, the test is too strict
  spurious   there is no modern road there and the matcher reached for one anyway

They are separable. For each C link we measure:

  * coverage at the test tolerance, and how much it recovers at looser tolerances.
    A displaced link sits just under the threshold and crosses it when the tolerance
    exceeds the sheet's registration residual. A spurious one stays near zero.
  * distance from the historic link to the OSM path the matcher chose. Since that path
    is built from OSM edges, a large distance means the matcher selected geometry far
    from the source -- the signature of a reach.
  * the composite score, against cell A as the control. If C links score no worse than
    A links, the matcher cannot tell them apart, which is the more troubling finding.

    python scripts/cell_c_analysis.py --sheet Annandale
    python scripts/cell_c_analysis.py --sheets Annandale,Alexandria,Sydney_1903
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
METRIC = "EPSG:7856"
COVER = 0.5
TOLS = [30.0, 45.0, 60.0, 100.0, 200.0]

RESIDUAL = {"Annandale": 7.8, "Alexandria": 9.6, "Waterloo": 8.7, "Paddington": 14.7,
            "Redfern": 4.5, "Camperdown": 8.7, "Macdonaldtown": 4.3, "The_Glebe": 7.6,
            "St_Peters": 8.3, "Drummoyne": 7.7, "Enfield": 12.2, "1889_Strathfield": 7.6,
            "Melbourne_1942": 17.2, "Sydney_1903": 30.0}


def source_key(geom) -> str:
    c = list(geom.coords)
    (x0, y0), (x1, y1) = c[0][:2], c[-1][:2]
    return f"('coord_{x0:.7f}_{y0:.7f}', 'coord_{x1:.7f}_{y1:.7f}', 0)"


def load_osm(sheet: str, bounds, pad: float = 0.02):
    import geopandas as gpd
    if sheet == "Melbourne_1942":
        import osmnx as ox
        g = ox.graph_to_gdfs(ox.load_graphml(OSM_MEL), nodes=False)
        return g.set_crs("EPSG:4326", allow_override=True).to_crs(METRIC)
    x0, y0, x1, y1 = bounds
    return gpd.read_file(OSM_SYD, bbox=(x0 - pad, y0 - pad, x1 + pad, y1 + pad)).to_crs(METRIC)


def analyse(sheet: str) -> dict | None:
    import geopandas as gpd
    import numpy as np

    hp, ap = BYPASS / f"{sheet}_geolocated.geojson", STAGE5 / sheet / "combined_high_medium.geojson"
    if not (hp.exists() and ap.exists()):
        print(f"  {sheet}: missing inputs"); return None

    h = gpd.read_file(hp)
    h = h[h.geom_type == "LineString"].copy()
    acc = gpd.read_file(ap)
    acc["source_id"] = acc["source_id"].astype(str)
    by_src = {k: g for k, g in zip(acc["source_id"], acc.geometry)}
    score_by_src = dict(zip(acc["source_id"], acc["score"]))

    h["key"] = [source_key(g) for g in h.geometry]
    h["matched"] = h["key"].isin(by_src)

    wgs = h.total_bounds
    h = h.set_crs("EPSG:4326", allow_override=True).to_crs(METRIC)
    h["len_m"] = h.length
    osm = load_osm(sheet, wgs)

    # coverage at each tolerance
    covers = {}
    for tol in TOLS:
        blob = osm.geometry.buffer(tol).union_all()
        covers[tol] = (h.geometry.intersection(blob).length / h["len_m"].clip(lower=1e-9)).values

    base = covers[30.0] >= COVER
    isC = (~base) & h["matched"].values
    isA = base & h["matched"].values
    # Section 7.5.4 makes a claim about every link outside 30 m, matched or not --
    # cells C and D together. D dominates on Melbourne and is absent from the cell-C
    # population, so the two are not interchangeable and both are reported.
    isCD = ~base
    if isC.sum() == 0:
        print(f"  {sheet}: cell C empty"); return None

    # Distance from each matched link to the OSM path the matcher chose.
    # Stage 5 exports in the UTM zone it worked in (EPSG:32756/32755), NOT lat/lon,
    # so take the file's declared CRS rather than assuming WGS84.
    matched_paths = gpd.GeoSeries(
        [by_src.get(k) for k in h["key"]], crs=acc.crs).to_crs(METRIC)
    d_to_path = np.array([
        (g.distance(p) if (p is not None and g is not None) else np.nan)
        for g, p in zip(h.geometry, matched_paths)])

    scores = np.array([score_by_src.get(k, np.nan) for k in h["key"]], float)

    def pct(mask, arr, q):
        v = arr[mask]; v = v[~np.isnan(v)]
        return float(np.percentile(v, q)) if len(v) else float("nan")

    # how much of C recovers as the tolerance loosens
    recovery = {}
    nC = int(isC.sum())
    for tol in TOLS:
        recovery[int(tol)] = round(float((covers[tol][isC] >= COVER).sum()) / nC, 3)

    nCD = int(isCD.sum())
    recovery_cd = {int(t): round(float((covers[t][isCD] >= COVER).sum()) / nCD, 3)
                   for t in TOLS} if nCD else {}

    row = {
        "sheet": sheet, "residual_m": RESIDUAL.get(sheet),
        "n_A": int(isA.sum()), "n_C": nC,
        "C_km": round(float(h.loc[isC, "len_m"].sum()) / 1000, 2),
        "C_median_len_m": round(float(np.median(h.loc[isC, "len_m"])), 1),
        "A_median_len_m": round(float(np.median(h.loc[isA, "len_m"])), 1),
        "C_median_coverage_at_30m": round(float(np.median(covers[30.0][isC])), 3),
        "C_zero_coverage_share": round(float((covers[30.0][isC] < 0.05).sum()) / nC, 3),
        "recovery_by_tolerance": recovery,
        "n_outside_30m": nCD,
        "recovery_outside_30m": recovery_cd,
        "C_dist_to_matched_path_m": {q: round(pct(isC, d_to_path, q), 1) for q in (50, 90)},
        "A_dist_to_matched_path_m": {q: round(pct(isA, d_to_path, q), 1) for q in (50, 90)},
        "C_score": {q: round(pct(isC, scores, q), 3) for q in (50, 90)},
        "A_score": {q: round(pct(isA, scores, q), 3) for q in (50, 90)},
    }
    print(f"  {sheet} done (C={nC})", flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet")
    ap.add_argument("--sheets", help="comma-separated")
    a = ap.parse_args()
    targets = ([a.sheet] if a.sheet else
               a.sheets.split(",") if a.sheets else
               ["Annandale", "Alexandria", "Redfern", "Melbourne_1942", "Sydney_1903"])

    rows = [r for s in targets if (r := analyse(s.strip()))]
    out = STAGE5 / "cell_c_analysis.json"
    out.write_text(json.dumps(rows, indent=2))

    print(f"\n{'='*100}\nCELL C: displaced or spurious?\n{'='*100}")
    print(f"{'sheet':<17}{'resid':>6}{'C':>7}{'C med cov':>11}{'~zero cov':>11}"
          f"{'dist to path (m)':>19}{'score C/A':>14}")
    for r in rows:
        print(f"{r['sheet']:<17}{r['residual_m']:>6.1f}{r['n_C']:>7}"
              f"{r['C_median_coverage_at_30m']:>11.3f}{r['C_zero_coverage_share']:>11.1%}"
              f"{r['C_dist_to_matched_path_m'][50]:>9.1f} /{r['A_dist_to_matched_path_m'][50]:>7.1f}"
              f"{r['C_score'][50]:>7.3f} /{r['A_score'][50]:>6.3f}")

    print(f"\n{'sheet':<17}  share of ALL links outside 30 m (cells C+D) reaching >=50% coverage")
    print(f"{'':<17}" + "".join(f"{t:>9}m" for t in (30, 45, 60, 100, 200)) + f"{'n':>8}")
    for r in rows:
        rec = r.get("recovery_outside_30m") or {}
        if rec:
            print(f"{r['sheet']:<17}" + "".join(f"{rec[t]:>10.1%}" for t in (30, 45, 60, 100, 200))
                  + f"{r['n_outside_30m']:>8}")

    print(f"\n{'sheet':<17}  share of C reaching >=50% coverage as tolerance loosens")
    print(f"{'':<17}" + "".join(f"{t:>9}m" for t in (30, 45, 60, 100, 200)))
    for r in rows:
        rec = r["recovery_by_tolerance"]
        print(f"{r['sheet']:<17}" + "".join(f"{rec[t]:>10.1%}" for t in (30, 45, 60, 100, 200)))
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
