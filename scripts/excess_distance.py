"""Excess distance: is the matched road the nearest road, or one passed over?

The four-cell classification combines a real matching result with a naive proximity test
against *any* modern road, and the two axes are independent. So cell A never verified that
the road a link sits near is the road it was matched to.

This measures that directly. For every matched link, on the same footing:

  nearest road   distance from the link to the closest drivable OSM edge
  matched path   distance from the link to the geometry the matcher actually chose
  excess         the difference, which is the quantity that separates the populations

Both are measured as the mean distance over points sampled along the link, so the two
columns are comparable; using a minimum for one and a mean for the other decides the
result by the metric.

A large excess means the matcher reached past a closer candidate. It is suspicious rather
than proof: the nearest road is not necessarily the correct one at junctions or beside
service lanes, and a stitched path spanning several edges can sit further from a distorted
source and still be right.

    python scripts/excess_distance.py --sheet Annandale
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
sys.path.insert(0, str(ROOT / "scripts"))

PHD = Path("data")
BYPASS = PHD / "Output_Repro/Bypass"
STAGE5 = PHD / "Output_Repro/Stage5/trial151"
OSM_SYD = PHD / "Input/Model/Sydney_Australia_drive_combined.geojson"
OSM_MEL = PHD / "Input/Model/melbourne_drive.graphml"
METRIC = "EPSG:7856"
COVER = 0.5
TOL = 30.0
OUT = PHD / "Output_Repro/ExcessDistance"


def source_key(geom) -> str:
    c = list(geom.coords)
    (x0, y0), (x1, y1) = c[0][:2], c[-1][:2]
    return f"('coord_{x0:.7f}_{y0:.7f}', 'coord_{x1:.7f}_{y1:.7f}', 0)"


def load_osm(sheet, bounds, pad=0.02):
    import geopandas as gpd
    if sheet == "Melbourne_1942":
        import osmnx as ox
        g = ox.graph_to_gdfs(ox.load_graphml(OSM_MEL), nodes=False)
        return g.set_crs("EPSG:4326", allow_override=True).to_crs(METRIC)
    x0, y0, x1, y1 = bounds
    return gpd.read_file(OSM_SYD, bbox=(x0 - pad, y0 - pad, x1 + pad, y1 + pad)).to_crs(METRIC)


def analyse(sheet: str, n_samples: int = 10) -> dict | None:
    import geopandas as gpd
    from shapely.strtree import STRtree

    hp = BYPASS / f"{sheet}_geolocated.geojson"
    ap = STAGE5 / sheet / "combined_high_medium.geojson"
    if not (hp.exists() and ap.exists()):
        print(f"  {sheet}: missing inputs")
        return None

    wgs = gpd.read_file(hp)
    wgs = wgs[wgs.geom_type == "LineString"]
    keys = [source_key(g) for g in wgs.geometry]

    h = wgs.set_crs("EPSG:4326", allow_override=True).to_crs(METRIC)
    acc = gpd.read_file(ap)
    acc["source_id"] = acc["source_id"].astype(str)
    acc = acc.to_crs(METRIC)
    path_by_key = dict(zip(acc["source_id"], acc.geometry))

    osm = load_osm(sheet, wgs.total_bounds)
    geoms = list(osm.geometry)
    tree = STRtree(geoms)
    blob = osm.geometry.buffer(TOL).union_all()

    def samples(src):
        return [src.interpolate(t, normalized=True) for t in np.linspace(0, 1, n_samples)]

    def mean_dist(pts, geom):
        return float(np.mean([geom.distance(q) for q in pts]))

    def mean_dist_to_network(pts):
        """Distance to the nearest road, resolved per sampled point.

        Taking the single edge nearest the whole link and measuring every sample
        against it is wrong: on a long link that edge may only be near one end, which
        reported cell A -- links lying on a road by construction -- as 23 m from the
        network. The nearest road is a property of each point, not of the link.
        """
        return float(np.mean([geoms[int(tree.nearest(q))].distance(q) for q in pts]))

    rows = []
    for key, g in zip(keys, h.geometry):
        path = path_by_key.get(key)
        if path is None:
            continue
        pts = samples(g)
        cov = g.intersection(blob).length / max(g.length, 1e-9)
        rows.append({"in30": cov >= COVER,
                     "d_near": mean_dist_to_network(pts),
                     "d_path": mean_dist(pts, path)})

    if not rows:
        return None

    def summarise(sub):
        if not sub:
            return None
        near = np.array([r["d_near"] for r in sub])
        path = np.array([r["d_path"] for r in sub])
        exc = path - near
        return {"n": len(sub),
                "nearest_road_m": round(float(np.median(near)), 1),
                "matched_path_m": round(float(np.median(path)), 1),
                "excess_median_m": round(float(np.median(exc)), 1),
                "excess_over_25m_share": round(float((exc > 25).mean()), 3)}

    out = {"sheet": sheet, "tol_m": TOL,
           "cell_A": summarise([r for r in rows if r["in30"]]),
           "cell_C": summarise([r for r in rows if not r["in30"]])}
    print(f"  {sheet} done (A={out['cell_A']['n'] if out['cell_A'] else 0}, "
          f"C={out['cell_C']['n'] if out['cell_C'] else 0})", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet")
    ap.add_argument("--sheets", default="Annandale")
    a = ap.parse_args()
    targets = [a.sheet] if a.sheet else a.sheets.split(",")

    rows = [r for t in targets if (r := analyse(t.strip()))]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "excess_distance.json").write_text(json.dumps(rows, indent=2))

    print(f"\n{'='*88}\n{'sheet':<16}{'cell':>8}{'n':>7}{'nearest':>10}"
          f"{'matched':>10}{'excess':>10}{'>25 m':>9}")
    for r in rows:
        for name in ("cell_A", "cell_C"):
            c = r[name]
            if not c:
                continue
            print(f"{r['sheet']:<16}{name[-1]:>8}{c['n']:>7}{c['nearest_road_m']:>10.1f}"
                  f"{c['matched_path_m']:>10.1f}{c['excess_median_m']:>10.1f}"
                  f"{c['excess_over_25m_share']:>9.1%}")
    print(f"\nwritten -> {OUT/'excess_distance.json'}")


if __name__ == "__main__":
    main()
