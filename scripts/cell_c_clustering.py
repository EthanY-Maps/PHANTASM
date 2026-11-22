"""Are cell C links clustered or scattered?

Clustered -> a local registration failure in one part of the sheet: the whole
neighbourhood is offset, so more control points there would fix it.
Scattered  -> per-link matching error, e.g. grabbing a parallel street.

Normalised test: for each C link, what share of its 8 nearest neighbours (among all
links) are also C. Under random scattering this equals C's share of the sheet; well
above that means clustering.
"""
import sys, warnings, json; warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import numpy as np, geopandas as gpd
from scipy.spatial import cKDTree
from cell_c_analysis import source_key, load_osm, BYPASS, STAGE5, METRIC, COVER

for sheet in ["Annandale", "Alexandria", "Redfern", "Waterloo", "Macdonaldtown", "Melbourne_1942"]:
    h = gpd.read_file(BYPASS / f"{sheet}_geolocated.geojson")
    h = h[h.geom_type == "LineString"].copy()
    acc = gpd.read_file(STAGE5 / sheet / "combined_high_medium.geojson")
    accs = set(acc["source_id"].astype(str))
    h["key"] = [source_key(g) for g in h.geometry]
    h["matched"] = h["key"].isin(accs)
    wgs = h.total_bounds
    h = h.set_crs("EPSG:4326", allow_override=True).to_crs(METRIC)
    osm = load_osm(sheet, wgs)
    blob = osm.geometry.buffer(30.0).union_all()
    cov = (h.geometry.intersection(blob).length / h.length.clip(lower=1e-9)).values
    isC = (cov < COVER) & h["matched"].values

    xy = np.c_[h.geometry.centroid.x, h.geometry.centroid.y]
    tree = cKDTree(xy)
    k = min(9, len(xy))
    _, idx = tree.query(xy, k=k)          # includes self
    nb = idx[:, 1:]
    share_C_nb = isC[nb].mean(axis=1)
    baseline = isC.mean()
    obs = share_C_nb[isC].mean()
    print(f"{sheet:<16} C={isC.sum():>4}  baseline share={baseline:.3f}  "
          f"C-neighbourhood share={obs:.3f}  ratio={obs/baseline if baseline else float('nan'):.1f}x")
