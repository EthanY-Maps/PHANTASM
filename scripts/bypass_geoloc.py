"""Bypass Stage 4 for sheets whose automated geolocation failed.

The paper reports end-to-end geolocation and matching on the 1903 sheet only:
Annandale and Macdonaldtown retained three anchors after filtering, Waterloo
none, so Stages 4-5 rest on n = 1.

A sheet that has a manually geolocated truth graph does not need Stage 4 at all.
The pixel->world transform is a property of the raster, so it can be recovered by
registering the *pixel-space* truth graph onto the *geolocated* truth graph --
both manual products of the same digitisation -- and then applied to whatever
predicted graph the pipeline produces for that sheet.

Deriving the transform from truth->truth (never from the prediction) is what
keeps the prediction independent of the fit. What comes out is a manual-GCP
baseline: matching performance given correct
georeferencing, with Stage 4 removed from the error chain rather than hidden
inside it.

Node ids do not survive between the two truth files (measured overlap: 1 of 853
for Annandale), and the geolocated truth is a filtered subset of the pixel truth,
so correspondence has to be established geometrically. Trimmed ICP over a grid of
rotation and scale starts, with a similarity model -- the same model the existing
control_points_*.json files use.

Usage:
    python scripts/bypass_geoloc.py fit --all
    python scripts/bypass_geoloc.py fit --sheet Annandale
    python scripts/bypass_geoloc.py apply --sheet Annandale \
        --pred  /path/to/Vectorisation/roads.geojson \
        --out   /path/to/Geolocation/Annandale_geolocated.geojson
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PHD = Path("data")
ATLAS = PHD / "Input/Maps of the World/Australia/Sydney/1886-1889 Atlas of Sydney"
GEOTRUTH = PHD / "Training/Geolocated Truths"
UTM = "EPSG:32756"          # Sydney; matches control_points_1877.json
OUT = Path(__file__).resolve().parent.parent / "control_points"

# sheet -> (pixel-space truth, geolocated truth). Spelling of the geolocated
# files is theirs, typos included.
SHEETS = {
    "Annandale":   (ATLAS / "Annandale/Truth.geojson",   GEOTRUTH / "Geolocated_Annadale.geojson"),
    "Waterloo":    (ATLAS / "Waterloo/Truth.geojson",    GEOTRUTH / "Geolocated_Waterloo.geojson"),
    "Alexandria":  (ATLAS / "Alexandria/Truth.geojson",  GEOTRUTH / "Geolocated_Alexandira.geojson"),
    "Camperdown":  (ATLAS / "Camperdown/Truth.geojson",  GEOTRUTH / "Geolocated_Camperdown.geojson"),
    "Glebe":       (ATLAS / "The Glebe/Truth.geojson",   GEOTRUTH / "Geolocated_Glebe.geojson"),
    "Paddington":  (ATLAS / "Paddington/Truth.geojson",  GEOTRUTH / "Geolocated_Paddingtone.geojson"),
    "Sydney_1877": (PHD / "Input/Maps of the World/Australia/Sydney/1877/Truth.geojson",
                    GEOTRUTH / "Geolocated_1877.geojson"),
}

# Macdonaldtown has a pixel truth but no geolocated truth, so it cannot be
# bypassed this way and stays absent from the table.

# Maps that already carry a manual control-point file need no registration at all:
# the file stores the correspondences directly. Melbourne 1942 is the paper's
# out-of-domain sheet, so bypassing it tests the approach outside Sydney.
TRUTHS = PHD / "Input/Maps of the World/Truths"
ATLAS_CP = TRUTHS / "Atlas of Syndey"          # spelling is theirs

CP_SHEETS = {
    "Melbourne_1942": TRUTHS / "control_points_1942.json",
    "Sydney_1903":    TRUTHS / "control_points_1903_Sydney.json",
}
# The atlas sheets gained their own GCP files on 2026-08-28, which supersede the
# truth->truth ICP fits for every sheet that has one -- same provenance as 1903 and
# Melbourne, and they bring in Macdonaldtown, which had no geolocated truth to
# register against. Discovered rather than listed so new sheets are picked up.
if ATLAS_CP.is_dir():
    for f in sorted(ATLAS_CP.glob("control_points_*.json")):
        CP_SHEETS.setdefault(f.stem.replace("control_points_", "").replace(" ", "_"), f)

# The stored scale/rotation/translation use a convention that does not round-trip
# as written, but the raw `pairs` do: pixel (col, -row) -> lon/lat reproduces the
# recorded fit exactly (Melbourne: scale 1.2215, median residual 17.18 m).
def fit_from_control_points(sheet: str) -> tuple["Fit", str]:
    import numpy as np
    from pyproj import Transformer

    cp = json.loads(CP_SHEETS[sheet].read_text())
    pairs = cp["pairs"]
    P = np.array([q["pixel_yx"] for q in pairs], float)
    L = np.array([q["osm_lonlat"] for q in pairs], float)
    epsg = cp.get("utm_epsg", UTM)
    tr = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    dst = np.column_stack(tr.transform(L[:, 0], L[:, 1]))
    src = np.column_stack([P[:, 1], -P[:, 0]])          # (col, -row)

    scale, R, t = umeyama(src, dst)
    res = np.linalg.norm(scale * (src @ R.T) + t - dst, axis=1)
    return Fit(scale=float(scale),
               rot_deg=float(math.degrees(math.atan2(R[1, 0], R[0, 0]))),
               translation=t, residuals=res, n_pairs=len(src)), epsg


# ── geometry ────────────────────────────────────────────────────────────────

def nodes_from_geojson(path: Path, to_utm: bool = False) -> np.ndarray:
    """Graph nodes = the endpoints of every LineString, deduplicated."""
    import geopandas as gpd

    g = gpd.read_file(path)
    g = g[g.geom_type.isin(["LineString", "MultiLineString"])]
    if to_utm:
        g = g.set_crs("EPSG:4326", allow_override=True).to_crs(UTM)

    pts = []
    for geom in g.geometry:
        parts = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        for part in parts:
            c = np.asarray(part.coords, float)
            if len(c):
                pts.append(c[0][:2])
                pts.append(c[-1][:2])
    if not pts:
        raise SystemExit(f"no linestring endpoints in {path}")
    pts = np.asarray(pts, float)
    # dedupe to ~0.5 unit, so shared junctions do not dominate the fit
    return np.unique(np.round(pts * 2) / 2, axis=0)


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Least-squares similarity (scale, rotation, translation) src -> dst."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    C = (d0.T @ s0) / len(src)
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    R = U @ S @ Vt
    var = (s0 ** 2).sum() / len(src)
    scale = float(np.trace(np.diag(D) @ S) / var) if var > 0 else 1.0
    t = mu_d - scale * (R @ mu_s)
    return scale, R, t


@dataclass
class Fit:
    scale: float
    rot_deg: float
    translation: np.ndarray
    residuals: np.ndarray
    n_pairs: int

    def apply(self, pts: np.ndarray) -> np.ndarray:
        r = math.radians(self.rot_deg)
        c, s = math.cos(r), math.sin(r)
        R = np.array([[c, -s], [s, c]])
        return self.scale * (np.asarray(pts, float).reshape(-1, 2) @ R.T) + self.translation

    @property
    def median(self) -> float: return float(np.median(self.residuals))

    @property
    def p90(self) -> float: return float(np.percentile(self.residuals, 90))


def icp(src, dst, scale0, rot0_deg, trim=0.7, iters=60):
    """Trimmed ICP with a similarity model. Trimming absorbs the subset mismatch."""
    from scipy.spatial import cKDTree

    tree = cKDTree(dst)
    r = math.radians(rot0_deg)
    c, s = math.cos(r), math.sin(r)
    R = np.array([[c, -s], [s, c]])
    scale = scale0
    t = dst.mean(0) - scale * (R @ src.mean(0))

    keep = max(8, int(len(src) * trim))
    prev = np.inf
    for _ in range(iters):
        moved = scale * (src @ R.T) + t
        d, idx = tree.query(moved)
        order = np.argsort(d)[:keep]                 # trim the worst pairs
        scale, R, t = umeyama(src[order], dst[idx[order]])
        err = float(np.mean(d[order]))
        if abs(prev - err) < 1e-4:
            break
        prev = err

    moved = scale * (src @ R.T) + t
    d, idx = tree.query(moved)
    order = np.argsort(d)[:keep]
    return Fit(scale=float(scale),
               rot_deg=float(math.degrees(math.atan2(R[1, 0], R[0, 0]))),
               translation=t,
               residuals=d[order],
               n_pairs=len(order))


def register(src: np.ndarray, dst: np.ndarray) -> Fit:
    """Multi-start ICP. Bounding boxes disagree because the geolocated truth is
    a subset, so the scale seed is swept rather than trusted."""
    span_s = np.ptp(src, axis=0).max()
    span_d = np.ptp(dst, axis=0).max()
    seed = span_d / span_s if span_s else 1.0

    best = None
    for k in (0.7, 0.85, 1.0, 1.15, 1.3):
        for rot in range(-20, 21, 5):
            f = icp(src, dst, seed * k, rot)
            if best is None or f.median < best.median:
                best = f
    return best


# ── commands ────────────────────────────────────────────────────────────────

def cmd_fit(sheet: str) -> Fit:
    if sheet in CP_SHEETS:
        fit, epsg = fit_from_control_points(sheet)
        OUT.mkdir(exist_ok=True)
        (OUT / f"control_points_{sheet}.json").write_text(json.dumps({
            "sheet": sheet, "derived_by": "scripts/bypass_geoloc.py (existing control-point pairs)",
            "source": str(CP_SHEETS[sheet]), "utm_epsg": epsg, "model": "similarity",
            "scale_m_per_px": fit.scale, "rotation_deg": fit.rot_deg,
            "translation": fit.translation.tolist(), "n_pairs_used": fit.n_pairs,
            "residual_median_m": fit.median, "residual_p90_m": fit.p90}, indent=2))
        print(f"{sheet:<16} scale={fit.scale:6.3f} m/px  rot={fit.rot_deg:+7.2f} deg  "
              f"pairs={fit.n_pairs:5d}  median={fit.median:7.1f} m  p90={fit.p90:7.1f} m")
        return fit

    px_path, geo_path = SHEETS[sheet]
    for p in (px_path, geo_path):
        if not p.exists():
            raise SystemExit(f"missing: {p}")

    src = nodes_from_geojson(px_path)
    dst = nodes_from_geojson(geo_path, to_utm=True)
    fit = register(src, dst)

    OUT.mkdir(exist_ok=True)
    payload = {
        "sheet": sheet,
        "derived_by": "scripts/bypass_geoloc.py (truth->truth registration)",
        "pixel_truth": str(px_path),
        "geolocated_truth": str(geo_path),
        "utm_epsg": UTM,
        "model": "similarity",
        "scale_m_per_px": fit.scale,
        "rotation_deg": fit.rot_deg,
        "translation": fit.translation.tolist(),
        "n_src_nodes": int(len(src)),
        "n_dst_nodes": int(len(dst)),
        "n_pairs_used": fit.n_pairs,
        "residual_median_m": fit.median,
        "residual_p90_m": fit.p90,
    }
    (OUT / f"control_points_{sheet}.json").write_text(json.dumps(payload, indent=2))

    print(f"{sheet:<12} scale={fit.scale:6.3f} m/px  rot={fit.rot_deg:+7.2f} deg  "
          f"pairs={fit.n_pairs:4d}  median={fit.median:7.1f} m  p90={fit.p90:7.1f} m")
    return fit


def cmd_apply(sheet: str, pred: Path, out: Path) -> None:
    import geopandas as gpd
    from shapely.geometry import LineString, Point
    from shapely.ops import transform as shp_transform

    cp = json.loads((OUT / f"control_points_{sheet}.json").read_text())
    fit = Fit(cp["scale_m_per_px"], cp["rotation_deg"],
              np.asarray(cp["translation"], float), np.array([0.0]), 0)

    g = gpd.read_file(pred)
    g["geometry"] = [shp_transform(lambda x, y, z=None: tuple(fit.apply(np.c_[x, y]).T), geom)
                     for geom in g.geometry]
    g = g.set_crs(cp.get("utm_epsg", UTM), allow_override=True).to_crs("EPSG:4326")
    out.parent.mkdir(parents=True, exist_ok=True)
    g.to_file(out, driver="GeoJSON")
    print(f"{sheet}: {len(g)} features -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="derive the pixel->world transform")
    f.add_argument("--sheet", choices=sorted(set(SHEETS) | set(CP_SHEETS)))
    f.add_argument("--all", action="store_true")

    a = sub.add_parser("apply", help="apply a fitted transform to a predicted graph")
    a.add_argument("--sheet", required=True, choices=sorted(set(SHEETS) | set(CP_SHEETS)))
    a.add_argument("--pred", required=True, type=Path)
    a.add_argument("--out", required=True, type=Path)

    args = ap.parse_args()
    if args.cmd == "fit":
        if args.all:
            print(f"{'sheet':<16} {'scale':>13}  {'rotation':>12}  {'pairs':>9}  "
                  f"{'median':>12}  {'p90':>10}")
            for s in sorted(set(SHEETS) | set(CP_SHEETS)):
                try:
                    cmd_fit(s)
                except SystemExit as e:
                    print(f"{s:<12} SKIPPED: {e}")
        elif args.sheet:
            cmd_fit(args.sheet)
        else:
            raise SystemExit("give --sheet or --all")
    else:
        cmd_apply(args.sheet, args.pred, args.out)


if __name__ == "__main__":
    main()
