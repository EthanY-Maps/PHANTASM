"""Full registration-residual distributions, and a held-out check on them.

`tab:residuals` reports median, IQR, p90 and maximum per map, plus a k-fold check that
the residuals are not inflated by being measured at the points that determine the fit.
The control-point files record only the median and p90, so the rest of that table had no
artefact behind it. This recomputes all of it from the source correspondences.

The fit is the same four-parameter similarity `bypass_geoloc.py` uses: pixel (col, -row)
onto projected control points, by Umeyama. The held-out check refits on a (k-1)/k share
and scores the remaining fold, over k folds, so no scored point contributes to the fit
that predicts it.

    python scripts/residual_stats.py --folds 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

PHD = Path("data")
TRUTHS = PHD / "Input/Maps of the World/Truths"
OUT = PHD / "Output_Repro/Residuals"

# Where each map's control-point pairs live. The atlas sheets share a directory;
# 1903 and Melbourne sit alongside it. Spelling of the folder is the project's.
ATLAS = TRUTHS / "Atlas of Syndey"
EXTRA = {"Sydney_1903": TRUTHS / "control_points_1903_Sydney.json",
         "Melbourne_1942": TRUTHS / "control_points_1942.json",
         "Sydney_1877": TRUTHS / "control_points_1877.json"}


def sources() -> dict[str, Path]:
    out = dict(EXTRA)
    if ATLAS.is_dir():
        for f in sorted(ATLAS.glob("control_points_*.json")):
            out.setdefault(f.stem.replace("control_points_", "").replace(" ", "_"), f)
    return {k: v for k, v in out.items() if v.exists()}


def pairs(path: Path):
    from pyproj import Transformer
    cp = json.loads(path.read_text())
    P = np.array([q["pixel_yx"] for q in cp["pairs"]], float)
    L = np.array([q["osm_lonlat"] for q in cp["pairs"]], float)
    epsg = cp.get("utm_epsg", "EPSG:32756")
    tr = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    dst = np.column_stack(tr.transform(L[:, 0], L[:, 1]))
    src = np.column_stack([P[:, 1], -P[:, 0]])       # (col, -row)
    return src, dst


def fit_predict(src, dst, query):
    from bypass_geoloc import umeyama
    s, R, t = umeyama(src, dst)
    return s * (query @ R.T) + t


def stats(r: np.ndarray) -> dict:
    return {"n": int(len(r)),
            "median_m": round(float(np.median(r)), 2),
            "q1_m": round(float(np.percentile(r, 25)), 2),
            "q3_m": round(float(np.percentile(r, 75)), 2),
            "p90_m": round(float(np.percentile(r, 90)), 2),
            "max_m": round(float(np.max(r)), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    rows = []
    for sheet, path in sorted(sources().items()):
        try:
            src, dst = pairs(path)
        except Exception as e:
            print(f"  {sheet}: FAILED {type(e).__name__}: {e}")
            continue
        if len(src) < a.folds:
            print(f"  {sheet}: only {len(src)} pairs, skipped")
            continue

        insample = np.linalg.norm(fit_predict(src, dst, src) - dst, axis=1)

        # k-fold held out: refit without each fold, predict it
        idx = rng.permutation(len(src))
        held = np.empty(len(src))
        for f in range(a.folds):
            test = idx[f::a.folds]
            train = np.setdiff1d(idx, test)
            held[test] = np.linalg.norm(
                fit_predict(src[train], dst[train], src[test]) - dst[test], axis=1)

        row = {"sheet": sheet, "source": str(path),
               "in_sample": stats(insample), "held_out": stats(held),
               "folds": a.folds}
        rows.append(row)
        print(f"  {sheet:<18} n={row['in_sample']['n']:>5}  "
              f"median {row['in_sample']['median_m']:>7.2f} in / "
              f"{row['held_out']['median_m']:>7.2f} held   "
              f"p90 {row['in_sample']['p90_m']:>7.2f}   "
              f"max {row['in_sample']['max_m']:>9.2f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "residual_stats.json").write_text(json.dumps(rows, indent=2))

    print(f"\n{'='*96}\n{'map':<18}{'pts':>7}{'median':>9}{'IQR':>16}{'p90':>9}"
          f"{'max':>11}{'held-out med':>14}")
    for r in rows:
        i, h = r["in_sample"], r["held_out"]
        iqr = "{:.1f}-{:.1f}".format(i["q1_m"], i["q3_m"])
        print(f"{r['sheet']:<18}{i['n']:>7}{i['median_m']:>9.1f}{iqr:>16}"
              f"{i['p90_m']:>9.1f}{i['max_m']:>11.1f}{h['median_m']:>14.1f}")
    print(f"\nwritten -> {OUT/'residual_stats.json'}")


if __name__ == "__main__":
    main()
