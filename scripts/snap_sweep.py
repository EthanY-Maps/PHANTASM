"""Pre-snap against post-snap agreement, swept over the evaluation buffer.

One concern is circularity: the matching scores could be inflated by prediction
and reference both lying on modern OSM geometry. The test is to score two things against
the same reference over a range of tolerances:

  pre-snap   the warped historic graph, which is what the pipeline reconstructs
  post-snap  the stitched OSM paths, which is what Eqs. (9)-(10) actually score

If the post-snap advantage were evidence of better reconstruction it would persist as the
tolerance loosens. If it comes from both sides lying on OSM geometry it should vanish once
the buffer stops discriminating on that basis.

This script regenerates that table.

    python scripts/snap_sweep.py                      # 1903, the headline sheet
    python scripts/snap_sweep.py --sheets Melbourne_1942,Waterloo --tols 5,25,50
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "modules" / "Similarity", ROOT / "_superceded" / "Similarity",
          ROOT / "scripts"):
    sys.path.insert(0, str(p))

PHD = Path("data")
BYPASS = PHD / "Output_Repro/Bypass"
STAGE5 = PHD / "Output_Repro/Stage5/trial151"
SHARED = PHD / "Output_Repro/Stage5"          # the cached reference clips
OUT = PHD / "Output_Repro/SnapSweep"

TOLS = [5, 10, 25, 50, 100, 150, 200]


def evaluate(truth: Path, pred: Path, tol: float) -> float | None:
    from Comparison import evaluate_network
    try:
        df = evaluate_network(str(truth), str(pred), output_folder=None,
                              tolerance_meters=tol)
    except Exception as e:
        print(f"    tol={tol}: FAILED {type(e).__name__}: {e}")
        return None
    if df.empty:
        return None
    return round(float(df.loc["Global_Network"]["F1_Score"]), 4)


def analyse(sheet: str, tols: list[int]) -> dict | None:
    import geopandas as gpd

    post = STAGE5 / sheet / "combined_high_medium.geojson"
    truth = SHARED / sheet / "truth_clipped_fullhull.geojson"
    pre_src = BYPASS / f"{sheet}_geolocated.geojson"
    if not (post.exists() and truth.exists() and pre_src.exists()):
        print(f"  {sheet}: missing inputs")
        return None

    # The pre-snap graph is the warped historic network. Only its LineStrings are
    # comparable to the reference, and it is written in WGS84 while the reference
    # clip is too, so `evaluate_network` reprojects both itself.
    OUT.mkdir(parents=True, exist_ok=True)
    pre = OUT / f"{sheet}_presnap.geojson"
    if not pre.exists():
        g = gpd.read_file(pre_src)
        g = g[g.geom_type.isin(["LineString", "MultiLineString"])]
        g.set_crs("EPSG:4326", allow_override=True).to_file(pre, driver="GeoJSON")

    row = {"sheet": sheet, "tols": {}}
    print(f"\n[{sheet}]", flush=True)
    for t in tols:
        a = evaluate(truth, pre, t)
        b = evaluate(truth, post, t)
        row["tols"][str(t)] = {"pre": a, "post": b,
                               "diff": (round(b - a, 4) if (a is not None and b is not None) else None)}
        d = row["tols"][str(t)]["diff"]
        print(f"  {t:>4} m   pre {a}   post {b}   diff {d:+.4f}" if d is not None
              else f"  {t:>4} m   pre {a}   post {b}", flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", default="Sydney_1903")
    ap.add_argument("--tols", default=",".join(str(t) for t in TOLS))
    a = ap.parse_args()

    tols = [int(t) for t in a.tols.split(",")]
    rows = [r for s in a.sheets.split(",") if (r := analyse(s.strip(), tols))]

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "snap_sweep.json"
    existing = json.loads(out.read_text()) if out.exists() else []
    keep = [r for r in existing if r["sheet"] not in {x["sheet"] for x in rows}]
    out.write_text(json.dumps(keep + rows, indent=2))

    print(f"\n{'='*64}\n{'sheet':<17}{'buffer':>8}{'pre-snap':>11}{'post-snap':>12}{'diff':>10}")
    for r in rows:
        for t, v in r["tols"].items():
            if v["pre"] is None or v["post"] is None:
                continue
            print(f"{r['sheet']:<17}{t+' m':>8}{v['pre']:>11.3f}{v['post']:>12.3f}"
                  f"{v['diff']:>+10.3f}")
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
