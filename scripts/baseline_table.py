"""Join the proximity baseline against the full pipeline, sheet by sheet.

The pipeline column comes from the canonical trial-151 run under `Stage5/trial151/`,
which is also what the manuscript's fourteen-map table, the four-cell decomposition and
the cell-C analysis are computed from, so all of them describe one execution rather than
several of the same configuration. The proximity arms are scored against the same clipped
reference, so the two columns are directly comparable.

    python scripts/baseline_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PHD = Path("data")
PIPE = PHD / "Output_Repro/Stage5/trial151/bypass_results_canonical.json"
PROX = PHD / "Output_Repro/Proximity/proximity_results_all.json"

NAMES = {"St_Peters": "St Peters", "1889_Strathfield": "Strathfield 1889",
         "The_Glebe": "The Glebe", "Melbourne_1942": "Melbourne 1942",
         "Sydney_1903": "Sydney 1903"}


def main() -> None:
    pipe = {r["sheet"]: r for r in json.loads(PIPE.read_text())}
    prox = json.loads(PROX.read_text())

    CONV = ("nearest_trim", "nearest_full", "collinear_trim", "collinear_full")

    # Each convention picks its own best warp arm. Comparing a convention's weak arm
    # against another's strong one would make the choice of convention look more
    # decisive than it is.
    by_sheet: dict[str, dict] = {}
    for r in prox:
        d = by_sheet.setdefault(r["sheet"], {})
        for c in CONV:
            f1 = r.get(c, {}).get("f1")
            if f1 is None:
                continue
            if c not in d or f1 > d[c]["f1"]:
                d[c] = {**r[c], "warp": r["warp"]}

    rows = []
    for sheet, d in by_sheet.items():
        p = pipe.get(sheet)
        if not p or p.get("f1") is None or not d:
            continue
        best_c = max(d, key=lambda c: d[c]["f1"])
        # `*_full` assigns the whole OSM edge, which covers more reference length than
        # the pipeline's paths, trimmed to the source extent. `collinear_trim` is the
        # like-for-like convention. Both are reported: the gap between them is the
        # width of an arbitrary construction choice, and it bears on how much weight
        # the comparison can carry.
        trim = d.get("collinear_trim", {}).get("f1")
        rows.append({"sheet": sheet, "pipeline": p["f1"],
                     "prox": d[best_c]["f1"], "conv": best_c,
                     "warp": d[best_c]["warp"],
                     "prox_trim": trim,
                     "delta": d[best_c]["f1"] - p["f1"],
                     "delta_trim": (trim - p["f1"]) if trim is not None else None,
                     "prox_r": d[best_c]["recall"], "prox_p": d[best_c]["precision"],
                     "pipe_r": p["recall"], "pipe_p": p["precision"]})
    rows.sort(key=lambda r: -r["delta"])

    print(f"{'sheet':<18}{'pipeline':>10}{'prox best':>11}{'delta':>8}"
          f"{'prox trim':>11}{'delta':>8}{'warp':>7}{'convention':>17}")
    for r in rows:
        t = f"{r['prox_trim']:>11.3f}{r['delta_trim']:>+8.3f}" if r["prox_trim"] is not None else f"{'--':>19}"
        print(f"{NAMES.get(r['sheet'], r['sheet']):<18}{r['pipeline']:>10.3f}"
              f"{r['prox']:>11.3f}{r['delta']:>+8.3f}{t}{str(r['warp']):>7}"
              f"{r['conv']:>17}")

    d = np.array([r["delta"] for r in rows])
    print(f"\nproximity (best convention) ahead on {int((d > 0).sum())} of {len(d)} "
          f"sheets; median {np.median(d):+.3f}, range {d.min():+.3f} to {d.max():+.3f}")
    dt = np.array([r["delta_trim"] for r in rows if r["delta_trim"] is not None])
    if len(dt):
        print(f"proximity (trimmed, like-for-like) ahead on {int((dt > 0).sum())} of "
              f"{len(dt)}; median {np.median(dt):+.3f}, range {dt.min():+.3f} to {dt.max():+.3f}")

    out = PHD / "Output_Repro/Proximity/baseline_table.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"written -> {out}")

    print("\nLaTeX rows:")
    for r in rows:
        print(f"{NAMES.get(r['sheet'], r['sheet']):<18} & {r['pipe_r']:.3f} & "
              f"{r['pipe_p']:.3f} & {r['pipeline']:.3f} & {r['prox_r']:.3f} & "
              f"{r['prox_p']:.3f} & {r['prox']:.3f} & ${r['delta']:+.3f}$ \\\\")


if __name__ == "__main__":
    main()
