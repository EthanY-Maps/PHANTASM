"""Is atlas pixel recall low because roads are missed, or because the
predicted centreline is thinner than the annotated mask?

Rasterise the paper-era predicted GRAPH at increasing stroke widths and score it
against the annotated pixel mask. If recall climbs steeply with width while the
geometry is unchanged, the low pixel recall in Table 3 is a width artefact, not a
detection failure -- which is what the graph metrics in Table 4 already imply.
"""
import warnings; warnings.filterwarnings("ignore")
import json
import numpy as np, cv2, geopandas as gpd
from pathlib import Path

PHD = Path("data")
A = PHD / "Output/Atlas_1866/Atlas_of_Sydney"
SHEETS = {
    "Annandale":     (A/"1083235_1193-01A-Annandale/Vectorisation/roads.geojson",
                      PHD/"Truth_Data/Annandale/UNET/1083235_1193-01A-Annandale.npy"),
    "Waterloo":      (A/"1083315_1193-41-Waterloo/Vectorisation/roads.geojson",
                      PHD/"Truth_Data/Waterloo/UNET/1083315_1193-41-Waterloo.npy"),
    "Macdonaldtown": (A/"1083281_1193-24A-Macdonaldtown/Vectorisation/roads.geojson",
                      PHD/"Truth_Data/Macdonaldtown/UNET/1083281_1193-24A-Macdonaldtown.npy"),
}
WIDTHS = [1, 3, 5, 7, 9, 11, 13, 15]

rows = []
print(f"{'sheet':<15}{'width':>7}{'recall':>9}{'prec':>8}{'F1':>8}")
for name, (gj, tp) in SHEETS.items():
    if not gj.exists() or not tp.exists():
        print(f"{name:<15}  SKIPPED (missing input)"); continue
    truth = np.load(tp) > 0
    H, W = truth.shape
    g = gpd.read_file(gj)
    g = g[g.geom_type.isin(["LineString", "MultiLineString"])]

    polys = []
    for geom in g.geometry:
        parts = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        for part in parts:
            c = np.asarray(part.coords, float)
            # graph pixel convention: x = col, y = -row
            polys.append(np.column_stack([c[:, 0], -c[:, 1]]).round().astype(np.int32))

    for w in WIDTHS:
        canvas = np.zeros((H, W), np.uint8)
        cv2.polylines(canvas, polys, False, 1, thickness=w)
        pred = canvas > 0
        tp_ = float((pred & truth).sum()); fp = float((pred & ~truth).sum()); fn = float((~pred & truth).sum())
        rec = tp_/(tp_+fn) if tp_+fn else 0; pre = tp_/(tp_+fp) if tp_+fp else 0
        f1 = 2*tp_/(2*tp_+fp+fn) if (2*tp_+fp+fn) else 0
        rows.append({"sheet": name, "width_px": w, "recall": round(rec, 4),
                     "precision": round(pre, 4), "f1": round(f1, 4)})
        print(f"{name if w==WIDTHS[0] else '':<15}{w:>7}{rec:>9.3f}{pre:>8.3f}{f1:>8.3f}")
    print()

# Persisted as well as printed: the first pass only printed, which left the
# manuscript's table resting on transcribed stdout.
OUT = PHD / "Output_Repro/WidthSensitivity"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "width_sensitivity.json").write_text(json.dumps(rows, indent=2))
print(f"written -> {OUT / 'width_sensitivity.json'}")
