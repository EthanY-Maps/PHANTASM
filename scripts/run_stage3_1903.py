"""Re-run Stages 1-4 on the 1903 sheet to rebuild the OCR attrition table.

Every intermediate Stage 3 artefact for 1903 was deleted -- `OCR Results`,
`Text Detection` and `OCR Cutouts` are empty in every surviving output directory --
so the attrition funnel (27,856 detections -> 441 validated
intersections -> 33 RANSAC inliers) cannot be recovered from disk and has to be
regenerated.

Model paths are the paper-era ones recorded in
Output/Atlas_1866/Atlas_of_Sydney/*/Local_config.json, not the current defaults:
the 93.5 MB U-Net (models/unet_road_extraction.keras), DBNET_256 for text detection
(models/dbnet_text_detection.keras), and the fine-tuned TrOCR for recognition
(models/trocr_street_names).

Stages 1-2 are included because Stage 4's name propagation needs a graph in the
run's own directory tree; Stage 5 is left off.

    python scripts/run_stage3_1903.py            # full run
    python scripts/run_stage3_1903.py --stages text,ocr,geoloc
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PHD = Path("data")
OUTPUT = PHD / "Output_Repro"

CONFIG = {
    "entries": {
        "input": str(PHD / "Input/Maps of the World/Australia/Sydney/1903/nla.obj-231540665_stitched_h.tif"),
        "output": str(OUTPUT),
        # paper-era checkpoints, per Local_config.json of the Mar-May 2026 run
        "model": str(ROOT / "models/unet_road_extraction.keras"),
        "text_model": str(ROOT / "models/dbnet_text_detection.keras"),
        "ocr": str(ROOT / "models/trocr_street_names"),
        "osm": str(PHD / "Input/Model/sydney_intersections.geojson"),
        "osm_road": str(PHD / "Input/Model/Sydney_Australia_drive_combined.geojson"),
        "expt": "1903_Stage3",
        # footprint of the geolocated 1903 network, rounded outward
        "bounding_coordinate_1": "151.045,-34.020",
        "bounding_coordinate_2": "151.300,-33.717",
    },
    "modules": {"pred": True, "graph": True, "crop": False, "dim": False,
                "text": True, "ocr": True, "geoloc": True, "similarity": False},
    # stride 170 and no CLAHE, per the paper-era Local_config. The May defaults
    # (stride 96, CLAHE on) give 3.1x the tiles for a worse prediction.
    "model_parameters_file": str(ROOT / "config/model_parameters_stage3.json"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", help="comma-separated subset, e.g. text,ocr,geoloc")
    a = ap.parse_args()

    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in CONFIG.items()}
    if a.stages:
        want = {s.strip() for s in a.stages.split(",")}
        cfg["modules"] = {k: (k in want) for k in cfg["modules"]}

    on = [k for k, v in cfg["modules"].items() if v]
    print(f"[1903] stages: {', '.join(on)}", flush=True)
    print(f"[1903] output: {OUTPUT / '1903_Stage3'}", flush=True)

    from pipeline import Pipeline

    t0 = time.time()
    try:
        Pipeline(cfg).run()
    except Exception as e:
        print(f"[FAIL] after {time.time()-t0:.0f}s: {type(e).__name__}: {e}", flush=True)
        raise
    print(f"[DONE] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
