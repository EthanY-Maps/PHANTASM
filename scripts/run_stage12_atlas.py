"""Re-run Stages 1-2 (U-Net + vectorisation) on the atlas sheets, paper-era.

The graphs under Output_2/ were produced by SegFormer in August and cannot stand
in for the paper's U-Net. This regenerates them with models/unet_road_extraction.keras
so that bypass_geoloc.py apply -> Stage 5 measures the pipeline the paper describes.

Each sheet is run on its BASE .tif, not a Cal/RoadCal variant: the pixel-space
Truth.geojson graphs were digitised on the base raster (the truth bbox fills
~96% of it in every case), and the fitted transforms are tied to that pixel space.

    python scripts/run_stage12_atlas.py --list
    python scripts/run_stage12_atlas.py --sheet Annandale
    python scripts/run_stage12_atlas.py --all
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
ATLAS = PHD / "Input/Maps of the World/Australia/Sydney/1886-1889 Atlas of Sydney"
OUTPUT = PHD / "Output_Repro"
UNET = ROOT / "models/unet_road_extraction.keras"

# sheet -> base raster (the one the truth graph was digitised on)
SHEETS = {
    "Annandale":     ATLAS / "Annandale/1083235_1193-01A-Annandale.tif",
    "Waterloo":      ATLAS / "Waterloo/1083315_1193-41-Waterloo.tif",
    "Alexandria":    ATLAS / "Alexandria/1083231_1193-01-Alexandria.tif",
    "Camperdown":    ATLAS / "Camperdown/1083245_1193-09-Camperdown.tif",
    "Glebe":         ATLAS / "The Glebe/1083264_1193-18-The Glebe.tif",
    "Paddington":    ATLAS / "Paddington/1083293_1193-29-Paddington.tif",
    # no geolocated truth, so it cannot be bypassed -- run anyway for Stages 1-2
    "Macdonaldtown": ATLAS / "Macdonaldtown/1083281_1193-24A-Macdonaldtown.tif",
}


def build_config(sheet: str) -> dict:
    return {
        "entries": {
            "input": str(SHEETS[sheet]),
            "output": str(OUTPUT),
            "model": str(UNET),
            "osm": str(PHD / "Input/Model/sydney_intersections.geojson"),
            "osm_road": str(PHD / "Input/Model/Sydney/sydney_drive.graphml"),
            "expt": f"Atlas_Repro/{sheet}",
            "bounding_coordinate_1": "",
            "bounding_coordinate_2": "",
        },
        # Stage 4 is deliberately off: it is what the bypass replaces.
        "modules": {"pred": True, "graph": True, "crop": False, "dim": False,
                    "text": False, "ocr": False, "geoloc": False, "similarity": False},
        "model_parameters_file": str(ROOT / "config/model_parameters_paper.json"),
    }


def run_sheet(sheet: str) -> bool:
    from pipeline import Pipeline

    src = SHEETS[sheet]
    if not src.exists():
        print(f"[SKIP] {sheet}: missing {src}")
        return False

    print(f"\n{'='*70}\n[RUN] {sheet}  <- {src.name}\n{'='*70}", flush=True)
    t0 = time.time()
    try:
        Pipeline(build_config(sheet)).run()
    except Exception as e:
        print(f"[FAIL] {sheet}: {type(e).__name__}: {e}", flush=True)
        return False
    print(f"[DONE] {sheet} in {time.time()-t0:.1f}s", flush=True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", choices=sorted(SHEETS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        for s, p in SHEETS.items():
            if p.exists():
                with Image.open(p) as im:
                    print(f"  {s:<14} {im.size[0]:>6} x {im.size[1]:<6} "
                          f"({im.size[0]*im.size[1]/1e6:5.1f} MP)  {p.name}")
            else:
                print(f"  {s:<14} MISSING")
        return

    targets = sorted(SHEETS) if a.all else ([a.sheet] if a.sheet else [])
    if not targets:
        raise SystemExit("give --sheet, --all, or --list")

    ok = [s for s in targets if run_sheet(s)]
    print(f"\n{'='*70}\n[SUMMARY] {len(ok)}/{len(targets)} succeeded: {', '.join(ok)}")


if __name__ == "__main__":
    main()
