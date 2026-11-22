# PHANTASM

Code for **PHANTASM: A Pipeline for Historic Analysis, Network Transformation, and Spatial Matching** (Yin and Levinson, *Computers, Environment and Urban Systems*). The pipeline turns a scanned historic street map into a georeferenced road network and matches each historic road to its present-day OpenStreetMap counterpart.

This repository holds the code as the paper describes it. The trained weights, and the processed outputs and ground-truth annotations behind the paper's tables, are attached to the [v1.0.0 release](https://github.com/EthanY-Maps/PHANTASM/releases/tag/v1.0.0).

## Stages

| Stage | Module | What it does |
|---|---|---|
| 1. Road extraction | `modules/road_extraction.py`, `modules/Road_Extraction/` | U-Net sliding-window inference gives a road probability map |
| 2. Vectorisation | `modules/road_vectorization.py`, `modules/Road_Graphing/` | Hysteresis thresholding and skeletonisation give a road graph (GeoJSON) |
| 3. Text detection and OCR | `modules/detection.py`, `modules/Text_Detection/`, `modules/ocr.py`, `modules/OCR/` | DBNet++ finds street labels and a fine-tuned TrOCR reads them |
| 4. Geolocation | `modules/geolocation.py`, `modules/Geolocation/` | Street names matched to OSM intersections give anchors; RANSAC and a thin-plate spline warp the graph to latitude and longitude |
| 5. Similarity | `modules/similarity.py`, `modules/Similarity/` | Each historic road is scored against OSM roads and matched |

`stages.py` lists the stages, `pipeline.py` runs them, and `model_parameters.py` defines every tunable parameter. `config/model_parameters_paper.json` holds the parameter values used in the paper.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`pdf2image` also needs [Poppler](https://poppler.freedesktop.org/) on your system (for example `brew install poppler` on macOS or `apt install poppler-utils` on Debian and Ubuntu).

`requirements.txt` does not pin versions. The reproduction scripts were run in an environment with Python 3.9, TensorFlow 2.16, PyTorch 2.8, transformers 4.57, GeoPandas 1.0, OSMnx 2.0 and scikit-image 0.24.

## Trained weights

Download the release assets into a `models/` folder at the repository root:

```
models/
  unet_road_extraction.keras      Stage 1 U-Net, trained on the 1877 John Sands map of Sydney
  dbnet_text_detection.keras      Stage 3 DBNet++ text detector
  trocr_street_names/             Stage 3 TrOCR, fine-tuned from microsoft/trocr-base-handwritten
```

`trocr_street_names.zip` unpacks to `trocr_street_names/`. `SHA256SUMS.txt` on the release lists a checksum for each file. The TrOCR weights were saved with `transformers` 4.57.1.

## Running the pipeline

1. Copy `config/map_pipeline_config.example.json` to `config/map_pipeline_config.json`.
2. Set `input` (a map image, PDF or folder of maps), `output`, `osm` (OSM intersections as GeoJSON) and `osm_road` (OSM roads as GeoJSON). The model entries already point at `models/`.
3. Start the GUI, choose the stages to run and press Run:

```bash
python launchpad.py
```

Each map gets its own output folder holding the probability map, the road graph, the text detections and OCR results, the georeferenced graph and the OSM matches.

## Reproducing the paper's experiments

The scripts in `scripts/` regenerate the paper's tables and figures. They read inputs from a `data/` folder at the repository root. Unzipping `phantasm_data.zip` from the release at the repository root creates that folder with the fourteen geolocated graphs, the control points, the Stage 5 matches and per-link classes, the 1877 truth graph, the road masks and the OCR test crops; `DATA.md` inside the archive lists each file and the table it supports. Map scans, OpenStreetMap extracts and the Turner et al. (2023) reference network are not redistributed; place them under `data/Input/` and `data/GreaterSydneyStreetsEvolution/` at the paths the scripts name.

| Script | Purpose |
|---|---|
| `run_stage12_atlas.py` | Stages 1 and 2 on the 1886-1889 Atlas of Sydney sheets |
| `run_stage3_1903.py` | Stages 1 to 4 on the 1903 sheet, including the OCR attrition counts |
| `bypass_geoloc.py` | Georeferences sheets from their manual truth graphs, so Stage 5 can be measured without Stage 4 |
| `run_stage5_bypass.py` | Stage 5 on those sheets, scored against Turner et al. (2023) |
| `tune_matching.py`, `verify_trial151.py`, `plot_pareto.py` | The NSGA-II search over Stage 5 weights, its selected operating point and the Pareto figure |
| `inlier_sweep.py`, `ransac_stability.py`, `sweep_summary.py`, `warp_residual_sweep.py`, `residual_stats.py`, `affine_only_baseline.py` | Stage 4 registration: RANSAC threshold, draw-to-draw stability, affine against thin-plate spline |
| `proximity_baseline.py`, `baseline_table.py`, `proximity_vs_pipeline.py`, `excess_distance.py`, `match_breakdown.py`, `cell_c_analysis.py`, `cell_c_clustering.py`, `snap_sweep.py` | Stage 5 baselines and the breakdown of matched and unmatched links |
| `ocr_accuracy.py` | TrOCR accuracy on held-out labelled crops |
| `width_sensitivity.py` | Pixel recall of the predicted graph against stroke width |

`_superceded/Similarity/` holds the original Stage 5 tuning (`Tuning.py`) and buffer evaluation (`Comparison.py`) code, kept unchanged because the paper's figures came from it. `evaluation_code/` holds the Stage 1 and graph evaluation scripts.

## Citation

```bibtex
@article{yin_phantasm,
  author  = {Yin, Ethan and Levinson, David},
  title   = {{PHANTASM}: A Pipeline for Historic Analysis, Network Transformation, and Spatial Matching},
  journal = {Computers, Environment and Urban Systems},
  year    = {2026}
}
```

## License

MIT; see [LICENSE](LICENSE). The weights on the release page are released under the same terms.
