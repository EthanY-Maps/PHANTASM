"""OCR stage — TrOCR inference on text-detection cutouts."""

import os
from pathlib import Path
from typing import Dict, Any

from modules._runner import stage_runner, add_paths
add_paths("OCR")

from TOCR_Inference import run_map_ocr_pipeline
from utils import read_local_config_output, write_local_config_output


@stage_runner
def run(config: Dict[str, Any], dirs: Dict[str, Path], log, update_status, params) -> None:
    p = params.ocr

    local_config_path = config['entries']['local_config']
    input_path = config['entries']['input']
    json_path = read_local_config_output(local_config_path, 'Text Detection', 'json_path')
    output_json = os.path.join(dirs['ocr_results'], "ocr_results.json")

    update_status("Running OCR...")
    output_json = run_map_ocr_pipeline(
        str(input_path), str(json_path), str(output_json),
        config['entries']['ocr'], p.buffer_px, p.batch_size,
    )

    write_local_config_output(local_config_path, 'OCR', {'json_path': output_json})
