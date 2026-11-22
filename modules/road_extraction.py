"""Road extraction stage — U-Net inference on the input map."""

from pathlib import Path
from typing import Dict, Any

from modules._runner import stage_runner, add_paths
add_paths()

from modules.Road_Extraction.Pred import predict_large_map_tta
from utils import write_local_config_output


@stage_runner
def run(config: Dict[str, Any], dirs: Dict[str, Path], log, update_status, params) -> None:
    p = params.road_extraction
    input_path = config['entries']['input']
    log(f"[INFO] Input path: {input_path}")

    update_status("Running road extraction...")
    npy_path, output_parameters = predict_large_map_tta(
        config['entries']['model'],
        input_path,
        str(dirs['predictions']),
        batch_size=p.batch_size,
        window_size=tuple(p.window_size),
        stride=p.stride,
        use_tta=p.use_tta,
        auto_scale=p.auto_scale,
        fixed_scale=p.fixed_scale,
        save_patches=p.save_patches,
        border_margin=p.border_margin,
        skip_empty_patches=p.skip_empty_patches,
        min_patch_std=p.min_patch_std,
        use_clahe=p.use_clahe,
        clahe_clip_limit=p.clahe_clip_limit,
        clahe_grid_size=p.clahe_grid_size,
        log=log,
    )

    write_local_config_output(config['entries']['local_config'], 'Road Extraction', output_parameters)
