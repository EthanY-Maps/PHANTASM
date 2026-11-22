"""Text detection stage — DBNet++ multi-scale inference on the map image."""

from pathlib import Path
from typing import Dict, Any

from modules._runner import stage_runner, add_paths
add_paths("Text_Detection")

from inference import predict_multi_scale
from utils import (
    find_image_file, get_map_name_from_input,
    read_local_config_output, write_local_config_output,
)


@stage_runner
def run(config: Dict[str, Any], dirs: Dict[str, Path], log, update_status, params) -> None:
    p = params.detection

    input_path = config['entries']['input']
    local_config_path = config['entries']['local_config']
    scale = read_local_config_output(local_config_path, 'Road Extraction', 'scale')

    image_path = find_image_file(input_path)
    if not image_path:
        raise FileNotFoundError(f"No image file found at input path: {input_path}")
    _ = get_map_name_from_input(input_path)

    update_status("Running detection...")
    json_path = predict_multi_scale(
        config['entries']['text_model'], image_path, dirs['text_output'],
        batch_size=p.batch_size,
        patch_size=p.patch_size,
        stride=p.stride,
        confidence_thresh=p.confidence_threshold,
        scales=[scale, 1.0],
        min_area=p.min_area,
        unshrink_ratio=p.unshrink_ratio,
        nms_iou_threshold=p.nms_iou_threshold,
        log=log,
    )
    write_local_config_output(local_config_path, 'Text Detection', {'json_path': str(json_path)})
