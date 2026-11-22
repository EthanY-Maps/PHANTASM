"""Road vectorisation stage — skeletonise + graph the U-Net output."""

from pathlib import Path
from typing import Dict, Any

from modules._runner import stage_runner, add_paths
add_paths()

from modules.Road_Graphing.graphing import Graph_Npy
from utils import write_local_config_output


@stage_runner
def run(config: Dict[str, Any], dirs: Dict[str, Path], log, update_status, params) -> None:
    p = params.road_vectorization

    local_config = config['entries']['local_config']
    pred_dir = dirs['predictions']
    input_path = config['entries']['input']
    input_name = Path(input_path).stem
    npy_raw_dir = Path(pred_dir) / 'npy_raw' / f"{input_name}.npy"
    graph_dir = dirs['graphing']

    update_status("Running road vectorisation...")
    geojson_path = graph_dir / 'roads.geojson'
    G, _stats = Graph_Npy(
        npy_raw_dir, str(graph_dir),
        p.Low_threshold, p.High_threshold, p.spur_threshold,
        log=log,
        geojson_path=str(geojson_path),
        max_hole_size=p.max_hole_size,
        gap_bridge_length=p.gap_bridge_length,
        junction_merge_radius=p.junction_merge_radius,
        min_object_size=p.min_object_size,
        min_skeleton_component=p.min_skeleton_component,
        endpoint_reconnect_radius_factor=p.endpoint_reconnect_radius_factor,
        endpoint_reconnect_max_bearing_deg=p.endpoint_reconnect_max_bearing_deg,
        input_img_path=input_path,
        evidence_enabled=p.evidence_enabled,
        evidence_ink_lum_threshold=p.evidence_ink_lum_threshold,
        evidence_min_casing_fraction=p.evidence_min_casing_fraction,
        evidence_casing_offset_factor=p.evidence_casing_offset_factor,
        evidence_casing_extra_search_px=p.evidence_casing_extra_search_px,
        snake_enabled=p.snake_enabled,
        snake_iterations=p.snake_iterations,
        snake_search_radius_px=p.snake_search_radius_px,
        snake_image_weight=p.snake_image_weight,
        snake_smoothness_weight=p.snake_smoothness_weight,
    )

    log(f"[INFO] Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    update_status(f"Graph completed: {G.number_of_nodes()} nodes")

    write_local_config_output(local_config, 'Road Vectorization',
                              {'geojson_path': str(geojson_path)})
