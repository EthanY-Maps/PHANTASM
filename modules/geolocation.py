"""Geolocation stage — OSM-anchor matching + TPS warp into world coords."""

from pathlib import Path
from typing import Dict, Any

from modules._runner import stage_runner, add_paths
add_paths("Geolocation", "OCR")

from OSM_Intersection_Dictionary import extract_intersection_names
from Post_OCR import post_ocr
from Clean_names import remove_lines, generate_cleaned_names
from Intersection_PostOCR import process_intersections
from Anchor_Nodes import anchor_pipeline
from Road_Propagation import run_analysis

from utils import (
    find_image_file, get_map_name_from_input, parse_bounding_coordinates,
    read_local_config_output, write_local_config_output,
)


@stage_runner
def run(config: Dict[str, Any], dirs: Dict[str, Path], log, update_status, params) -> None:
    p = params.geolocation

    local_config_path = config['entries']['local_config']
    osm_dir = config['entries']['osm']
    input_path = config['entries']['input']

    _ = find_image_file(input_path)
    map_name = get_map_name_from_input(input_path)

    txt_path = dirs['geoloc'] / 'intersection_names.txt'
    street_txt_path = dirs['geoloc'] / 'street_names.txt'
    post_ocr_file = dirs['geoloc'] / 'OCR.geojson'
    geojson_file = read_local_config_output(local_config_path, 'Road Vectorization', 'geojson_path')
    geojson_output = read_local_config_output(local_config_path, 'OCR', 'json_path')
    geolocated_road = dirs['geoloc'] / f"{map_name}_geolocated.geojson"

    update_status("Running geolocation...")

    log("[SUB-STAGE] Geolocation preparation...")
    extract_intersection_names(osm_dir, str(txt_path))
    generate_cleaned_names(str(txt_path), str(street_txt_path))
    remove_lines(str(txt_path))
    post_ocr(str(geojson_output), str(street_txt_path), str(post_ocr_file), "predicted_text",
             threshold=p.fuzzy_match_threshold)

    log("[SUB-STAGE] Road propagation...")
    prop_file = run_analysis(
        str(geojson_file), str(post_ocr_file), dirs["geoloc"],
        min_component_size=p.min_component_size,
        initial_radius=p.propagation_initial_radius,
        propagation_iterations=p.propagation_iterations,
        propagation_angle_tolerance=p.propagation_angle_tolerance,
        propagation_overtake_multiplier=p.propagation_overtake_multiplier,
    )

    log("[SUB-STAGE] Intersection processing...")
    network_file = process_intersections(
        str(prop_file), str(txt_path), dirs["geoloc"],
        fuzzy_weight=p.intersection_fuzzy_weight,
        verification_threshold=p.intersection_verification_threshold,
    )

    bbox1 = config['entries'].get('bounding_coordinate_1', '')
    bbox2 = config['entries'].get('bounding_coordinate_2', '')
    if bbox1 and bbox2:
        log("[SUB-STAGE] Anchor pipeline...")
        boundary_points = parse_bounding_coordinates(bbox1, bbox2)
        anchor_pipeline(
            str(network_file), osm_dir, str(geolocated_road), boundary_points,
            p.RANSAC_THRESHOLD, p.TPS_SMOOTHING,
            match_threshold=p.anchor_match_threshold,
        )
    else:
        log("[WARNING] Bounding coordinates not provided, skipping anchor pipeline")

    write_local_config_output(local_config_path, 'Geolocation',
                              {'geolocated_path': str(geolocated_road)})
