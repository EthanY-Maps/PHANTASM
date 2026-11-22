"""Similarity stage — match the geolocated historic graph against OSM roads."""

from pathlib import Path
from typing import Dict, Any

from modules._runner import stage_runner, add_paths
add_paths("Similarity")

from HistoricMapMatcher import HistoricMapMatcher
from load_osmnx import load_osmnx

from utils import read_local_config_output, write_local_config_output


@stage_runner
def run(config: Dict[str, Any], dirs: Dict[str, Path], log, update_status, params) -> None:
    matching_config = params.similarity

    local_config_path = config['entries']['local_config']
    osm_road_path = config['entries']['osm_road']
    geolocated_road = read_local_config_output(local_config_path, 'Geolocation', 'geolocated_path')
    similarity_output = dirs['similarity']

    update_status("Running similarity matching...")
    G_hist, G_osm = load_osmnx(str(geolocated_road), osm_road_path)
    matcher = HistoricMapMatcher(G_hist, G_osm, matching_config, crs_meters=matching_config.crs_meters)
    match_results = matcher.run_pipeline(n_jobs=1)
    matcher.export_results(similarity_output, match_results)

    write_local_config_output(local_config_path, 'Similarity', {'output_dir': str(similarity_output)})
