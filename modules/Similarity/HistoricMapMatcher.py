"""Match a geolocated historic road graph against an OSM road network.

Scoring combines Hausdorff shape distance, end-to-end bearing, sinuosity, and
a topology penalty for sharp turns on otherwise-straight historic edges.
"""

from __future__ import annotations

import os
import sys
import warnings
import json
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils import save_json

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
from scipy.spatial import cKDTree
from scipy.spatial.distance import directed_hausdorff
from shapely.geometry import LineString, MultiLineString, Point, box
from shapely.ops import substring, transform
from shapely.strtree import STRtree 

from skimage.transform import AffineTransform
from skimage.measure import ransac 

from tqdm import tqdm
from joblib import Parallel, delayed

try:
    from model_parameters import SimilarityParameters as MatchingConfig
except ModuleNotFoundError:
    _PIPELINE_ROOT = Path(__file__).resolve().parents[2]
    if str(_PIPELINE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PIPELINE_ROOT))
    from model_parameters import SimilarityParameters as MatchingConfig

warnings.filterwarnings("ignore")

# ── 1. Helpers ──

def auto_utm_epsg(lon: float, lat: float) -> str:
    """
    Pick the WGS84 UTM EPSG code for a given lon/lat. Accurate to ~0.04% for
    points within the chosen zone (~500 km wide). Range 32601-32660 (north),
    32701-32760 (south).
    """
    zone = int((lon + 180) / 6) + 1
    base = 32700 if lat < 0 else 32600
    return f"EPSG:{base + zone}"


def _graph_centroid_lonlat(G) -> Tuple[float, float]:
    """Mean (lon, lat) of all nodes in a WGS84 graph. Used to auto-pick a UTM zone."""
    xs = [d['x'] for _, d in G.nodes(data=True) if 'x' in d]
    ys = [d['y'] for _, d in G.nodes(data=True) if 'y' in d]
    if not xs or not ys:
        raise ValueError("Cannot derive UTM zone: graph has no nodes with x/y attributes.")
    return float(np.mean(xs)), float(np.mean(ys))


def calculate_bearing(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0: return 0.0
    angle = np.degrees(np.arctan2(dy, dx))
    if angle < 0: angle += 360
    return angle % 180 

def get_geometry_coords(geom) -> List[Tuple[float, float]]:
    if geom is None: return []
    if isinstance(geom, LineString): return list(geom.coords)
    if isinstance(geom, MultiLineString):
        coords = []
        for part in geom.geoms: coords.extend(list(part.coords))
        return coords
    return []

OSMNX_RESERVED_NODE_ATTRS = {"osmid"}

def _reproject_geographic_edge_geoms(G: nx.MultiDiGraph, target_crs: str) -> None:
    """
    Reproject edge geometry attributes from WGS84 to target_crs in-place.
    Only acts when the first edge geometry has coordinates in geographic range
    (lon/lat values), meaning ox.project_graph didn't reproject them.
    """
    from pyproj import Transformer
    sample_geom = next(
        (d['geometry'] for _, _, d in G.edges(data=True) if 'geometry' in d), None
    )
    if sample_geom is None:
        return
    first_x = list(sample_geom.coords)[0][0]
    if not (-360 < first_x < 360):
        return  # Already projected to a metric CRS
    transformer = Transformer.from_crs('EPSG:4326', target_crs, always_xy=True)
    for _, _, d in G.edges(data=True):
        geom = d.get('geometry')
        if isinstance(geom, LineString) and len(geom.coords) >= 2:
            xs, ys = zip(*geom.coords)
            new_xs, new_ys = transformer.transform(xs, ys)
            d['geometry'] = LineString(zip(new_xs, new_ys))
        elif isinstance(geom, MultiLineString):
            new_parts = []
            for part in geom.geoms:
                xs, ys = zip(*part.coords)
                new_xs, new_ys = transformer.transform(xs, ys)
                new_parts.append(LineString(zip(new_xs, new_ys)))
            d['geometry'] = MultiLineString(new_parts)

def drop_reserved_node_attrs(graph: nx.MultiDiGraph) -> None:
    """Remove node attrs that collide with OSMnx GeoDataFrame index names."""
    for _, attrs in graph.nodes(data=True):
        for attr in OSMNX_RESERVED_NODE_ATTRS:
            attrs.pop(attr, None)

# ── 2. Config & Data ──

class MatchConfidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"
    REJECTED = "rejected"

@dataclass
class MatchResult:
    source_edge: Tuple[int, int, int]
    target_path: List[int]
    score: float
    confidence: MatchConfidence
    metrics: Dict[str, float] = field(default_factory=dict)
    displacement_vector: Optional[Tuple[float, float]] = None
    metadata: Dict = field(default_factory=dict)
    direction: str = "forward"
    match_geometry: Optional[LineString] = None 

# ── 3. High-Performance Indexes ──

class SpatialIndex:
    def __init__(self, nodes_gdf: gpd.GeoDataFrame):
        self.node_ids = nodes_gdf.index.tolist()
        coords = np.column_stack((nodes_gdf.geometry.x.values, nodes_gdf.geometry.y.values))
        self.kdtree = cKDTree(coords)
    
    def find_nearest_nodes(self, point: np.ndarray, radius: float, max_results: int = 30):
        if point.ndim == 1: point = point.reshape(1, -1)
        distances, indices = self.kdtree.query(point, k=min(max_results, len(self.node_ids)), distance_upper_bound=radius)
        if distances.ndim == 1: distances, indices = distances.reshape(1, -1), indices.reshape(1, -1)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if np.isfinite(dist) and idx < len(self.node_ids):
                results.append((self.node_ids[int(idx)], float(dist)))
        return sorted(results, key=lambda x: x[1])

class FastEdgeIndex:
    def __init__(self, graph: nx.MultiDiGraph):
        self.edge_keys = []   
        self.geometries = []  
        self.bearings = []    
        
        for u, v, k, d in graph.edges(keys=True, data=True):
            if 'geometry' in d:
                geom = d['geometry']
                coords = geom.coords
                if len(coords) >= 2:
                    self.edge_keys.append((u, v, k))
                    self.geometries.append(geom)
                    self.bearings.append(calculate_bearing(coords[0], coords[-1]))
        
        self.tree = STRtree(self.geometries) if self.geometries else None
        self.bearings = np.array(self.bearings, dtype=np.float32)

    def bulk_query(self, points: List[Point], radius: float):
        if not self.tree: return [[] for _ in points]
        boxes = [box(p.x - radius, p.y - radius, p.x + radius, p.y + radius) for p in points]
        query_res = self.tree.query(boxes)
        grouped = [[] for _ in points]
        if query_res.size > 0:
            q_indices = query_res[0]
            t_indices = query_res[1]
            sorter = np.argsort(q_indices)
            q_indices, t_indices = q_indices[sorter], t_indices[sorter]
            split_idx = np.unique(q_indices, return_index=True)[1][1:]
            groups = np.split(t_indices, split_idx)
            unique_q = np.unique(q_indices)
            for q_idx, group in zip(unique_q, groups):
                grouped[q_idx] = group
        return grouped

# ============================================================================
# 4. Scorer & Matcher Logic (UPDATED)
# ============================================================================

class PathScorer:
    def __init__(self, config: MatchingConfig):
        self.config = config

    def _calculate_max_internal_angle(self, geom: LineString) -> float:
        """
        Calculates the maximum deviation angle (in degrees) found within the linestring.
        Vectorized for speed.
        """
        coords = np.array(geom.coords)
        if len(coords) < 3: return 0.0
        
        # Create vectors: V1 (p_curr - p_prev), V2 (p_next - p_curr)
        # We want the angle between the incoming vector and the outgoing vector
        v1 = coords[1:-1] - coords[:-2]
        v2 = coords[2:] - coords[1:-1]
        
        norm1 = np.linalg.norm(v1, axis=1)
        norm2 = np.linalg.norm(v2, axis=1)
        
        # Avoid division by zero
        valid_mask = (norm1 > 1e-6) & (norm2 > 1e-6)
        if not np.any(valid_mask): return 0.0
        
        v1 = v1[valid_mask]
        v2 = v2[valid_mask]
        norm1 = norm1[valid_mask]
        norm2 = norm2[valid_mask]
        
        # Dot product
        dot_products = np.sum(v1 * v2, axis=1)
        cosine_angles = np.clip(dot_products / (norm1 * norm2), -1.0, 1.0)
        
        # Calculate angle in degrees
        angles = np.degrees(np.arccos(cosine_angles))
        
        return np.max(angles)

    def score_geometry(self, match_geom: LineString, src_coords: List[Tuple], src_len: float) -> Tuple[float, Dict]:
        path_coords = list(match_geom.coords)
        match_len = match_geom.length
        
        # --- 1. Soft Length Penalty ---
        len_ratio = match_len / max(src_len, 1.0)
        len_penalty = 0.0
        if len_ratio < self.config.min_len_ratio:
            len_penalty += (self.config.min_len_ratio - len_ratio) * 5.0
        elif len_ratio > self.config.max_len_ratio:
            len_penalty += (len_ratio - self.config.max_len_ratio) * 0.5

        # --- 2. Soft Bearing Penalty ---
        v1 = np.array(src_coords[-1]) - np.array(src_coords[0])
        v2 = np.array(path_coords[-1]) - np.array(path_coords[0])
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0: bearing_score = 0.5
        else:
            dot = np.clip(np.dot(v1/norm1, v2/norm2), -1.0, 1.0)
            bearing_score = np.arccos(dot) / np.pi
            
        bearing_penalty = 0.0
        if bearing_score > self.config.bearing_hard_penalty_threshold:
            bearing_penalty = (bearing_score - self.config.bearing_hard_penalty_threshold) * 2.0

        # --- 3. Weighted Hausdorff (Standard) ---
        d_fwd = directed_hausdorff(src_coords, path_coords)[0]
        d_bwd = directed_hausdorff(path_coords, src_coords)[0]
        
        r_fwd = d_fwd / src_len if src_len > 0 else 1.0
        r_bwd = (d_bwd / src_len) * 0.2 if src_len > 0 else 1.0
        combined = r_fwd + r_bwd
        
        if combined < 0.1: shape_score = 0.0
        elif combined > 0.5: shape_score = 1.0
        else: shape_score = (combined - 0.1) / 0.4

        # --- 3b. Absolute displacement penalty ---
        # shape_score above is normalised by link length AND saturates, so it cannot
        # distinguish a match 50 m away from one 500 m away. d_fwd is already in metres;
        # penalise it directly, unsaturated, beyond a cap set from the sheet's
        # registration residual.
        offset_penalty = 0.0
        _cap = getattr(self.config, "source_offset_cap_m", 0.0)
        if _cap > 0:
            # MEDIAN offset, not Hausdorff. d_fwd is a maximum over vertices, so a
            # correct match that diverges at one corner is penalised as hard as a
            # match onto the next street over; using it rejected five good matches
            # per bad one. The median measures typical separation, which is what
            # actually separates "on the road" from "one street away".
            _pl = LineString(path_coords)
            _sc = np.asarray(src_coords, dtype=float)   # may arrive as a list
            _pts = _sc if len(_sc) <= 12 else _sc[
                np.linspace(0, len(_sc) - 1, 12).astype(int)]
            _med = float(np.median([_pl.distance(Point(c)) for c in _pts]))
            _over = _med - _cap
            if _over > 0:
                offset_penalty = (_over / _cap) * getattr(
                    self.config, "source_offset_weight", 0.0)

        # --- 4. Sinuosity Penalty (NEW) ---
        # Checks if the path is zig-zagging unnecessarily
        start_pt = np.array(path_coords[0])
        end_pt = np.array(path_coords[-1])
        euclidean_dist = np.linalg.norm(end_pt - start_pt)
        
        if euclidean_dist > 1e-3:
            sinuosity = match_len / euclidean_dist
        else:
            sinuosity = 1.0
            
        sinuosity_penalty = 0.0
        # If path is 10% longer than air-distance, start penalizing
        if sinuosity > 1.10: 
            sinuosity_penalty = (sinuosity - 1.10) * 1.5 
            # Cap it so it doesn't explode completely
            sinuosity_penalty = min(sinuosity_penalty, 1.0)

        # --- 5. Topology/Dogleg Penalty (NEW) ---
        # If historic line is straight, punish sharp turns in match
        topology_penalty = 0.0
        
        src_euclidean = np.linalg.norm(np.array(src_coords[-1]) - np.array(src_coords[0]))
        src_sinuosity = src_len / src_euclidean if src_euclidean > 1e-3 else 1.0
        
        # Only apply if historic line is relatively straight
        if src_sinuosity < 1.15:
            max_turn = self._calculate_max_internal_angle(match_geom)
            # 45 degrees is a significant turn for a "straight" road
            if max_turn > 45.0:
                # Add penalty proportional to severity of turn
                topology_penalty = (max_turn - 45.0) / 180.0
                
        # --- FINAL SCORE ---
        base_score = (shape_score * self.config.shape_weight + 
                      bearing_score * self.config.bearing_weight +
                      sinuosity_penalty * self.config.sinuosity_weight +
                      topology_penalty * self.config.topology_weight)
        
        final_score = base_score + len_penalty + bearing_penalty + offset_penalty
        
        metrics = {
            'shape': round(shape_score, 3), 
            'bearing': round(bearing_score, 3), 
            'sinuosity': round(sinuosity, 3),
            'topology': round(topology_penalty, 3)
        }
        return float(final_score), metrics

    def calculate_flow_score(self, current_vector, history_deque) -> float:
        if len(history_deque) < 3: return 0.3
        hist = np.array(list(history_deque))
        avg = np.mean(hist, axis=0)
        curr = np.array(current_vector)
        mag_diff = abs(np.linalg.norm(curr) - np.linalg.norm(avg))
        mag_score = min(1.0, mag_diff / self.config.drift_tolerance_meters)
        if np.linalg.norm(curr) > 0 and np.linalg.norm(avg) > 0:
            cosine = np.dot(curr, avg) / (np.linalg.norm(curr) * np.linalg.norm(avg))
            dir_score = (1.0 - cosine) / 2.0
        else:
            dir_score = 0.5
        return mag_score * self.config.flow_magnitude_weight + dir_score * self.config.flow_direction_weight

class HistoricMapMatcher:
    def __init__(self, historic_graph, osm_graph, config: MatchingConfig,
                 crs_meters: Optional[str] = None, filter_highway_types=None):
        self.config = config
        # If no metric CRS given, auto-pick the UTM zone for the historic graph's
        # centroid. Hard-coding (e.g. "EPSG:3308" for NSW) silently produces
        # wrong distances anywhere outside that zone.
        if crs_meters is None:
            lon, lat = _graph_centroid_lonlat(historic_graph)
            crs_meters = auto_utm_epsg(lon, lat)
            print(f"  -> Auto-selected metric CRS: {crs_meters} (centroid lon={lon:.3f}, lat={lat:.3f})")
        self.crs_meters = crs_meters
        historic_graph = historic_graph.copy()
        drop_reserved_node_attrs(historic_graph)
        
        print("Initializing V5.3 Matcher (Sinuosity & Topology Fix)...")
        
        # 1. Determine Target Types
        # If no filter is provided, fallback to the config defaults (which exclude motorways)
        # instead of accepting ALL edges.
        target_types = filter_highway_types if filter_highway_types else self.config.highway_types

        # 2. Define Explicit Blacklist
        # Hard constraint: Never allow motorways, even if passed in custom filters.
        blacklist_types = {'motorway', 'motorway_link'}
        
        # 3. Filter Edges
        valid_edges = []
        for u, v, k, d in osm_graph.edges(keys=True, data=True):
            highway_val = d.get('highway')
            
            # Check A: Is it in our allowed list?
            if self._check_highway(highway_val, target_types):
                # Check B: Is it strictly NOT in the blacklist?
                if not self._check_highway(highway_val, blacklist_types):
                    valid_edges.append((u, v, k))

        # Create the subgraph based on filtered edges
        self.G_osm = osm_graph.edge_subgraph(valid_edges).copy()
        drop_reserved_node_attrs(self.G_osm)

        # Project Graphs
        self.G_hist = ox.project_graph(historic_graph, to_crs=crs_meters)
        self.G_osm = ox.project_graph(self.G_osm, to_crs=crs_meters)
        drop_reserved_node_attrs(self.G_hist)
        drop_reserved_node_attrs(self.G_osm)
        # ox.project_graph updates node x,y but may not reproject edge geometry attrs.
        # Ensure any edge geometries still in geographic coordinates are reprojected.
        _reproject_geographic_edge_geoms(self.G_hist, crs_meters)
        _reproject_geographic_edge_geoms(self.G_osm, crs_meters)
        
        print(f"  -> OSM Graph filtered. {len(self.G_osm.edges)} edges remaining (Motorways removed).")

        print("Building Indexes...")
        self.osm_nodes_gdf = ox.graph_to_gdfs(self.G_osm, edges=False)
        self.node_index = SpatialIndex(self.osm_nodes_gdf)
        self.edge_index = FastEdgeIndex(self.G_osm)

        self.node_geoms = {n: Point(d['x'], d['y']) if 'x' in d else d['geometry']
                           for n, d in self.G_osm.nodes(data=True)}
        self.osm_node_coords = {n: (p.x, p.y) for n, p in self.node_geoms.items()}
        self.scorer = PathScorer(self.config)
        self.his_nodes = ox.graph_to_gdfs(self.G_hist, edges=False)

    def run_pipeline(self, n_jobs=1, save_warped_path=None) -> List[MatchResult]:
        edge_data = self._prepare_edge_data()

        if self.config.enable_registration:
            print("[Pass 0] Registration (KDTree Node Match)...")
            probe_results = Parallel(n_jobs=n_jobs)(
                delayed(self._probe_node_fast)(item) for item in tqdm(edge_data)
            )
            self._apply_warp(probe_results)
            edge_data = self._prepare_edge_data() 
            if save_warped_path:
                warped_edges = ox.graph_to_gdfs(self.G_hist, nodes=False)
                warped_edges.to_file(save_warped_path, driver="GeoJSON", index=False)

        print("[Pass 1] Batch Query & Match...")
        all_starts = [item['start_pt'] for item in edge_data]
        all_ends = [item['end_pt'] for item in edge_data]
        
        s_cands_list = self.edge_index.bulk_query(all_starts, self.config.initial_search_radius)
        e_cands_list = self.edge_index.bulk_query(all_ends, self.config.initial_search_radius)
        
        tasks = []
        for i, item in enumerate(edge_data):
            tasks.append((item, s_cands_list[i], e_cands_list[i], self.config.initial_search_radius))
            
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._solve_edge_batch)(*task) for task in tqdm(tasks)
        )
        
        results = self._apply_flow(results)
        
        rejected_indices = [i for i, r in enumerate(results) if r.confidence == MatchConfidence.REJECTED]
        if rejected_indices:
            print(f"[Pass 2] Smart Relaxation on {len(rejected_indices)} edges...")
            
            relax_starts = [edge_data[i]['start_pt'] for i in rejected_indices]
            relax_ends = [edge_data[i]['end_pt'] for i in rejected_indices]
            
            s_cand_relax = self.edge_index.bulk_query(relax_starts, self.config.relaxation_search_radius)
            e_cand_relax = self.edge_index.bulk_query(relax_ends, self.config.relaxation_search_radius)
            
            relax_tasks = []
            for k, original_idx in enumerate(rejected_indices):
                relax_tasks.append((edge_data[original_idx], s_cand_relax[k], e_cand_relax[k], self.config.relaxation_search_radius))
                
            recovered = Parallel(n_jobs=n_jobs)(
                delayed(self._solve_edge_batch)(*task, is_relax=True) for task in tqdm(relax_tasks)
            )
            recovered = self._apply_flow(recovered)
            
            for k, original_idx in enumerate(rejected_indices):
                if recovered[k].confidence != MatchConfidence.REJECTED:
                    results[original_idx] = recovered[k]

        return results

    def _solve_edge_batch(self, item, s_indices, e_indices, radius, is_relax=False):
        if len(s_indices) == 0 or len(e_indices) == 0: return self._reject(item['edge'])

        # Filter by Bearing
        s_mask = np.abs(self.edge_index.bearings[s_indices] - item['bearing'])
        s_mask = np.minimum(s_mask, 180 - s_mask) < self.config.pre_filter_bearing_diff
        e_mask = np.abs(self.edge_index.bearings[e_indices] - item['bearing'])
        e_mask = np.minimum(e_mask, 180 - e_mask) < self.config.pre_filter_bearing_diff
        
        valid_s_idx = s_indices[s_mask]
        valid_e_idx = e_indices[e_mask]
        
        if len(valid_s_idx) == 0 or len(valid_e_idx) == 0:
            if not is_relax: return self._reject(item['edge'])
            valid_s_idx, valid_e_idx = s_indices, e_indices 

        # Hydrate
        s_cands = self._hydrate_candidates(valid_s_idx, item['start_pt'])
        e_cands = self._hydrate_candidates(valid_e_idx, item['end_pt'])
        
        # Limit candidates
        limit = self.config.relaxation_max_candidates if is_relax else self.config.max_candidates_edges
        s_cands.sort(key=lambda x: x['dist'])
        e_cands.sort(key=lambda x: x['dist'])
        s_cands = s_cands[:limit]
        e_cands = e_cands[:limit]

        # --- SMART PAIR GENERATION ---
        pairs = []
        hist_len = item['len']
        
        for s in s_cands:
            for e in e_cands:
                # Prevent self-loops
                if s['edge'] == e['edge'] and s['dist_proj'] > e['dist_proj']: continue
                
                # EUCLIDEAN PRE-CHECK
                air_dist = s['proj_pt'].distance(e['proj_pt'])
                
                if hist_len > 0:
                    ratio = air_dist / hist_len
                    if ratio < self.config.relaxation_length_min or ratio > self.config.relaxation_length_max:
                        continue
                
                rank_score = abs(air_dist - hist_len)
                pairs.append({'s': s, 'e': e, 'rank': rank_score})
        
        pairs.sort(key=lambda x: x['rank'])
        
        max_attempts = self.config.relaxation_max_paths if is_relax else self.config.max_paths_to_solve
        best_res = None
        best_score = float('inf')
        attempts = 0
        
        for pair in pairs:
            if attempts >= max_attempts: break
            
            res = self._stitch_and_score(pair['s'], pair['e'], item)
            
            if res and res.score < best_score:
                best_score = res.score
                best_res = res
                if best_score < self.config.early_exit_score: return best_res
            
            if res: attempts += 1 
                
        if not best_res: return self._reject(item['edge'])
        return best_res

    def _hydrate_candidates(self, indices, point):
        cands = []
        snap_thresh = self.config.virtual_node_snap_threshold
        for idx in indices:
            geom = self.edge_index.geometries[idx]
            dist = geom.distance(point)
            proj_dist = geom.project(point)
            proj_pt = geom.interpolate(proj_dist)
            
            u, v, k = self.edge_index.edge_keys[idx]
            snap_node = None
            if point.distance(self.node_geoms[u]) < snap_thresh: snap_node = u
            elif point.distance(self.node_geoms[v]) < snap_thresh: snap_node = v
            
            cands.append({
                'edge': (u, v, k), 'geom': geom, 'dist': dist, 
                'proj_pt': proj_pt, 'dist_proj': proj_dist, 'snap_node': snap_node
            })
        return cands

    def _stitch_and_score(self, s_cand, e_cand, item):
        s_u, s_v, _ = s_cand['edge']
        e_u, e_v, _ = e_cand['edge']
        start_node = s_cand['snap_node'] if s_cand['snap_node'] else s_v
        end_node = e_cand['snap_node'] if e_cand['snap_node'] else e_u

        path_nodes = []
        match_geom = None

        if s_cand['edge'] == e_cand['edge']:
            full = s_cand['geom']
            match_geom = substring(full, s_cand['dist_proj'], e_cand['dist_proj'])
            path_nodes = [s_u, s_v]
        else:
            try:
                path_nodes = nx.astar_path(self.G_osm, start_node, end_node, weight='length')
            except nx.NetworkXNoPath:
                return None

            # Concatenate edge geometries, skipping the shared endpoint between
            # consecutive segments so the resulting LineString has no zero-length
            # segments (cleaner output, smaller geojson).
            def extend_no_dup(target, new_pts):
                new_pts = list(new_pts)
                if target and new_pts and tuple(target[-1]) == tuple(new_pts[0]):
                    new_pts = new_pts[1:]
                target.extend(new_pts)

            coords = []
            if not s_cand['snap_node']:
                tail = substring(s_cand['geom'], s_cand['dist_proj'], s_cand['geom'].length)
                extend_no_dup(coords, tail.coords)

            if len(path_nodes) > 1:
                for i in range(len(path_nodes) - 1):
                    d = self.G_osm.get_edge_data(path_nodes[i], path_nodes[i+1])
                    if d:
                        extend_no_dup(coords, list(d.values())[0]['geometry'].coords)

            if not e_cand['snap_node']:
                head = substring(e_cand['geom'], 0, e_cand['dist_proj'])
                extend_no_dup(coords, head.coords)

            if len(coords) < 2: return None
            match_geom = LineString(coords)
            # Only prepend/append source-edge endpoints that aren't already at
            # the corresponding end of `path_nodes` (which already starts at
            # `start_node` and ends at `end_node` per astar's contract).
            prefix = [s_u] if start_node != s_u else []
            suffix = [e_v] if end_node != e_v else []
            path_nodes = prefix + path_nodes + suffix

        score, metrics = self.scorer.score_geometry(match_geom, item['coords'], item['len'])
        if score == float('inf'): return None
        
        c = item['geom'].centroid
        disp_vec = (match_geom.centroid.x - c.x, match_geom.centroid.y - c.y)
        return MatchResult(item['edge'], path_nodes, score, MatchConfidence.REJECTED, metrics, disp_vec, {}, match_geometry=match_geom)

    def _probe_node_fast(self, item) -> MatchResult:
        start_coord = np.array(item['coords'][0])
        candidates = self.node_index.find_nearest_nodes(
            start_coord,
            radius=self.config.registration_probe_radius,
            max_results=1,
        )
        if not candidates: return self._reject(item['edge'])
        res = MatchResult(item['edge'], [candidates[0][0]], 0.1, MatchConfidence.LOW)
        res.metadata = {'centroid': item['geom'].centroid.coords[0]} 
        res.match_geometry = Point(self.osm_node_coords[candidates[0][0]])
        return res

    def _apply_warp(self, results):
        src, dst = [], []
        for r in results:
            if r.confidence != MatchConfidence.REJECTED:
                src.append(r.metadata['centroid'])
                dst.append((r.match_geometry.x, r.match_geometry.y))
        
        if len(src) >= self.config.registration_min_matches:
            model, inliers = ransac((np.array(src), np.array(dst)), AffineTransform, min_samples=3,
                                    residual_threshold=self.config.ransac_residual_threshold)
            print(f"  -> RANSAC Inliers: {np.sum(inliers)}")
            self._warp_graph(model)

    def _warp_graph(self, model):
        for n, d in self.G_hist.nodes(data=True):
            if 'x' in d:
                res = model(np.array([[d['x'], d['y']]]))[0]
                d['x'], d['y'] = res[0], res[1]
                d['geometry'] = Point(res[0], res[1])
        for _,_,d in self.G_hist.edges(data=True):
            if 'geometry' in d:
                coords = np.array(d['geometry'].coords)
                new_coords = model(coords)
                d['geometry'] = LineString(new_coords)

    def _prepare_edge_data(self):
        data = []
        for u, v, k, d in self.G_hist.edges(keys=True, data=True):
            if 'geometry' in d:
                geom = d['geometry']
                coords = list(geom.coords)
                data.append({
                    'edge': (u,v,k), 'geom': geom, 'coords': coords, 
                    'bearing': calculate_bearing(coords[0], coords[-1]), 'len': geom.length,
                    'start_pt': Point(coords[0]), 'end_pt': Point(coords[-1])
                })
        return data

    def _apply_flow(self, results):
        history = deque(maxlen=15)
        final = []
        for res in results:
            if not res.match_geometry or res.score == float('inf'):
                res.confidence = MatchConfidence.REJECTED
                final.append(res)
                continue
            
            flow_score = self.scorer.calculate_flow_score(res.displacement_vector, history)
            res.score += flow_score * self.config.displacement_weight
            
            if res.score < self.config.high_confidence_threshold: res.confidence = MatchConfidence.HIGH
            elif res.score < self.config.medium_confidence_threshold: res.confidence = MatchConfidence.MEDIUM
            elif res.score < self.config.low_confidence_threshold: res.confidence = MatchConfidence.LOW
            elif res.score < self.config.very_low_confidence_threshold: res.confidence = MatchConfidence.VERY_LOW
            else: res.confidence = MatchConfidence.REJECTED
            
            if res.confidence != MatchConfidence.REJECTED: history.append(res.displacement_vector)
            final.append(res)
        return final

    def _check_highway(self, val, types):
        return any(x in types for x in val) if isinstance(val, list) else val in types

    def _reject(self, edge):
        return MatchResult(edge, [], float('inf'), MatchConfidence.REJECTED)

    def export_results(self, out_dir, results):
        os.makedirs(out_dir, exist_ok=True)
        
        # Lists for bucketing
        high = []
        medium = []
        low = []
        all_matches = []
        high_medium = [] # For Evaluation
        rejected_sources = []
        summary = {confidence.value: 0 for confidence in MatchConfidence}
        summary["total"] = len(results)
        summary["with_match_geometry"] = 0

        def geometry_to_json(geom):
            return json.loads(gpd.GeoSeries([geom], crs=self.crs_meters).to_json())['features'][0]['geometry']

        def json_safe(value):
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.floating,)):
                return None if not np.isfinite(value) else float(value)
            if isinstance(value, float):
                return None if not math.isfinite(value) else value
            return value

        def json_safe_props(props):
            return {key: json_safe(value) for key, value in props.items()}

        for r in results:
            summary[r.confidence.value] = summary.get(r.confidence.value, 0) + 1
            if r.match_geometry:
                summary["with_match_geometry"] += 1
            if r.confidence == MatchConfidence.REJECTED or not r.match_geometry:
                edge_data = self.G_hist.get_edge_data(*r.source_edge)
                if edge_data and edge_data.get('geometry') is not None:
                    rejected_sources.append({
                        "type": "Feature",
                        "geometry": geometry_to_json(edge_data['geometry']),
                        "properties": json_safe_props({
                            "source_id": str(r.source_edge),
                            "confidence": r.confidence.value,
                            "score": None if r.score == float('inf') else r.score,
                        }),
                    })
                continue
            
            # Create Feature
            props = {'score': r.score, 'confidence': r.confidence.value, 'source_id': str(r.source_edge), **r.metrics}
            feature = {"type": "Feature", "geometry": geometry_to_json(r.match_geometry), "properties": json_safe_props(props)}
            
            all_matches.append(feature)
            
            if r.confidence == MatchConfidence.HIGH:
                high.append(feature)
                high_medium.append(feature)
            elif r.confidence == MatchConfidence.MEDIUM:
                medium.append(feature)
                high_medium.append(feature)
            elif r.confidence in [MatchConfidence.LOW, MatchConfidence.VERY_LOW]:
                low.append(feature)

        # Helper to save, including empty FeatureCollections so the stage is visible.
        def save(features, name):
            payload = {
                "type": "FeatureCollection",
                "name": os.path.splitext(name)[0],
                "crs": {"type": "name", "properties": {"name": self.crs_meters}},
                "features": features,
            }
            save_json(os.path.join(out_dir, name), payload)

        save(high, "high_confidence.geojson")
        save(medium, "medium_confidence.geojson")
        save(low, "low_confidence.geojson")
        save(high_medium, "combined_high_medium.geojson")
        save(all_matches, "combined_all.geojson")
        save(rejected_sources, "rejected_sources.geojson")

        summary["accepted"] = len(all_matches)
        summary["rejected_source_features"] = len(rejected_sources)
        save_json(os.path.join(out_dir, "match_summary.json"), summary, indent=2)

        print(
            "Exported similarity results: "
            f"{summary['accepted']} accepted, {summary[MatchConfidence.REJECTED.value]} rejected "
            f"-> {out_dir}"
        )
