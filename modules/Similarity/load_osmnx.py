"""GeoJSON → NetworkX loader plus an osmnx-based simplification/consolidation pass."""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import networkx as nx
import osmnx as ox
import geopandas as gpd
from scipy.spatial import cKDTree
from tqdm import tqdm
from shapely.geometry import Point, LineString, MultiLineString, shape

from utils import load_json

logger = logging.getLogger(__name__)

OSMNX_RESERVED_NODE_ATTRS = {"osmid"}


# ── PART 1: Raw GeoJSON Loader ──

def _coord_to_node_id(lon: float, lat: float, prefix: str = "coord") -> str:
    return f"{prefix}_{lon:.7f}_{lat:.7f}"

def _sanitize_attrs(d: dict, drop: set = None) -> dict:
    drop = drop or set()
    out = {}
    for k, v in d.items():
        if k in drop: continue
        if isinstance(v, (list, dict)):
            out[k] = str(v) # Convert lists to strings for OSMnx compatibility
        else:
            out[k] = v
    return out

def _drop_reserved_node_attrs(G: nx.MultiDiGraph) -> None:
    """Remove node attrs that OSMnx also uses as GeoDataFrame index columns."""
    for _, attrs in G.nodes(data=True):
        for attr in OSMNX_RESERVED_NODE_ATTRS:
            attrs.pop(attr, None)

def _is_osmid_collision(exc: Exception) -> bool:
    return "cannot insert osmid, already exists" in str(exc)

def build_graph_from_geojson(geojson_data: Dict, graph_name: str = "graph") -> nx.MultiDiGraph:
    """Robustly loads GeoJSON into NetworkX with Shapely geometries."""
    G = nx.MultiDiGraph(name=graph_name)
    G.graph['crs'] = "EPSG:4326"

    features = geojson_data.get("features", [])
    if not features: return G

    # Partition features
    nodes_raw = [f for f in features if f.get("geometry", {}).get("type") == "Point"]
    edges_raw = [f for f in features if f.get("geometry", {}).get("type") in ("LineString", "MultiLineString")]

    # Process Nodes
    for f in tqdm(nodes_raw, desc=f"Loading {graph_name} nodes", leave=False):
        props = f.get("properties", {})
        coords = f.get("geometry", {}).get("coordinates")
        if not coords: continue
        
        lon, lat = float(coords[0]), float(coords[1])
        node_id = props.get("osmid") or props.get("id") or props.get("node_id") or f.get("id")
        if node_id is None: node_id = _coord_to_node_id(lon, lat, f"node_{graph_name}")
        else: node_id = str(node_id)
        
        attrs = _sanitize_attrs(props, drop={"x", "y", *OSMNX_RESERVED_NODE_ATTRS})
        G.add_node(node_id, x=lon, y=lat, geometry=Point(lon, lat), **attrs)

    # Process Edges
    for f in tqdm(edges_raw, desc=f"Loading {graph_name} edges", leave=False):
        props = f.get("properties", {})
        geom = shape(f.get("geometry"))
        
        # Handle MultiLineString: split into segments
        if isinstance(geom, MultiLineString):
            segments = list(geom.geoms)
        else:
            segments = [geom]
            
        attrs = _sanitize_attrs(props, drop={"u", "v", "key", "geometry"})
        
        for sub_geom in segments:
            if sub_geom.is_empty: continue
            
            # Endpoints
            start, end = sub_geom.coords[0], sub_geom.coords[-1]
            slon, slat = start[0], start[1]
            elon, elat = end[0], end[1]
            
            # IDs
            u = str(props.get("u") or props.get("source") or props.get("start") or _coord_to_node_id(slon, slat))
            v = str(props.get("v") or props.get("target") or props.get("end") or _coord_to_node_id(elon, elat))
            key = props.get("key", 0)

            # Ensure endpoints exist
            if u not in G: G.add_node(u, x=slon, y=slat, geometry=Point(slon, slat))
            if v not in G: G.add_node(v, x=elon, y=elat, geometry=Point(elon, elat))
            
            G.add_edge(u, v, key=key, geometry=sub_geom, **attrs)

    return G

def load_graph_file(path: str, kind: str) -> nx.MultiDiGraph:
    """Helper to load file from disk."""
    return build_graph_from_geojson(load_json(path), kind)


# ── PART 2: Graph Maker (Preprocessing Logic) ──

class Graph_Maker:
    def __init__(self, G_his: nx.MultiDiGraph, G_osm: nx.MultiDiGraph):
        print("Initializing Graph_Maker preprocessing...")
        self.G_his = G_his
        self.G_osm = G_osm
        self._prepare_for_osmnx(self.G_his)
        self._prepare_for_osmnx(self.G_osm)
        
        # Ensure CRS
        if 'crs' not in self.G_osm.graph: self.G_osm.graph['crs'] = 'EPSG:4326'
        if 'crs' not in self.G_his.graph: self.G_his.graph['crs'] = 'EPSG:4326'
        
        # 1. Fix Geometries & Attributes
        self._fix_edge_geometries(self.G_his)
        self._clean_edge_attributes(self.G_his)
        
        # 2. Simplify Historic
        print("Simplifying historic graph...")
        self._prepare_for_osmnx(self.G_his)
        self.G_his = ox.simplification.simplify_graph(self.G_his)
        self._prepare_for_osmnx(self.G_his)

        # 3. Consolidate OSM (Project -> Consolidate -> Project back)
        print("Consolidating OSM intersections...")
        try:
            # Project to UTM/Metric automatically
            self._prepare_for_osmnx(self.G_osm)
            osm_projected = ox.project_graph(self.G_osm)
            utm_crs = osm_projected.graph.get('crs', 'EPSG:32756')
            self._prepare_for_osmnx(osm_projected)
            osm_projected = ox.consolidate_intersections(osm_projected, tolerance=25, rebuild_graph=True, dead_ends=True)
            self._prepare_for_osmnx(osm_projected)
            # Project back to WGS84
            self.G_osm = ox.project_graph(osm_projected, to_crs='EPSG:4326')
            self._prepare_for_osmnx(self.G_osm)
            # ox.project_graph updates node x,y but does NOT reproject edge geometry attrs.
            # consolidate_intersections creates edge geometries in UTM, so we fix them here.
            self._reproject_edge_geoms(self.G_osm, utm_crs, 'EPSG:4326')
        except ValueError as exc:
            if not _is_osmid_collision(exc):
                raise
            print("  -> Skipping OSM intersection consolidation due to OSMnx osmid column collision.")
            self._prepare_for_osmnx(self.G_osm)

        # 4. Build OSM node table (used as the anchoring target)
        self._prepare_for_osmnx(self.G_osm)
        self.osm_nodes, _ = ox.graph_to_gdfs(self.G_osm)
        self._ensure_xy_cols(self.osm_nodes)

        # 5. Anchoring
        print("Anchoring historic nodes to OSM...")
        self._anchor_points(50.0)  # 50 m threshold (now genuinely meters)

    def _reproject_edge_geoms(self, G: nx.MultiDiGraph, from_crs: str, to_crs: str) -> None:
        """Reproject edge geometry attributes in-place using pyproj."""
        from pyproj import Transformer
        transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
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

    def _prepare_for_osmnx(self, G: nx.MultiDiGraph) -> None:
        _drop_reserved_node_attrs(G)
        
    def _ensure_xy_cols(self, gdf):
        """Ensure x and y columns exist in GeoDataFrame."""
        if 'x' not in gdf.columns or 'y' not in gdf.columns:
            if 'geometry' in gdf.columns:
                gdf['x'] = gdf.geometry.x
                gdf['y'] = gdf.geometry.y
            elif 'lon' in gdf.columns and 'lat' in gdf.columns:
                gdf['x'] = gdf['lon']
                gdf['y'] = gdf['lat']

    def _fix_edge_geometries(self, G: nx.MultiDiGraph):
        """Ensure all edges have valid Shapely LineStrings."""
        for u, v, k, d in G.edges(keys=True, data=True):
            geom = d.get('geometry')
            if isinstance(geom, (list, tuple)):
                try:
                    if len(geom) >= 2:
                        d['geometry'] = LineString(geom)
                    else:
                        del d['geometry']
                except Exception:
                    if 'geometry' in d: del d['geometry']

    def _clean_edge_attributes(self, G: nx.MultiDiGraph):
        """Convert lists to strings for OSMnx simplification compatibility."""
        for u, v, k, d in G.edges(keys=True, data=True):
            to_del = []
            for attr, val in d.items():
                if attr == 'geometry': continue
                if isinstance(val, (list, dict)):
                    try:
                        d[attr] = str(val)
                    except Exception:
                        to_del.append(attr)
            for attr in to_del:
                del d[attr]

    def _anchor_points(self, threshold_meters: float):
        """Anchor historic nodes to nearest OSM nodes within `threshold_meters`.

        Projects both nodesets to an auto-picked UTM zone so the threshold is
        in real metres. The previous implementation built the KDTree on raw
        (lon, lat) and converted the threshold via a global 111320 m/deg
        constant — accurate north-south anywhere but biased east-west by
        cos(lat), giving wrong anchor radii outside the equator.
        """
        from pyproj import Transformer

        self._prepare_for_osmnx(self.G_his)
        self.his_nodes, _ = ox.graph_to_gdfs(self.G_his)
        self._ensure_xy_cols(self.his_nodes)

        nx.set_node_attributes(self.G_his, False, 'anchor')
        nx.set_node_attributes(self.G_his, None, 'anchor_osm_id')
        nx.set_node_attributes(self.G_osm, False, 'anchor')
        nx.set_node_attributes(self.G_osm, None, 'anchor_his_id')

        if len(self.his_nodes) == 0 or len(self.osm_nodes) == 0:
            return

        # Auto-pick UTM zone from the historic graph's centroid.
        mean_lon = float(self.his_nodes['x'].mean())
        mean_lat = float(self.his_nodes['y'].mean())
        zone = int((mean_lon + 180) / 6) + 1
        utm_epsg = f"EPSG:{(32700 if mean_lat < 0 else 32600) + zone}"
        transformer = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)

        osm_x, osm_y = transformer.transform(
            self.osm_nodes['x'].values, self.osm_nodes['y'].values
        )
        his_x, his_y = transformer.transform(
            self.his_nodes['x'].values, self.his_nodes['y'].values
        )
        osm_metric = np.column_stack((osm_x, osm_y))
        his_metric = np.column_stack((his_x, his_y))

        metric_tree = cKDTree(osm_metric)
        dists, idxs = metric_tree.query(his_metric, distance_upper_bound=threshold_meters)

        for i, (dist, osm_idx) in enumerate(zip(dists, idxs)):
            if dist <= threshold_meters and osm_idx < len(self.osm_nodes):
                his_id = self.his_nodes.index[i]
                osm_id = self.osm_nodes.index[osm_idx]

                self.G_his.nodes[his_id]['anchor'] = True
                self.G_his.nodes[his_id]['anchor_osm_id'] = osm_id

                if osm_id in self.G_osm.nodes:
                    self.G_osm.nodes[osm_id]['anchor'] = True
                    self.G_osm.nodes[osm_id]['anchor_his_id'] = his_id

    def get_graphs(self):
        return self.G_his, self.G_osm


# ── PART 3: Public API (Matches HistoricMapMatcher calls) ──

def load_osmnx(historic_path: str, osm_path: str) -> Tuple[nx.MultiDiGraph, nx.MultiDiGraph]:
    """
    Main entry point used by HistoricMapMatcher.
    Loads files -> Creates GraphMaker -> Returns processed graphs.
    """
    # 1. Load Historic (GeoJSON)
    print(f"Loading Historic: {historic_path}")
    G_his = load_graph_file(historic_path, 'historic')

    # 2. Load OSM (GraphML or GeoJSON)
    print(f"Loading OSM: {osm_path}")
    if osm_path.endswith('.graphml'):
        try:
            G_osm = ox.load_graphml(osm_path)
        except Exception as e:
            print(f"  -> ox.load_graphml failed ({e.__class__.__name__}: {e}); falling back to JSON loader")
            G_osm = load_graph_file(osm_path, 'osm')
    else:
        G_osm = load_graph_file(osm_path, 'osm')

    _drop_reserved_node_attrs(G_his)
    _drop_reserved_node_attrs(G_osm)

    # 3. Preprocess
    maker = Graph_Maker(G_his, G_osm)
    return maker.get_graphs()
