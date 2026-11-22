import math
import os
import warnings
from dataclasses import dataclass, asdict
from typing import Tuple, List, Optional, Any

import cv2
import numpy as np
import networkx as nx
import PIL.Image as Image
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from utils import save_json

# Assuming these exist in your project structure
try:
    import modules.Road_Graphing.Util as Util
    import modules.Road_Graphing.skeleton as skeleton
except ModuleNotFoundError:
    import Util as Util
    import skeleton as skeleton

warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None

# Constants
MIN_ROAD_WIDTH, MAX_ROAD_WIDTH = 5, 50

@dataclass
class PipelineConfig:
    # Kept only the parameter actively used in the pipeline
    cleaning_passes: int = 2

@dataclass
class GraphStats:
    tile_name: str
    road_width_px: int
    nodes: int
    edges: int
    intersections: int
    endpoints: int
    components: int
    avg_degree: float
    total_edge_length_px: float
    largest_component_nodes: int

# ---------------------- Visualization Class -------------------------

class PipelineVisualizer:
    @staticmethod
    def find_crop_from_graph(graph: nx.Graph, image_shape: tuple, crop_size: int = 256) -> Tuple[int, int]:
        """Find a good crop centred on the densest part of the cleaned graph.
        Using the graph (not skeleton density) avoids locking onto border artefacts."""
        h, w = image_shape[:2]
        default_y = max(0, h // 2 - crop_size // 2)
        default_x = max(0, w // 2 - crop_size // 2)
        if not graph.nodes:
            return default_y, default_x

        # Centre on the largest connected component's median position
        largest_comp = max(nx.connected_components(graph), key=len)
        positions = np.array([graph.nodes[n]['pos'] for n in largest_comp])
        center_y, center_x = np.median(positions, axis=0)

        top_y = int(np.clip(center_y - crop_size // 2, 0, max(0, h - crop_size)))
        top_x = int(np.clip(center_x - crop_size // 2, 0, max(0, w - crop_size)))
        return top_y, top_x

    @staticmethod
    def draw_graph_on_patch(image: np.ndarray, graph: nx.Graph, 
                            top_y: int, top_x: int, size: int, 
                            node_color=(0, 255, 255), edge_color=(0, 255, 0)) -> np.ndarray:
        
        # --- FIX START: Handle pre-cropped vs full images ---
        h, w = image.shape[:2]
        if h == size and w == size:
            # Input image is already the patch; just copy it
            patch = image.copy()
        else:
            # Input image is the full map; crop it
            patch = image[top_y:top_y+size, top_x:top_x+size].copy()
        # --- FIX END ---

        # Ensure contiguous array for OpenCV (fixes layout error)
        patch = np.ascontiguousarray(patch)
        
        if len(patch.shape) == 2:
            patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
            
        for u, v, data in graph.edges(data=True):
            coords = data.get('coordinates', [])
            pts_local = []
            if coords:
                for y, x in coords:
                    if top_y <= y < top_y + size and top_x <= x < top_x + size:
                        pts_local.append((int(x - top_x), int(y - top_y)))
            
            if len(pts_local) > 1:
                # Explicitly cast to int32 for cv2.polylines
                pts_arr = np.array(pts_local, dtype=np.int32)
                cv2.polylines(patch, [pts_arr], False, edge_color, 1, cv2.LINE_AA)

        for n, data in graph.nodes(data=True):
            y, x = data['pos']
            if top_y <= y < top_y + size and top_x <= x < top_x + size:
                lx, ly = int(x - top_x), int(y - top_y)
                color = (0, 0, 255) if data.get('type') == 'intersection' else node_color
                radius = 3 if data.get('type') == 'intersection' else 2
                cv2.circle(patch, (lx, ly), radius + 1, (0,0,0), -1)
                cv2.circle(patch, (lx, ly), radius, color, -1)
        return patch

    @staticmethod
    def create_strip(npy_data: np.ndarray, binary: np.ndarray, skel: np.ndarray,
                    raw_graph: nx.Graph, final_graph: nx.Graph, save_path: str):
        CROP_SIZE = 256

        # Anchor all panels to the same crop, derived from the cleaned graph
        # so borders don't hijack the "busiest" region search.
        ty, tx = PipelineVisualizer.find_crop_from_graph(final_graph, skel.shape, CROP_SIZE)

        def crop(arr):
            return arr[ty:ty+CROP_SIZE, tx:tx+CROP_SIZE]

        # 1. Probability map
        raw_norm = cv2.normalize(npy_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_prob = cv2.cvtColor(crop(raw_norm), cv2.COLOR_GRAY2BGR)

        # 2. Binary mask
        img_bin = cv2.cvtColor(crop(binary), cv2.COLOR_GRAY2BGR)

        # 3. Skeleton
        skel_crop = crop(skel)
        img_skel = cv2.cvtColor(skel_crop, cv2.COLOR_GRAY2BGR)
        img_skel[img_skel[:, :, 0] > 0] = [255, 255, 255]

        # 4. Raw graph overlaid on dimmed skeleton crop
        img_raw_g = PipelineVisualizer.draw_graph_on_patch(
            (img_skel * 0.3).astype(np.uint8), raw_graph, ty, tx, CROP_SIZE
        )

        # 5. Final (optimised) graph overlaid on dimmed binary crop
        img_final_g = PipelineVisualizer.draw_graph_on_patch(
            (img_bin * 0.3).astype(np.uint8), final_graph, ty, tx, CROP_SIZE,
            edge_color=(50, 100, 255)
        )

        def add_label(img, text):
            res = img.copy()
            cv2.putText(res, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(res, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            return res

        full_strip = np.hstack([
            add_label(img_prob,    "1. Prob"),
            add_label(img_bin,     "2. Bin"),
            add_label(img_skel,    "3. Skel"),
            add_label(img_raw_g,   "4. Raw"),
            add_label(img_final_g, "5. Opt"),
        ])
        cv2.imwrite(save_path, full_strip)

# ---------------------- Processing Logic -------------------------

def compute_global_road_width(cleaned: np.ndarray, skel: np.ndarray) -> int:
    dist = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
    vals = dist[skel > 0]
    if vals.size == 0: return MIN_ROAD_WIDTH
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    keep = (vals >= max(1.0, q1 - 1.5*iqr)) & (vals <= q3 + 1.5*iqr)
    width_px = 2.0 * float(np.median(vals[keep] if np.any(keep) else vals))
    return int(np.clip(width_px, MIN_ROAD_WIDTH, MAX_ROAD_WIDTH))

_OFFSETS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _trace_skeleton_path(start_yx, first_path_yx, skel_u8, coord_to_id, h, w):
    """
    Walk along a degree-2 skeleton chain from `start_yx` (a node) through
    `first_path_yx` (a path pixel) until another node is reached.

    Returns (end_yx, coords) where coords is the full polyline including both
    endpoints, or (None, None) if the chain doesn't terminate at a node.
    """
    coords = [start_yx, first_path_yx]
    prev, curr = start_yx, first_path_yx
    # Safety bound: a chain can't be longer than the whole image.
    for _ in range(skel_u8.size):
        nxt = None
        for dy, dx in _OFFSETS_8:
            cand = (curr[0] + dy, curr[1] + dx)
            if cand == prev:
                continue
            if 0 <= cand[0] < h and 0 <= cand[1] < w and skel_u8[cand[0], cand[1]]:
                nxt = cand
                break
        if nxt is None:
            return None, None
        coords.append(nxt)
        if nxt in coord_to_id:
            return nxt, coords
        prev, curr = curr, nxt
    return None, None


def _polyline_length_px(coords):
    """Sum of Euclidean segment lengths along a (y, x) polyline."""
    total = 0.0
    for (y0, x0), (y1, x1) in zip(coords, coords[1:]):
        total += np.hypot(x1 - x0, y1 - y0)
    return total


def _annotate_edge_confidence(G: nx.Graph, prob_map: np.ndarray) -> None:
    """Sample `prob_map` along each edge polyline; store mean + min probability per edge.

    Lets downstream stages (geolocation, similarity) filter or weight edges by
    the underlying U-Net confidence — an edge built from prob 0.95 pixels is
    very different from one threading along prob 0.36 pixels even though
    both made it past hysteresis.
    """
    h, w = prob_map.shape[:2]
    for u, v, data in G.edges(data=True):
        coords = data.get('coordinates') or []
        if not coords:
            continue
        ys = np.clip(np.array([c[0] for c in coords]), 0, h - 1).astype(int)
        xs = np.clip(np.array([c[1] for c in coords]), 0, w - 1).astype(int)
        probs = prob_map[ys, xs]
        data['mean_prob'] = float(np.mean(probs))
        data['min_prob'] = float(np.min(probs))


def graph_from_skeleton(skel: np.ndarray,
                        prob_map: Optional[np.ndarray] = None) -> Tuple[nx.Graph, Optional[np.ndarray]]:
    skel_u8 = skel.astype(np.uint8)
    h, w = skel_u8.shape
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_u8, -1, kernel, borderType=cv2.BORDER_CONSTANT) * skel_u8

    y_idxs, x_idxs = np.where((neighbor_count == 1) | (neighbor_count > 2))
    G = nx.Graph()
    coord_to_id = {(y, x): i for i, (y, x) in enumerate(zip(y_idxs, x_idxs))}
    for (y, x), nid in coord_to_id.items():
        G.add_node(nid, pos=(y, x), type="intersection" if neighbor_count[y, x] > 2 else "endpoint")

    # Each degree-2 chain is traced exactly once; the second time we'd reach
    # it from the opposite end its first path pixel is already marked.
    traced_path_pixels = set()

    for y, x in zip(y_idxs, x_idxs):
        sid = coord_to_id[(y, x)]
        for dy, dx in _OFFSETS_8:
            ny, nx_l = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx_l < w and skel_u8[ny, nx_l]):
                continue
            if (ny, nx_l) in coord_to_id:
                eid = coord_to_id[(ny, nx_l)]
                if sid < eid:
                    G.add_edge(sid, eid, length_px=np.hypot(dx, dy),
                               coordinates=[(y, x), (ny, nx_l)])
            elif neighbor_count[ny, nx_l] == 2 and (ny, nx_l) not in traced_path_pixels:
                end_yx, coords = _trace_skeleton_path((y, x), (ny, nx_l), skel_u8, coord_to_id, h, w)
                if end_yx is None:
                    continue
                eid = coord_to_id[end_yx]
                # nx.Graph silently overwrites parallel edges; prefer the longer chain.
                length = _polyline_length_px(coords)
                if G.has_edge(sid, eid) and G[sid][eid].get('length_px', 0) >= length:
                    pass
                else:
                    G.add_edge(sid, eid, length_px=length, coordinates=coords)
                for px in coords[1:-1]:
                    traced_path_pixels.add(px)

    if prob_map is not None:
        _annotate_edge_confidence(G, prob_map)

    return G, None

class GraphCleaner:
    def __init__(self, g: nx.Graph, cfg: Any, road_width_px: int,
                 image_shape: Optional[Tuple[int, int]] = None, log=print,
                 endpoint_reconnect_radius_factor: float = 2.0,
                 endpoint_reconnect_max_bearing_deg: float = 30.0):
        self.graph = g.copy()
        self.cfg = cfg
        self.road_width_px = road_width_px
        self.image_shape = image_shape  # (height, width)
        self.log = log
        self.endpoint_reconnect_radius_factor = endpoint_reconnect_radius_factor
        self.endpoint_reconnect_max_bearing_deg = endpoint_reconnect_max_bearing_deg

    def clean(self) -> nx.Graph:
        """Main execution pipeline for graph cleaning."""
        initial_nodes = self.graph.number_of_nodes()

        self._simplify_topology()
        self._merge_close_intersections_kdtree(threshold=self.road_width_px)

        # Heal broken roads: gap-broken parts of the same line that left two
        # endpoints sitting close and roughly collinear. Run before the
        # iterative pruning so reconnected edges survive into later passes.
        if self.endpoint_reconnect_radius_factor > 0:
            reconnect_radius = self.road_width_px * self.endpoint_reconnect_radius_factor
            self._reconnect_close_endpoints(
                max_distance=reconnect_radius,
                max_bearing_deg=self.endpoint_reconnect_max_bearing_deg,
            )

        for pass_num in range(self.cfg.cleaning_passes):
            self.log(f"[CLEAN] Pass {pass_num+1}/{self.cfg.cleaning_passes}")
            self._simplify_topology()
            prune_len = max(5.0, self.road_width_px * 1.5)
            self._prune_spurs_recursive(min_len=prune_len)
            min_component_len = self.road_width_px * 3
            self.log(f"[CLEAN]   min_component_len = {min_component_len}")
            self._remove_small_components_by_length(min_len=min_component_len)

        # 4. Remove border/frame edges (very long straight edges spanning map boundary)
        if self.image_shape is not None:
            self._remove_long_border_edges()

        # 5. Aggressive Filter — also cleans up nodes isolated by border removal
        self._remove_simple_components(max_nodes=10)

        # 6. Edge geometry simplification (Douglas-Peucker, per-point)
        # Replaces the prior all-or-nothing rectification: a near-straight edge
        # still collapses to a 2-point chord, but real bends are preserved
        # because RDP only drops points whose perpendicular distance to the
        # chord between their neighbours is < tolerance.
        simplify_tol = min(self.road_width_px / 4.0, 5.0)
        self._simplify_edge_geometries(tolerance=simplify_tol)

        # 7a. Savitzky-Golay denoise per-edge — kills the U-Net's high-frequency
        # stair-step noise before any corner-aware smoother gets to look at it.
        # Run BEFORE the adaptive smoother so its corner detection sees clean
        # angles rather than micro-jags that mimic corners.
        self._savgol_smooth_edges(window=9, polyorder=2)

        # 7b. Adaptive Smoothing (corner-anchored sin-window weighted average)
        self._smooth_edge_geometry()

        # 8. Chaikin corner-cutting — rounds residual polygonal steps without
        # rounding real corners (anchored at sharp bends).
        self._chaikin_smooth_edges(iterations=2)

        # 9. Confidence-based hair removal — drop low-confidence short components
        # that survived earlier passes. Uses per-edge mean_prob from the U-Net.
        self._drop_low_confidence_components(prob_threshold=0.45, max_length=80.0)

        self._refresh_types()
        self.log(f"[CLEAN] Finished. Nodes: {initial_nodes} -> {self.graph.number_of_nodes()}")
        return self.graph

    def _simplify_edge_geometries(self, tolerance: float):
        """Apply Douglas-Peucker simplification to each edge polyline.

        Drops only the redundant near-collinear points (perpendicular distance
        from the local chord < `tolerance`). Genuinely-straight edges still
        collapse to a 2-point chord; curved edges keep the points that carry
        their shape. Strictly preserves more geometric information than the
        old all-or-nothing rectification.
        """
        simplified_edges = 0
        pts_before = pts_after = 0
        for u, v, data in self.graph.edges(data=True):
            coords = data.get('coordinates', [])
            n_in = len(coords)
            if n_in < 3:
                pts_before += n_in; pts_after += n_in
                continue
            line = LineString(coords)
            simplified = line.simplify(tolerance, preserve_topology=False)
            new_coords = [tuple(c) for c in simplified.coords]
            if len(new_coords) < n_in:
                data['coordinates'] = new_coords
                simplified_edges += 1
            pts_before += n_in
            pts_after += len(new_coords) if len(new_coords) < n_in else n_in
        self.log(
            f"[CLEAN] Simplified geometry on {simplified_edges} edges "
            f"({pts_before} → {pts_after} points, tolerance={tolerance:.2f} px)"
        )

    def _smooth_edge_geometry(self):
        smoothed_count = 0
        CORNER_THRESH_DEG = 30.0

        for u, v, data in self.graph.edges(data=True):
            coords = data.get('coordinates', [])
            if len(coords) < 3: continue

            # Drop consecutive duplicate points before computing segment vectors.
            # Previously the smoother only normalised "valid" (non-zero) vectors
            # but then computed dot products over ALL vectors (vecs[:-1] vs vecs[1:]),
            # so zero-length segments produced meaningless angles and could fire
            # false corner anchors. Deduping at the source guarantees every
            # segment vector has non-zero magnitude.
            pts_list = [coords[0]]
            for p in coords[1:]:
                if p[0] != pts_list[-1][0] or p[1] != pts_list[-1][1]:
                    pts_list.append(p)
            if len(pts_list) < 3:
                continue
            pts = np.array(pts_list)

            vecs = pts[1:] - pts[:-1]
            norms = np.linalg.norm(vecs, axis=1)
            if (norms < 1e-9).any():
                # Sub-pixel near-duplicates that survived the integer-equality dedup.
                continue
            unit_vecs = vecs / norms[:, None]
            dots = np.clip(np.einsum('ij,ij->i', unit_vecs[:-1], unit_vecs[1:]), -1.0, 1.0)
            angles_deg = np.degrees(np.arccos(dots))

            corner_indices = np.where(angles_deg > CORNER_THRESH_DEG)[0] + 1
            anchors = sorted(set([0] + corner_indices.tolist() + [len(pts) - 1]))

            new_pts = pts.copy()
            for k in range(len(anchors) - 1):
                start, end = anchors[k], anchors[k+1]
                if end - start < 3: continue
                idx_range = np.arange(start + 1, end)
                t_vals = (idx_range - start) / (end - start)
                weights = np.sin(np.pi * t_vals)
                p_prev, p_curr, p_next = pts[idx_range - 1], pts[idx_range], pts[idx_range + 1]
                p_smooth = 0.25 * p_prev + 0.5 * p_curr + 0.25 * p_next
                w = weights[:, None]
                new_pts[idx_range] = (1 - w) * p_curr + w * p_smooth

            data['coordinates'] = [tuple(p) for p in new_pts]
            smoothed_count += 1
        self.log(f"[CLEAN] Adaptive smoothing applied to {smoothed_count} edges.")

    def _savgol_smooth_edges(self, window: int = 9, polyorder: int = 2):
        """1-D Savitzky-Golay filter on each edge's coordinate sequence.

        Treats x(t) and y(t) as independent 1-D signals indexed by point order
        and fits a `polyorder` polynomial in a sliding `window`. Removes the
        single-pixel stair-step noise that Lee skeletonisation produces on
        thick predictions, without shrinking endpoints (we keep the original
        first/last points and pin them post-filter).
        """
        try:
            from scipy.signal import savgol_filter
        except ImportError:
            self.log("[CLEAN] scipy not available — skipping Savitzky-Golay smoothing.")
            return
        if window % 2 == 0:
            window += 1
        if window < polyorder + 2:
            window = polyorder + 3 + (polyorder + 3) % 2 ^ 1  # next odd ≥ polyorder+3
        smoothed = 0
        for u, v, data in self.graph.edges(data=True):
            coords = data.get('coordinates', [])
            if len(coords) < window:
                continue
            pts = np.asarray(coords, dtype=np.float64)
            ys = savgol_filter(pts[:, 0], window, polyorder, mode='interp')
            xs = savgol_filter(pts[:, 1], window, polyorder, mode='interp')
            # Pin endpoints — junctions must stay where the graph says they are.
            ys[0], xs[0] = pts[0, 0], pts[0, 1]
            ys[-1], xs[-1] = pts[-1, 0], pts[-1, 1]
            data['coordinates'] = list(zip(ys.tolist(), xs.tolist()))
            smoothed += 1
        self.log(f"[CLEAN] Savitzky-Golay (w={window}, p={polyorder}) applied to {smoothed} edges.")

    def _drop_low_confidence_components(self, prob_threshold: float, max_length: float):
        """Drop connected components whose mean U-Net confidence < threshold AND total
        length < max_length. Catches hallucinated meanders in field/label areas that
        the spatial pruners miss because they are technically connected to each other.
        """
        # Average mean_prob over each component, weighted by length.
        components = list(nx.connected_components(self.graph))
        dropped_nodes = 0
        dropped_components = 0
        for comp in components:
            sub = self.graph.subgraph(comp)
            total_len = 0.0
            weighted = 0.0
            valid = False
            for u, v, data in sub.edges(data=True):
                mp = data.get('mean_prob')
                length = float(data.get('length_px', 0)) or 1.0
                if mp is None:
                    continue
                valid = True
                weighted += mp * length
                total_len += length
            if not valid or total_len == 0:
                continue
            avg_prob = weighted / total_len
            if avg_prob < prob_threshold and total_len < max_length:
                self.graph.remove_nodes_from(comp)
                dropped_nodes += len(comp)
                dropped_components += 1
        if dropped_components:
            self.log(f"[CLEAN] Dropped {dropped_components} low-confidence components "
                     f"({dropped_nodes} nodes, p<{prob_threshold}, len<{max_length}).")

    def _chaikin_smooth_edges(self, iterations: int = 2, corner_thresh_deg: float = 40.0):
        """Chaikin's corner-cutting between sharp-angle anchors.

        Each iteration replaces every interior segment AB with two new points
        at 1/4 and 3/4 of the way from A to B. Two iterations turn an N-point
        polyline into a near-C¹ curve while only growing length by a few
        percent. Endpoints and any vertex whose turn exceeds
        `corner_thresh_deg` are pinned so junctions and genuine bends survive.
        """
        if iterations <= 0:
            return
        cut_count = 0
        for u, v, data in self.graph.edges(data=True):
            coords = data.get('coordinates', [])
            if len(coords) < 3:
                continue

            # Dedupe consecutive duplicates (same defensive step as the smoother).
            pts_list = [coords[0]]
            for p in coords[1:]:
                if p[0] != pts_list[-1][0] or p[1] != pts_list[-1][1]:
                    pts_list.append(p)
            if len(pts_list) < 3:
                continue
            pts = np.array(pts_list, dtype=np.float64)

            vecs = pts[1:] - pts[:-1]
            norms = np.linalg.norm(vecs, axis=1)
            if (norms < 1e-9).any():
                continue
            unit_vecs = vecs / norms[:, None]
            dots = np.clip(np.einsum('ij,ij->i', unit_vecs[:-1], unit_vecs[1:]), -1.0, 1.0)
            angles_deg = np.degrees(np.arccos(dots))
            corner_idx = (np.where(angles_deg > corner_thresh_deg)[0] + 1).tolist()
            anchors = sorted(set([0] + corner_idx + [len(pts) - 1]))

            out = []
            for k in range(len(anchors) - 1):
                a, b = anchors[k], anchors[k + 1]
                seg = pts[a:b + 1]
                if len(seg) < 3:
                    out.extend(seg.tolist() if k == 0 else seg[1:].tolist())
                    continue
                # Chaikin keeps the first and last point of the segment fixed
                # (so anchors stay pinned) and cuts every interior corner.
                cur = seg
                for _ in range(iterations):
                    a0 = cur[:-1]
                    a1 = cur[1:]
                    q = 0.75 * a0 + 0.25 * a1
                    r = 0.25 * a0 + 0.75 * a1
                    new = np.empty((2 * len(a0) + 2, 2), dtype=cur.dtype)
                    new[0] = cur[0]
                    new[1:-1:2] = q
                    new[2:-1:2] = r
                    new[-1] = cur[-1]
                    cur = new
                cur_list = cur.tolist()
                if k == 0:
                    out.extend(cur_list)
                else:
                    out.extend(cur_list[1:])

            data['coordinates'] = [tuple(p) for p in out]
            cut_count += 1
        self.log(f"[CLEAN] Chaikin smoothing ({iterations}x) applied to {cut_count} edges.")

    def _endpoint_outgoing_unit_vec(self, node):
        """Unit vector pointing from `node` AWAY into its (only) edge. None if undefined."""
        nbrs = list(self.graph.neighbors(node))
        if len(nbrs) != 1:
            return None
        coords = self.graph[node][nbrs[0]].get('coordinates', [])
        pos_n = self.graph.nodes[node]['pos']
        if coords and len(coords) >= 2:
            def d2(p1, p2): return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
            ref = coords[1] if d2(coords[0], pos_n) <= d2(coords[-1], pos_n) else coords[-2]
        else:
            ref = self.graph.nodes[nbrs[0]]['pos']
        dy, dx = ref[0] - pos_n[0], ref[1] - pos_n[1]
        norm = (dy * dy + dx * dx) ** 0.5
        if norm < 1e-6:
            return None
        return (dy / norm, dx / norm)

    def _reconnect_close_endpoints(self, max_distance: float, max_bearing_deg: float):
        """Heal gap-broken roads by reconnecting endpoint pairs that are close + collinear.

        Counters the U-Net pattern where a single road comes out as two skeleton
        fragments separated by a small gap. After cleaning, both fragments end
        in degree-1 nodes that would otherwise get pruned as short dead-ends.

        Two endpoints A, B reconnect when:
          1. they sit within `max_distance` of each other, AND
          2. A's existing-edge direction is opposite to vec(A→B), AND
          3. B's existing-edge direction is parallel to vec(A→B).
        Both alignment checks must be within `max_bearing_deg`. Each endpoint
        consumes only one match (closest pair wins) to avoid star-pattern fusions.
        """
        endpoints = [n for n, d in self.graph.degree() if d == 1]
        if len(endpoints) < 2:
            return

        positions = np.array([self.graph.nodes[n]['pos'] for n in endpoints], dtype=float)
        tree = cKDTree(positions)
        pairs = tree.query_pairs(r=max_distance)
        if not pairs:
            return

        out_vec = {n: self._endpoint_outgoing_unit_vec(n) for n in endpoints}
        cos_thresh = math.cos(math.radians(max_bearing_deg))

        # Sort by distance so the closest valid pair wins when an endpoint can match multiple.
        pair_distances = [
            (float(np.linalg.norm(positions[i] - positions[j])), endpoints[i], endpoints[j])
            for i, j in pairs
        ]
        pair_distances.sort()

        consumed: set = set()
        reconnected = 0
        for d, a, b in pair_distances:
            if a in consumed or b in consumed:
                continue
            va, vb = out_vec.get(a), out_vec.get(b)
            if va is None or vb is None:
                continue

            pa = self.graph.nodes[a]['pos']
            pb = self.graph.nodes[b]['pos']
            ab_dy, ab_dx = pb[0] - pa[0], pb[1] - pa[1]
            ab_norm = (ab_dy * ab_dy + ab_dx * ab_dx) ** 0.5
            if ab_norm < 1e-6:
                continue
            ab_unit = (ab_dy / ab_norm, ab_dx / ab_norm)

            # A's edge should point AWAY from B (cos < -cos_thresh); B's toward B (cos > +cos_thresh)
            cos_a = va[0] * ab_unit[0] + va[1] * ab_unit[1]
            cos_b = vb[0] * ab_unit[0] + vb[1] * ab_unit[1]
            if -cos_a < cos_thresh or cos_b < cos_thresh:
                continue

            self.graph.add_edge(
                a, b, length_px=float(d), coordinates=[tuple(pa), tuple(pb)],
            )
            consumed.add(a); consumed.add(b)
            reconnected += 1

        if reconnected:
            self.log(f"[CLEAN] Reconnected {reconnected} endpoint pair(s) within {max_distance:.1f} px.")

    def _merge_close_intersections_kdtree(self, threshold: float):
        intersections = [n for n, d in self.graph.degree() if d > 2]
        if len(intersections) < 2: return
        coords = np.array([self.graph.nodes[n]['pos'] for n in intersections])
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=threshold)
        if not pairs: return

        cluster_graph = nx.Graph()
        cluster_graph.add_nodes_from(range(len(intersections)))
        cluster_graph.add_edges_from(pairs)
        clusters = list(nx.connected_components(cluster_graph))
        self.log(f"[CLEAN] Merging {len(clusters)} intersection clusters.")
        for cluster_indices in clusters:
            self._merge_node_cluster([intersections[i] for i in cluster_indices])

    def _merge_node_cluster(self, nodes: List[int]):
        if not nodes: return
        positions = np.array([self.graph.nodes[n]['pos'] for n in nodes])
        centroid = tuple(np.mean(positions, axis=0))
        survivor = nodes[0]
        self.graph.nodes[survivor]['pos'] = centroid
        
        for nbr in list(self.graph.neighbors(survivor)):
            if nbr in nodes: continue
            self._snap_edge_to_node(survivor, nbr, centroid)

        for victim in nodes[1:]:
            for nbr in list(self.graph.neighbors(victim)):
                if nbr in nodes: continue 
                if not self.graph.has_edge(survivor, nbr):
                    attr = self.graph[victim][nbr].copy()
                    self.graph.add_edge(survivor, nbr, **attr)
                    self._snap_edge_to_node(survivor, nbr, centroid)
            self.graph.remove_node(victim)

    def _snap_edge_to_node(self, u, v, new_pos_u):
        if not self.graph.has_edge(u, v): return
        props = self.graph[u][v]
        coords = props.get('coordinates', [])
        pos_v = self.graph.nodes[v]['pos']
        
        if not coords:
            props['coordinates'] = [new_pos_u, pos_v]
            props['length_px'] = np.hypot(new_pos_u[0]-pos_v[0], new_pos_u[1]-pos_v[1])
            return

        def dist_sq(p1, p2): return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
        d_start = dist_sq(coords[0], pos_v)
        d_end   = dist_sq(coords[-1], pos_v)
        if d_start < d_end: coords[-1] = new_pos_u
        else: coords[0] = new_pos_u
        props['coordinates'] = coords

    def _simplify_topology(self):
        dirty = True
        while dirty:
            dirty = False
            candidates = [n for n in self.graph.nodes if self.graph.degree(n) == 2]
            for n in candidates:
                if not self.graph.has_node(n): continue
                neighbors = list(self.graph.neighbors(n))
                # Self-loop-only nodes have degree 2 (loops count twice in nx.Graph)
                # but only one neighbour — handled by len != 2 check above.
                # Two edges to the same neighbour is impossible in nx.Graph (it
                # collapses parallels), so `len == 2` always implies u != v here.
                if len(neighbors) != 2: continue
                u, v = neighbors

                attr_u = self.graph[u][n]
                attr_v = self.graph[n][v]
                new_len = attr_u.get('length_px', 0) + attr_v.get('length_px', 0)
                new_coords = self._merge_geometries(
                    attr_u.get('coordinates', []),
                    attr_v.get('coordinates', []),
                    self.graph.nodes[u]['pos'],
                    self.graph.nodes[n]['pos'],
                    self.graph.nodes[v]['pos']
                )
                
                # nx.Graph collapses parallel edges silently. When u-v already
                # exists (triangle topology around n), keep whichever side has
                # the longer geometry rather than discarding the contraction.
                if self.graph.has_edge(u, v):
                    if self.graph[u][v].get('length_px', 0) < new_len:
                        self.graph[u][v]['length_px'] = new_len
                        self.graph[u][v]['coordinates'] = new_coords
                else:
                    self.graph.add_edge(u, v, length_px=new_len, coordinates=new_coords)
                self.graph.remove_node(n)
                dirty = True

    def _merge_geometries(self, c1, c2, pos_u, pos_n, pos_v):
        if not c1: c1 = [pos_u, pos_n]
        if not c2: c2 = [pos_n, pos_v]
        def d2(p1, p2): return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
        if d2(c1[-1], pos_n) > d2(c1[0], pos_n): c1 = c1[::-1]
        if d2(c2[0], pos_n) > d2(c2[-1], pos_n): c2 = c2[::-1]
        return c1 + c2[1:]

    def _prune_spurs_recursive(self, min_len: float):
        while True:
            pruned_in_pass = 0
            endpoints = [n for n in self.graph.nodes if self.graph.degree(n) == 1]
            for n in endpoints:
                if not self.graph.has_node(n): continue
                neighbors = list(self.graph.neighbors(n))
                if not neighbors: 
                    self.graph.remove_node(n)
                    continue
                neighbor = neighbors[0]
                edge_data = self.graph[n][neighbor]
                if (edge_data.get('length_px', float('inf')) < min_len and 
                    self.graph.degree(neighbor) > 1):
                    self.graph.remove_node(n)
                    pruned_in_pass += 1
            if pruned_in_pass == 0: break

    def _remove_small_components_by_length(self, min_len: float):
        """Remove connected components whose total edge length (in pixels) is below `min_len`.

        Previously this argument was ignored — threshold was computed as
        `mean - 0.5 * std`, which is a Gaussian outlier filter and behaves
        unpredictably on heavy-tailed road-component distributions (one big
        network plus many small ones). Now uses the caller's absolute floor;
        stats are still logged so the cutoff can be sanity-checked.
        """
        components = list(nx.connected_components(self.graph))
        if not components:
            self.log("[CLEAN]   no components to evaluate, skipping length filter")
            return

        component_lengths = []
        to_remove: List[int] = []
        for comp_nodes in components:
            subgraph = self.graph.subgraph(comp_nodes)
            total_len = sum(d.get('length_px', 0) for _, _, d in subgraph.edges(data=True))
            component_lengths.append(total_len)
            if total_len < min_len:
                to_remove.extend(comp_nodes)

        kept = len(components) - sum(1 for l in component_lengths if l < min_len)
        self.log(
            f"[CLEAN]   component lengths: mean={np.mean(component_lengths):.1f} "
            f"max={max(component_lengths):.1f} std={np.std(component_lengths):.1f} "
            f"threshold={min_len:.1f} kept={kept}/{len(components)}"
        )

        if to_remove:
            self.graph.remove_nodes_from(to_remove)

        

    def _remove_long_border_edges(self, margin_fraction: float = 0.06, axis_std_fraction: float = 0.015):
        """Remove map border/frame edges.

        Two-condition filter — an edge is a border artefact if it is BOTH:
          1. Nearly axis-aligned: std of its coordinate points along one axis
             is less than `axis_std_fraction * image_dim` (i.e. it runs as a
             flat horizontal or vertical line), AND
          2. Located near the corresponding image boundary: its mean position
             along that axis is within `margin_fraction * image_dim` of the edge.

        This survives segmented borders (broken by tick marks) because each
        segment still passes both tests independently.  It will not affect
        diagonal or interior roads even if they briefly approach an edge.
        """
        if self.image_shape is None:
            return
        h, w = self.image_shape[:2]
        margin_y   = h * margin_fraction
        margin_x   = w * margin_fraction
        std_thr_y  = h * axis_std_fraction   # flatness threshold for horizontal edges
        std_thr_x  = w * axis_std_fraction   # flatness threshold for vertical edges

        edges_to_remove = []
        for u, v, data in self.graph.edges(data=True):
            coords = data.get('coordinates', [])
            if not coords:
                coords = [self.graph.nodes[u]['pos'], self.graph.nodes[v]['pos']]

            pts    = np.array(coords, dtype=float)
            ys, xs = pts[:, 0], pts[:, 1]

            mean_y, std_y = float(np.mean(ys)), float(np.std(ys))
            mean_x, std_x = float(np.mean(xs)), float(np.std(xs))

            # Nearly horizontal AND near top or bottom boundary
            if std_y < std_thr_y and (mean_y < margin_y or mean_y > h - margin_y):
                edges_to_remove.append((u, v))
            # Nearly vertical AND near left or right boundary
            elif std_x < std_thr_x and (mean_x < margin_x or mean_x > w - margin_x):
                edges_to_remove.append((u, v))

        self.graph.remove_edges_from(edges_to_remove)
        self.log(f"[CLEAN] Border removal: dropped {len(edges_to_remove)} axis-aligned boundary edges "
                 f"(margin={margin_fraction*100:.0f}%, axis_std={axis_std_fraction*100:.1f}%). "
                 f"Isolated nodes cleared by _remove_simple_components.")

    def _remove_simple_components(self, max_nodes: int = 5):
        to_remove = []
        for comp in nx.connected_components(self.graph):
            if len(comp) <= max_nodes:
                to_remove.extend(list(comp))
        if to_remove:
            self.graph.remove_nodes_from(to_remove)

    def _refresh_types(self):
        for n in self.graph.nodes:
            d = self.graph.degree(n)
            if d == 1: n_type = "endpoint"
            elif d == 2: n_type = "path"
            else: n_type = "intersection"
            self.graph.nodes[n]["type"] = n_type

# -------------------------- Orchestration --------------------------

def _load_base_image_as_luminance(input_img_path: str, target_shape):
    """Load the source map image, convert to greyscale (0–255), and resize to
    match the prediction grid. Returns ``(luminance_uint8, ok_flag)``.

    Resizing keeps the edge coordinate space (already in prediction-grid pixels)
    valid for direct lookup against this base image — no scale conversion needed
    at the call sites.
    """
    try:
        from PIL import Image as _PIL
        _PIL.MAX_IMAGE_PIXELS = None
        img = _PIL.open(input_img_path).convert("L")
        h, w = int(target_shape[0]), int(target_shape[1])
        if (img.size[0], img.size[1]) != (w, h):
            img = img.resize((w, h), _PIL.LANCZOS)
        return np.asarray(img, dtype=np.uint8), True
    except Exception:
        return None, False


def image_evidence_prune(G: nx.Graph,
                          base_image: np.ndarray,
                          *,
                          road_width_px: float,
                          ink_lum_threshold: int = 160,
                          min_dark_fraction: float = 0.30,
                          casing_offset_factor: float = 0.60,
                          casing_extra_search_px: int = 2,
                          log=print) -> nx.Graph:
    """Drop edges whose ink-evidence in the base image is too weak.

    Engraved-style historic maps render roads as **two parallel casing
    strokes with paper between** — the centreline itself is paper-coloured.
    So the test that proves "this is a real road" is: at each point along
    the centreline, walk perpendicular ``road_width/2 × casing_offset_factor``
    pixels in BOTH directions and look for dark ink within a small search
    band of ``±casing_extra_search_px`` (allows for misalignment of the
    U-Net centreline vs. true engraved centreline). If the fraction of
    sample points where we found ink on EITHER side is below
    ``min_dark_fraction``, drop the edge.

    Catches hallucinated meanders in fields, building outlines mis-tagged
    as roads, and label-baseline strokes the model picked up — those don't
    have the casing-pair signature.
    """
    if base_image is None:
        log("[EVID] no base image — skipping")
        return G
    if base_image.ndim == 3:
        base_image = (
            0.299 * base_image[..., 0] +
            0.587 * base_image[..., 1] +
            0.114 * base_image[..., 2]
        ).astype(np.uint8)
    h, w = base_image.shape[:2]
    thr = int(ink_lum_threshold)
    half_w = max(1.0, float(road_width_px) * 0.5 * float(casing_offset_factor))
    search = int(max(1, casing_extra_search_px))

    def _has_ink_in_band(cy: float, cx: float, py: float, px: float, sign: float) -> bool:
        # Walk a small range of offsets near the target casing distance to
        # tolerate misalignment between predicted centreline and true casing.
        for s in range(-search, search + 1):
            d = half_w + s
            iy = int(round(cy + py * sign * d))
            ix = int(round(cx + px * sign * d))
            if 0 <= iy < h and 0 <= ix < w and base_image[iy, ix] < thr:
                return True
        return False

    edges_to_drop = []
    n_total = 0
    for u, v, data in G.edges(data=True):
        n_total += 1
        coords = data.get('coordinates', [])
        if len(coords) < 2:
            continue
        pts = np.asarray(coords, dtype=np.float64)
        diffs = np.diff(pts, axis=0)
        norms = np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-9
        unit = diffs / norms
        unit = np.vstack([unit, unit[-1:]])
        perp_y, perp_x = -unit[:, 1], unit[:, 0]

        n_samples = 0
        n_with_casing = 0
        for i, (y, x) in enumerate(pts):
            left_ink  = _has_ink_in_band(y, x, perp_y[i], perp_x[i], +1.0)
            right_ink = _has_ink_in_band(y, x, perp_y[i], perp_x[i], -1.0)
            n_samples += 1
            # Either side counts as casing evidence (one-sided streets,
            # courtyards, end-of-row segments — all real and one-sided).
            if left_ink or right_ink:
                n_with_casing += 1
        if n_samples == 0:
            edges_to_drop.append((u, v)); continue
        casing_frac = n_with_casing / n_samples
        data['evidence_casing_frac'] = float(casing_frac)
        if casing_frac < float(min_dark_fraction):
            edges_to_drop.append((u, v))
    for u, v in edges_to_drop:
        if G.has_edge(u, v):
            G.remove_edge(u, v)
    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    G.remove_nodes_from(isolated)
    log(f"[EVID] casing-evidence pruned: {len(edges_to_drop)}/{n_total} dropped "
        f"(ink_thr={thr}, min_dark_frac={min_dark_fraction}, half_w={half_w:.1f}px)")
    return G


def snake_snap_edges(G: nx.Graph,
                      base_image: np.ndarray,
                      *,
                      road_width_px: float,
                      n_iterations: int = 8,
                      search_radius_px: Optional[int] = None,
                      image_weight: float = 0.5,
                      smoothness_weight: float = 0.5,
                      log=print) -> nx.Graph:
    """Active-contour ("snake") snapping of edge interiors to ink in the base image.

    Each edge polyline is treated as a 1-D chain whose interior points are
    iteratively pulled toward the locally darkest pixel within a small search
    window (the "external energy"), regularised by a smoothness term that
    blends each point toward the midpoint of its neighbours.

    Endpoints (graph nodes) are pinned — they're shared between edges and the
    cleanup should not move them.

    The blend is:
        new_pt = image_weight * argmin_dark + smoothness_weight * neighbour_mid

    Make ``image_weight + smoothness_weight ≤ 1`` for stable behaviour; the
    defaults (0.5, 0.5) work well on engraved historic maps.
    """
    if base_image is None:
        log("[SNAKE] no base image — skipping")
        return G
    if base_image.ndim == 3:
        base_image = (
            0.299 * base_image[..., 0] +
            0.587 * base_image[..., 1] +
            0.114 * base_image[..., 2]
        ).astype(np.uint8)
    base = base_image.astype(np.float32)
    h, w = base.shape[:2]
    if search_radius_px is None:
        search_radius_px = max(2, int(road_width_px) // 2)
    sr = int(search_radius_px)
    iw = float(image_weight)
    sw = float(smoothness_weight)

    snapped = 0
    for u, v, data in G.edges(data=True):
        coords = data.get('coordinates', [])
        if len(coords) < 3:
            continue
        pts = np.asarray(coords, dtype=np.float64)
        for _ in range(int(n_iterations)):
            new_pts = pts.copy()
            for i in range(1, len(pts) - 1):
                cy, cx = pts[i]
                y0 = max(0, int(round(cy)) - sr)
                y1 = min(h, int(round(cy)) + sr + 1)
                x0 = max(0, int(round(cx)) - sr)
                x1 = min(w, int(round(cx)) + sr + 1)
                if y0 >= y1 or x0 >= x1:
                    continue
                patch = base[y0:y1, x0:x1]
                # Darkest pixel = lowest luminance. Tie-break by closest to
                # current centre so the snake doesn't lurch around.
                flat_idx = int(np.argmin(patch))
                dy, dx = np.unravel_index(flat_idx, patch.shape)
                target_y = y0 + float(dy)
                target_x = x0 + float(dx)
                neigh_y = 0.5 * (pts[i - 1, 0] + pts[i + 1, 0])
                neigh_x = 0.5 * (pts[i - 1, 1] + pts[i + 1, 1])
                new_pts[i, 0] = iw * target_y + sw * neigh_y
                new_pts[i, 1] = iw * target_x + sw * neigh_x
            pts = new_pts
        data['coordinates'] = [tuple(p) for p in pts]
        snapped += 1
    log(f"[SNAKE] snake-snapped {snapped} edges "
        f"(iters={n_iterations}, sr={sr}, iw={iw}, sw={sw})")
    return G


def Graph_Npy(npy_path: str, out_base: str, low_thr: float, high_thr: float,
              spur_thr: int, log=print, geojson_path: Optional[str] = None,
              max_hole_size: int = 50, gap_bridge_length: int = 7,
              junction_merge_radius: int = 3, min_object_size: int = 100,
              min_skeleton_component: int = 20,
              endpoint_reconnect_radius_factor: float = 2.0,
              endpoint_reconnect_max_bearing_deg: float = 30.0,
              input_img_path: Optional[str] = None,
              # Image-evidence pruning (A)
              evidence_enabled: bool = True,
              evidence_ink_lum_threshold: int = 160,
              evidence_min_casing_fraction: float = 0.30,
              evidence_casing_offset_factor: float = 0.60,
              evidence_casing_extra_search_px: int = 2,
              # Snake snapping (B)
              snake_enabled: bool = True,
              snake_iterations: int = 8,
              snake_search_radius_px: Optional[int] = None,
              snake_image_weight: float = 0.55,
              snake_smoothness_weight: float = 0.45):
    Util.ensure_subfolders(out_base, ["binary", "skeleton", "graph_viz", "stats"])
    name = os.path.splitext(os.path.basename(npy_path))[0]
    raw_data = np.load(npy_path)
    if raw_data.ndim > 2: raw_data = raw_data[:,:,0]

    skel, cleaned = skeleton.process_skeleton(
        npy_path, out_base,
        low_thr=low_thr, high_thr=high_thr, prune_iter=spur_thr,
        max_hole_size=max_hole_size, gap_bridge_length=gap_bridge_length,
        junction_merge_radius=junction_merge_radius, min_object_size=min_object_size,
        min_skeleton_component=min_skeleton_component,
    )
    road_width = compute_global_road_width(cleaned, skel)
    G_raw, _ = graph_from_skeleton(skel, prob_map=raw_data)

    cfg = PipelineConfig()
    cleaner = GraphCleaner(
        G_raw, cfg, road_width, image_shape=raw_data.shape, log=log,
        endpoint_reconnect_radius_factor=endpoint_reconnect_radius_factor,
        endpoint_reconnect_max_bearing_deg=endpoint_reconnect_max_bearing_deg,
    )
    G_final = cleaner.clean()

    # Create visualization strip BEFORE coordinate transformation
    PipelineVisualizer.create_strip(raw_data, cleaned, skel, G_raw, G_final,
                                   os.path.join(out_base, "graph_viz", f"{name}_pipeline.png"))

    # ── Image-evidence and snake stages (uses the BASE map, not the U-Net mask) ──
    # Both stages need a luminance view of the input map at the same pixel grid
    # as the prediction. Load + resize once; reuse for both.
    base_image_lum = None
    if input_img_path and (evidence_enabled or snake_enabled):
        base_image_lum, ok = _load_base_image_as_luminance(input_img_path, raw_data.shape)
        if not ok:
            log(f"[WARN] couldn't load base image for evidence/snake passes")
            base_image_lum = None

    # Temp underlay (prob map) is reused for every viz so each side-output sits
    # over the same canvas. The actual underlay shown is `input_img_path`.
    temp_img_path = os.path.join(out_base, f"temp_{name}.png")
    try:
        cv2.imwrite(temp_img_path, (raw_data * 255).astype(np.uint8))

        # Stage 1: post-geometric-cleanup (pre image-evidence)
        Util.visualize_graph_final(
            temp_img_path, G_final,
            os.path.join(out_base, "graph_viz", f"{name}_stage1_cleaned.png"),
            underlay_path=input_img_path,
        )

        # Stage 2: image-evidence pruning (drops hallucinated edges)
        if evidence_enabled and base_image_lum is not None:
            G_final = image_evidence_prune(
                G_final, base_image_lum,
                road_width_px=road_width,
                ink_lum_threshold=int(evidence_ink_lum_threshold),
                min_dark_fraction=float(evidence_min_casing_fraction),
                casing_offset_factor=float(evidence_casing_offset_factor),
                casing_extra_search_px=int(evidence_casing_extra_search_px),
                log=log,
            )
            Util.visualize_graph_final(
                temp_img_path, G_final,
                os.path.join(out_base, "graph_viz", f"{name}_stage2_evidence.png"),
                underlay_path=input_img_path,
            )

        # Stage 3: snake snapping (slides edge interiors onto ink)
        if snake_enabled and base_image_lum is not None:
            G_final = snake_snap_edges(
                G_final, base_image_lum,
                road_width_px=road_width,
                n_iterations=int(snake_iterations),
                search_radius_px=snake_search_radius_px,
                image_weight=float(snake_image_weight),
                smoothness_weight=float(snake_smoothness_weight),
                log=log,
            )
            Util.visualize_graph_final(
                temp_img_path, G_final,
                os.path.join(out_base, "graph_viz", f"{name}_stage3_snake.png"),
                underlay_path=input_img_path,
            )

        # Final viz — same as last stage but kept under the original filename
        # so the pipeline's expected output path doesn't change.
        Util.visualize_graph_final(
            temp_img_path, G_final,
            os.path.join(out_base, "graph_viz", f"{name}_final.png"),
            underlay_path=input_img_path,
        )
    finally:
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

    # TRANSFORM COORDINATES FOR GEOJSON (Invert Y)
    for nid in G_final.nodes:
        y, x = G_final.nodes[nid]['pos']
        G_final.nodes[nid]['pos'] = (- y, x)

    # Update edge coordinates
    for u, v, data in G_final.edges(data=True):
        data['coordinates'] = [(-y, x) for y, x in data['coordinates']]
       
    
    if geojson_path is None:
        geojson_path = os.path.join(out_base, f"{name}_roads.geojson")
    Util.export_to_geojson(G_final, geojson_path)
    log(f"[DONE] Pipeline complete. GeoJSON: {geojson_path}")
    
    # 6. Stats
    n_nodes = G_final.number_of_nodes()
    n_edges = G_final.number_of_edges()
    stats = GraphStats(
        tile_name=name,
        road_width_px=road_width,
        nodes=n_nodes,
        edges=n_edges,
        intersections=sum(1 for n,d in G_final.nodes(data=True) if d.get('type')=='intersection'),
        endpoints=sum(1 for n,d in G_final.nodes(data=True) if d.get('type')=='endpoint'),
        components=nx.number_connected_components(G_final),
        avg_degree=(2.0 * n_edges / n_nodes) if n_nodes else 0.0,  # handshake lemma
        total_edge_length_px=sum(d.get('length_px',0) for u,v,d in G_final.edges(data=True)),
        largest_component_nodes=len(max(nx.connected_components(G_final), key=len)) if n_nodes else 0
    )
    
    save_json(os.path.join(out_base, "stats", f"{name}_stats.json"), asdict(stats), indent=2)
        
    return G_final, stats

if __name__ == "__main__":

    npy_file = 'data/Truth_Data/Waterloo/UNET/1083315_1193-41-Waterloo.npy'
    out_dir = 'data/Output/Atlas_of_Sydney/Truths/Waterloo'
    print(f"Running Graph Extraction on {npy_file} -> {out_dir}")
    Graph_Npy(npy_file, out_dir, low_thr=0.2, high_thr=0.5, spur_thr=2)