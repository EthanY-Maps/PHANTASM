"""Utilities for Graphing"""
import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.filters import apply_hysteresis_threshold
import networkx as nx
from typing import List, Optional
from PIL import Image

from utils import save_json

def load_and_hysteresis_npy(npy_path, low=0.35, high=0.6):
    print(f"Loading {npy_path}...")
    data = np.load(npy_path)
    if data.ndim == 3: data = np.squeeze(data)

    # Smooth before thresholding so jagged probability edges don't survive into the binary mask.
    data = cv2.GaussianBlur(data, (5, 5), 0)

    print(f"  - Applying Hysteresis (Low: {low}, High: {high})...")
    binary_mask = apply_hysteresis_threshold(data, low, high)
    return (binary_mask.astype(np.uint8) * 255)


def clean_image(img, kernel_size=3):
    """Median blur + morphological close/open. Smooths contours without destroying topology."""
    img = cv2.medianBlur(img, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return img

def remove_small_objects(img, min_size=50):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    mask = np.zeros_like(img, dtype=np.uint8)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_size:
            mask[labels == label_id] = 255
    return mask


def fill_small_holes(binary: np.ndarray, max_hole_size: int) -> np.ndarray:
    """Fill interior holes in the foreground that are smaller than `max_hole_size` pixels.

    U-Net road predictions frequently have 1–5 pixel speckle holes inside thick
    roads. After skeletonisation those holes become false bifurcations — the
    skeleton routes *around* each hole, producing a loop and two short spurs
    where there should be one centreline. Filling them at the binary stage
    eliminates the loops entirely.

    Distinct from `scipy.ndimage.binary_fill_holes`, which fills ALL interior
    holes regardless of size and would also collapse genuine empty regions
    (building footprints, courtyards). We only touch holes below the threshold.
    """
    if max_hole_size <= 0:
        return binary

    h, w = binary.shape
    inverted = (binary == 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=4)

    fill_mask = np.zeros_like(binary)
    for i in range(1, n_labels):
        x, y, ww, hh, area = stats[i]
        # A component touching any image border is exterior space, not a hole.
        if x == 0 or y == 0 or x + ww == w or y + hh == h:
            continue
        if area < max_hole_size:
            fill_mask[labels == i] = 255

    return cv2.bitwise_or(binary, fill_mask)


def remove_small_skeleton_components(skel: np.ndarray, min_pixels: int) -> np.ndarray:
    """Drop skeleton components smaller than `min_pixels` pixels.

    `remove_small_objects` already runs on the binary mask, but skeletonisation
    thins each blob, so a 100-pixel binary patch might become a 10-pixel
    skeleton fragment that the binary-stage filter missed. The downstream
    length filter would delete these anyway — this skips the wasted graph
    build for them.
    """
    if min_pixels <= 1:
        return skel
    skel_u8 = (skel > 0).astype(np.uint8)
    if not skel_u8.any():
        return skel

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skel_u8, connectivity=8)
    kept = np.zeros_like(skel)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_pixels:
            kept[labels == i] = 255
    return kept


def bridge_oriented_gaps(binary: np.ndarray, kernel_length: int) -> np.ndarray:
    """Close short gaps in linear features using four oriented closings (0°/45°/90°/135°).

    A standard `MORPH_CLOSE` with a circular kernel bridges gaps in any
    direction up to the kernel radius — useful for noise but also liable to
    merge nearby parallel roads. Oriented line kernels close gaps only when
    pixels on either side are already aligned along that axis, which is what
    we want for broken road predictions without fusing adjacent geometry.
    """
    if kernel_length < 3:
        return binary

    L = kernel_length
    k_h  = np.ones((1, L), dtype=np.uint8)
    k_v  = np.ones((L, 1), dtype=np.uint8)
    k_d1 = np.eye(L, dtype=np.uint8)
    k_d2 = np.fliplr(k_d1)

    result = binary.copy()
    for k in (k_h, k_v, k_d1, k_d2):
        result = cv2.bitwise_or(result, cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k))
    return result


def ensure_subfolders(base_folder: str, subfolders: List[str]) -> None:
    for sub in subfolders:
        os.makedirs(os.path.join(base_folder, sub), exist_ok=True)

def export_to_geojson(G: nx.Graph, out_path: str):
    feats = []
    for u, v, data in G.edges(data=True):
        coords = data.get('coordinates', [])
        xy_coords = [[float(c[1]), float(c[0])] for c in coords]  # (y, x) → (x, y) for GeoJSON

        # Fallback when edge has no `coordinates`: Graph_Npy already negated `pos`
        # before calling here, so do NOT negate y again — that flips fallback edges
        # to the opposite hemisphere from the rest.
        if not xy_coords:
            y1, x1 = G.nodes[u]['pos']
            y2, x2 = G.nodes[v]['pos']
            xy_coords = [[x1, y1], [x2, y2]]

        props = {"length": data.get('length_px', 0), "id": f"{u}-{v}"}
        # Per-edge U-Net confidence, when graph_from_skeleton was given a prob_map.
        for k in ("mean_prob", "min_prob"):
            v_ = data.get(k)
            if v_ is not None:
                props[k] = round(float(v_), 3)
        feats.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": xy_coords},
            "properties": props,
        })

    # Path nodes (degree 2) are skipped — only intersections and endpoints get a feature.
    for n, data in G.nodes(data=True):
        n_type = data.get('type')
        if n_type in ('intersection', 'endpoint'):
            y, x = data['pos']
            feats.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(x), float(y)]},
                "properties": {"type": n_type, "id": str(n)},
            })

    save_json(out_path, {"type": "FeatureCollection", "features": feats})


def visualize_graph_final(base_img_path: str, G: nx.Graph, out_path: str,
                          underlay_path: Optional[str] = None,
                          underlay_alpha: float = 0.18):
    """Render the cleaned graph as a road map.

    `base_img_path` sets the canvas dimensions (kept for back-compat — the
    prob-map written by `Graph_Npy` is the natural choice).
    `underlay_path`, if given, is composited at `underlay_alpha` beneath the
    roads. Pass the original input map here for a "does this line up?" view.
    """
    base = cv2.imread(base_img_path)
    if base is None:
        print(f"Error: Could not load image at {base_img_path}")
        return
    H, W = base.shape[:2]

    # Cream paper background. cv2 is BGR.
    paper = (240, 244, 245)
    canvas = np.full((H, W, 3), paper, dtype=np.uint8)

    if underlay_path:
        under = cv2.imread(underlay_path)
        if under is not None:
            if under.shape[:2] != (H, W):
                under = cv2.resize(under, (W, H), interpolation=cv2.INTER_AREA)
            # Desaturate + brighten the underlay so it reads as a ghost behind
            # the graph rather than competing ink. Grayscale → gamma 1.6 lift →
            # back to BGR → low-alpha blend over the cream paper.
            gray = cv2.cvtColor(under, cv2.COLOR_BGR2GRAY)
            lifted = np.power(gray / 255.0, 1.0 / 1.6) * 255.0
            ghost = cv2.cvtColor(lifted.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            canvas = cv2.addWeighted(canvas, 1.0 - underlay_alpha, ghost, underlay_alpha, 0)

    casing_color = (30, 35, 45)       # near-black for the outer stroke
    center_color = (40, 90, 230)      # vivid red-orange (BGR) — "this is the graph"
    casing_thickness = 4
    center_thickness = 2
    aa = cv2.LINE_AA

    # Two-pass draw: casing on all edges first, then centers, so casings of
    # crossing edges never sit on top of another edge's center.
    for u, v, data in G.edges(data=True):
        coords = data.get('coordinates', [])
        if len(coords) > 0:
            pts = np.array(coords, np.int32)[:, ::-1].reshape((-1, 1, 2))  # (row,col) → (x,y)
            cv2.polylines(canvas, [pts], False, casing_color, casing_thickness, aa)
        else:
            y1, x1 = G.nodes[u]['pos']
            y2, x2 = G.nodes[v]['pos']
            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)),
                     casing_color, casing_thickness, aa)

    for u, v, data in G.edges(data=True):
        coords = data.get('coordinates', [])
        if len(coords) > 0:
            pts = np.array(coords, np.int32)[:, ::-1].reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], False, center_color, center_thickness, aa)
        else:
            y1, x1 = G.nodes[u]['pos']
            y2, x2 = G.nodes[v]['pos']
            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)),
                     center_color, center_thickness, aa)

    junction_color = (70, 110, 200)   # warm red-orange in BGR
    endpoint_color = (170, 130, 60)   # muted teal
    for n, data in G.nodes(data=True):
        y, x = data['pos']
        degree = G.degree[n]
        n_type = data.get('type')
        is_intersection = (n_type == 'intersection') or (degree > 2)
        is_endpoint     = (n_type == 'endpoint') or (degree == 1)
        if is_intersection:
            cv2.circle(canvas, (int(x), int(y)), 4, casing_color, -1, aa)
            cv2.circle(canvas, (int(x), int(y)), 3, junction_color, -1, aa)
        elif is_endpoint:
            cv2.circle(canvas, (int(x), int(y)), 3, endpoint_color, -1, aa)

    cv2.imwrite(out_path, canvas)

    

