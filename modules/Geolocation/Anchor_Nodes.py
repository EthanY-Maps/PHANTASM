import logging
import os
from contextlib import contextmanager

import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from shapely.geometry import Point, LineString, Polygon
from skimage.measure import ransac
from skimage.transform import AffineTransform
from scipy.interpolate import RBFInterpolator
from shapely.affinity import affine_transform

# ---------------- Configuration ----------------

log = logging.getLogger("anchor_pipeline")


@contextmanager
def _scoped_numpy_seed(seed: int):
    """Temporarily seed numpy's global RNG so RANSAC is reproducible, then
    restore the prior state. The previous implementation seeded at module
    import time, leaking the seed to every other numpy.random consumer in
    the pipeline (TTA, scale optimization sampling, etc.)."""
    saved = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(saved)

# ---------------- Logic Helpers ----------------

def ensure_lon_lat(points):
    """
    Cast (lon, lat) tuples to a float numpy array.

    Input is required to be (lon, lat) — the GUI labels the bounding-coord
    fields explicitly as `(lon,lat)` and `parse_bounding_coordinates` preserves
    that order. The previous `abs(a) > 90` heuristic to auto-detect axis order
    silently swapped axes for any location with `|lon| <= 90` (UK, Iceland,
    Cape Town, NYC, most of Africa/Europe/the Americas east of meridian-90),
    breaking anchor matching outside Australia.
    """
    return np.array([(float(a), float(b)) for a, b in points], dtype=float)

def create_rectangle_from_points(points):
    pts = ensure_lon_lat(points)
    xs, ys = pts[:, 0], pts[:, 1]
    return Polygon([(min(xs), min(ys)), (min(xs), max(ys)), (max(xs), max(ys)), (max(xs), min(ys))])

def get_robust_bounds(pts):
    min_b = np.percentile(pts, 1, axis=0)
    max_b = np.percentile(pts, 99, axis=0)
    return min_b, max_b

def filter_outliers_ransac(old_pts, new_pts, RANSAC_THRESHOLD):
    if len(old_pts) < 4: 
        # If we have 3 points, we can't really do RANSAC, but we can return all as True
        # to attempt an Affine fit on what we have.
        return np.ones(len(old_pts), dtype=bool)
    
    old_min, old_max = get_robust_bounds(old_pts)
    new_min, new_max = get_robust_bounds(new_pts)
    
    old_clipped = np.clip(old_pts, old_min, old_max)
    new_clipped = np.clip(new_pts, new_min, new_max)

    norm_old = (old_clipped - old_min) / (old_max - old_min + 1e-9)
    norm_new = (new_clipped - new_min) / (new_max - new_min + 1e-9)

    with _scoped_numpy_seed(250):
        model, inliers = ransac((norm_old, norm_new), AffineTransform, min_samples=3,
                                residual_threshold=RANSAC_THRESHOLD, max_trials=5000)

    if inliers is None: return np.zeros(len(old_pts), dtype=bool)
    return inliers

def pre_scale_gdf(gdf, clean_old, clean_new):
    scale = (clean_new.max(axis=0) - clean_new.min(axis=0)) / \
            (clean_old.max(axis=0) - clean_old.min(axis=0) + 1e-9)
    offset = clean_new.min(axis=0) - (clean_old.min(axis=0) * scale)
    matrix = [scale[0], 0, 0, scale[1], offset[0], offset[1]]
    gdf['geometry'] = gdf['geometry'].apply(lambda g: affine_transform(g, matrix))
    return gdf, (clean_old * scale) + offset

# --- FALLBACK 1: BOUNDS STRETCH (N < 3) ---
def fit_gdf_to_bounds(gdf, target_polygon):
    minx, miny, maxx, maxy = gdf.total_bounds
    t_minx, t_miny, t_maxx, t_maxy = target_polygon.bounds
    
    scale_x = (t_maxx - t_minx) / (maxx - minx + 1e-9)
    scale_y = (t_maxy - t_miny) / (maxy - miny + 1e-9)
    off_x = t_minx - (minx * scale_x)
    off_y = t_miny - (miny * scale_y)
    
    matrix = [scale_x, 0, 0, scale_y, off_x, off_y]
    
    new_gdf = gdf.copy()
    new_gdf['geometry'] = new_gdf['geometry'].apply(lambda g: affine_transform(g, matrix))
    return new_gdf

# --- FALLBACK 2: AFFINE TRANSFORM (3 <= N < 10) ---
def apply_affine_transform(gdf, anchor_old, anchor_new):
    """Calculates a global Least-Squares Affine fit.

    Returns None if the affine estimation fails (e.g. collinear or duplicate
    anchors). Caller should fall back to a simpler method like bounds-fit.
    Previously the unchecked `model.estimate(...)` left `model.params` as the
    identity matrix on failure, silently producing un-warped output that
    looked like a successful geolocation.
    """
    model = AffineTransform()
    if not model.estimate(anchor_old, anchor_new):
        log.warning("Affine estimation failed (degenerate anchor configuration); "
                    "caller should fall back.")
        return None

    # Skimage matrix is 3x3: [[a, b, xoff], [d, e, yoff], [0, 0, 1]]
    # Shapely expects: [a, b, d, e, xoff, yoff]
    p = model.params
    matrix = [p[0,0], p[0,1], p[1,0], p[1,1], p[0,2], p[1,2]]

    log.info(f"Applying Global Affine Transform:\n{p}")

    new_gdf = gdf.copy()
    new_gdf['geometry'] = new_gdf['geometry'].apply(lambda g: affine_transform(g, matrix))
    return new_gdf

# --- PRIMARY METHOD: TPS WARP (N >= 10) ---
def apply_tps_transform(gdf, anchor_old, anchor_new, TPS_SMOOTHING):
    _, unique_indices = np.unique(anchor_old, axis=0, return_index=True)
    clean_old = anchor_old[unique_indices]
    clean_new = anchor_new[unique_indices]

    interp = RBFInterpolator(clean_old, clean_new, kernel='thin_plate_spline', smoothing=TPS_SMOOTHING)

    def warp_geom(geom):
        if geom.is_empty: return geom
        def tr(coords): return interp(np.array(coords))
        
        if geom.geom_type == 'Point': return Point(tr(geom.coords)[0])
        elif geom.geom_type in ['LineString', 'LinearRing']: return LineString(tr(geom.coords))
        elif geom.geom_type == 'Polygon': 
            ext = tr(geom.exterior.coords)
            ints = [tr(ring.coords) for ring in geom.interiors]
            return Polygon(ext, ints)
        elif geom.geom_type.startswith('Multi'): return type(geom)([warp_geom(g) for g in geom.geoms])
        return geom

    gdf['geometry'] = gdf['geometry'].apply(warp_geom)
    return gdf

# ---------------- Visualizers ----------------

def vis_step_1_raw_matches(a_old, a_new, plot_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(a_old[:, 0], a_old[:, 1], c='gray', s=10)
    ax1.set_title("1. Raw Historical Anchors")
    ax2.scatter(a_new[:, 0], a_new[:, 1], c='blue', s=10)
    ax2.set_title("1. Raw Modern Targets")
    plt.savefig(os.path.join(plot_path, "step_1_raw_matches.png"))
    plt.close()

def vis_step_2_clean_anchors(a_old, a_new, mask, plot_path):
    clean_o, clean_n = a_old[mask], a_new[mask]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(a_old[:, 0], a_old[:, 1], c='red', s=5, alpha=0.3, label='Discarded')
    ax1.scatter(clean_o[:, 0], clean_o[:, 1], c='green', s=15, label='Inliers')
    ax1.set_title(f"2. RANSAC Filtering (N={sum(mask)})")
    ax1.legend()
    ax2.scatter(clean_n[:, 0], clean_n[:, 1], c='green', s=15)
    ax2.set_title("2. Validated GPS Targets")
    plt.savefig(os.path.join(plot_path, "step_2_clean_anchors.png"))
    plt.close()

def vis_step_3_overlay(warped_gdf, modern_gdf, plot_path, method_label):
    fig, ax = plt.subplots(figsize=(20, 20))
    modern_gdf.plot(ax=ax, color='blue', alpha=0.2, label='Modern OSM')
    warped_gdf.plot(ax=ax, color='red', alpha=0.5, label=f'Warped ({method_label})')
    plt.title(f"3. Final Geolocated Overlay - Method: {method_label}")
    plt.legend()
    plt.savefig(os.path.join(plot_path, "step_3_overlay.png"))
    plt.close()

# ---------------- Pipeline ----------------

def anchor_pipeline(geojson_path, OSM_geojson, output_path, boundary_points,
                    RANSAC_THRESHOLD, TPS_SMOOTHING, match_threshold: float = 0.7):

    # Load Data
    original_gdf = gpd.read_file(geojson_path) 
    gdf = original_gdf.copy()
    
    modern_gdf = gpd.read_file(OSM_geojson)
    if modern_gdf.crs is None: modern_gdf.set_crs("EPSG:4326", inplace=True)
    if gdf.crs is None:
        log.warning("Historic GeoJSON has no CRS metadata; assuming EPSG:4326. "
                    "If the data was in a different projection, set its CRS upstream.")
        gdf.set_crs("EPSG:4326", inplace=True)
    gdf = gdf.to_crs(modern_gdf.crs)

    output_parent = os.path.dirname(output_path)
    plot_path = os.path.join(output_parent, "plots")
    os.makedirs(plot_path, exist_ok=True)
    
    # 1. Boundary Filter
    rect = create_rectangle_from_points(boundary_points)
    modern_subset = modern_gdf[modern_gdf.geometry.intersects(rect)].copy()

    # 2. Extract Matches
    modern_map = {str(row.get("intersection_name", "")).strip().lower(): row.geometry.coords[0] 
                  for _, row in modern_subset.iterrows() if row.get("intersection_name")}
    
    a_old, a_new = [], []
    for _, row in gdf.iterrows():
        name = str(row.get("canonical_name", "")).strip().lower()
        score = row.get("confidence", 0)
        if name in modern_map and score > match_threshold:
            a_old.append(row.geometry.coords[0])
            a_new.append(modern_map[name])
    
    a_old, a_new = np.array(a_old), np.array(a_new)
    
    # ---------------- 3-TIER LOGIC ----------------
    final_gdf = None
    method_used = ""

    # Check Raw Count
    if len(a_old) < 3:
        log.warning("Tier 3: Too few raw matches (<3). Using Bounds Fit.")
        final_gdf = fit_gdf_to_bounds(original_gdf, rect)
        method_used = "Fallback (Bounds Fit)"
    else:
        # Pre-Scale
        scaled_gdf, a_old_scaled = pre_scale_gdf(gdf.copy(), a_old, a_new)
        vis_step_1_raw_matches(a_old_scaled, a_new, plot_path)

        # RANSAC
        mask = filter_outliers_ransac(a_old_scaled, a_new, RANSAC_THRESHOLD)
        vis_step_2_clean_anchors(a_old_scaled, a_new, mask, plot_path)
        
        inlier_count = np.sum(mask)
        c_old, c_new = a_old_scaled[mask], a_new[mask]

        if inlier_count < 3:
            log.warning(f"Tier 3: RANSAC left <3 points ({inlier_count}). Using Bounds Fit.")
            final_gdf = fit_gdf_to_bounds(original_gdf, rect)
            method_used = "Fallback (Bounds Fit)"
            
        elif inlier_count < 10:
            log.warning(f"Tier 2: Low inlier count ({inlier_count}). Using Global Affine (Stable).")
            # We use scaled_gdf here because c_old are coordinates from the scaled space
            final_gdf = apply_affine_transform(scaled_gdf, c_old, c_new)
            method_used = "Fallback (Affine)"
            if final_gdf is None:
                log.warning("Affine fit failed; falling back to Bounds Fit.")
                final_gdf = fit_gdf_to_bounds(original_gdf, rect)
                method_used = "Fallback (Bounds Fit, post-affine-failure)"
            
        else:
            log.info(f"Tier 1: Sufficient inliers ({inlier_count}). Using TPS Warp (Flexible).")
            final_gdf = apply_tps_transform(scaled_gdf, c_old, c_new, TPS_SMOOTHING)
            method_used = "TPS Warp"

    # VIS 3: Final Overlay
    vis_step_3_overlay(final_gdf, modern_subset, plot_path, method_used)

    final_gdf.to_file(output_path, driver='GeoJSON')
    log.info(f"Process Complete ({method_used}). Saved to {output_path}")


if __name__ == "__main__":
    bounds = [(-33.769, 151.312), (-33.92, 150.9956), (-33.769, 150.9956), (-33.92, 151.312)]
    
    anchor_pipeline(
        geojson_path='data/Output/1890_10-02-26/GeoLoc/graph_network_combined.geojson',
        OSM_geojson="data/Input/Model/sydney_intersections.geojson",
        output_path="data/Output/1890_10-02-26/GeoLoc/Sydney_1890_Geolocated.geojson",
        boundary_points=bounds,
        RANSAC_THRESHOLD=0.01,
        TPS_SMOOTHING=0.02
    )