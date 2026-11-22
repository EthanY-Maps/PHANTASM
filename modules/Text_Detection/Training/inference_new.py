import os
import sys
import json
import time
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from shapely.ops import unary_union
from skimage.exposure import match_histograms
from config import *

# --- 1. SETTINGS & CONFIG ---
Image.MAX_IMAGE_PIXELS = None 

INFERENCE_BATCH_SIZE = 128
STRIDE = 160
CONFIDENCE_THRESHOLD = 0.1 # Lower threshold since we have TTA/Matching
IOU_MERGE_THRESHOLD = 0.20

# Path to the image used for training (or the style profile png)
REFERENCE_IMAGE_PATH = "data/Output/1903_Testing/Text_Detection/style_profile.png"

# --- 2. HELPERS ---

def load_scale_from_metadata(npy_path):
    print(f"Loading metadata from {npy_path}...")
    try:
        data = np.load(npy_path, allow_pickle=True)
        if "scale" in data.item():
            return data.item()["scale"]
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return 1.0

def load_models():
    print("Loading Models...")
    from training_utils import hybrid_blob_loss, f1_metric
    
    path_1 = os.path.join(CHECKPOINT_DIR, "stage1_center_best.keras")
    model_1 = tf.keras.models.load_model(path_1, custom_objects={'hybrid_blob_loss': hybrid_blob_loss, 'f1_metric': f1_metric}, compile=False)
    
    path_2 = os.path.join(CHECKPOINT_DIR, "stage2_poly_best.keras")
    model_2 = tf.keras.models.load_model(path_2, compile=False)
    
    return model_1, model_2

def prepare_reference_stats(ref_path):
    """
    Pre-calculates the color statistics of the training image.
    We do this once so we can quickly apply it to small crops later.
    """
    print("Loading Reference Image Stats...")
    ref = Image.open(ref_path).convert('RGB')
    # If using the tiny profile, no resize needed. If using full map, resize to 1024x1024.
    if ref.size[0] > 1024:
        ref = ref.resize((1024, 1024)) 
    return np.array(ref)

# --- 3. STAGE 1: DETECTION (ON SCALED MAP) ---

def get_centers_on_scaled(model, scaled_img_array, threshold=CONFIDENCE_THRESHOLD):
    """
    Runs heatmap detection on the DOWN-SCALED map.
    Returns center coordinates in the SCALED coordinate system.
    """
    h, w, c = scaled_img_array.shape
    print(f"Stage 1: Scanning Scaled Map ({w}x{h})...")
    
    pad_h = (PATCH_SIZE - h % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - w % PATCH_SIZE) % PATCH_SIZE
    img_padded = np.pad(scaled_img_array, ((0, pad_h), (0, pad_w), (0, 0)))
    final_h, final_w = img_padded.shape[:2]
    
    heatmap_canvas = np.zeros((final_h, final_w), dtype=np.float32)
    all_coords = [(y, x) for y in range(0, final_h - PATCH_SIZE + 1, STRIDE) 
                         for x in range(0, final_w - PATCH_SIZE + 1, STRIDE)]
            
    for i in tqdm(range(0, len(all_coords), INFERENCE_BATCH_SIZE), desc="Heatmap Scan"):
        batch_coords = all_coords[i : i + INFERENCE_BATCH_SIZE]
        batch_images = [img_padded[y:y+PATCH_SIZE, x:x+PATCH_SIZE] for y, x in batch_coords]
        
        if batch_images:
            preds = model.predict(np.array(batch_images), verbose=0)
            for j, (py, px) in enumerate(batch_coords):
                heatmap_canvas[py:py+PATCH_SIZE, px:px+PATCH_SIZE] = np.maximum(
                    heatmap_canvas[py:py+PATCH_SIZE, px:px+PATCH_SIZE], 
                    preds[j, ..., 0]
                )

    heatmap_final = heatmap_canvas[:h, :w]
    binary = (heatmap_final > threshold).astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    
    centers_scaled = []
    for i in range(1, num):
        cx, cy = centroids[i]
        centers_scaled.append([int(cx), int(cy)])
    
    return centers_scaled, heatmap_final

# --- 4. STAGE 2: REGRESSION (ON FULL MAP) ---

def predict_polygons_on_full(model, full_pil_image, centers_scaled, scale_factor, ref_stats):
    """
    1. Upscales the centers to the Full Map coordinates.
    2. Crops 128x128 patches from the Full Map.
    3. Applies histogram matching to the crop.
    4. Predicts.
    """
    if not centers_scaled: return []
    
    polygons_global = []
    w, h = full_pil_image.size
    
    # Pre-calculate upscale ratio
    upscale = 1.0 / scale_factor
    
    # Calculate Full-Res Centers
    centers_full = []
    for (cx_s, cy_s) in centers_scaled:
        centers_full.append((int(cx_s * upscale), int(cy_s * upscale)))
        
    print(f"Stage 2: Processing {len(centers_full)} centers on FULL RES map...")

    for i in tqdm(range(0, len(centers_full), INFERENCE_BATCH_SIZE), desc="Poly Regression"):
        batch_centers = centers_full[i : i + INFERENCE_BATCH_SIZE]
        batch_crops = []
        batch_offsets = [] # To map back to global coords
        
        for (cx, cy) in batch_centers:
            # Crop 128x128 from the massive original map
            x1, y1 = cx - 64, cy - 64
            
            # Use PIL crop (lazy loading handles massive TIFs better than numpy slicing)
            # PIL crop is (left, top, right, bottom)
            patch = full_pil_image.crop((x1, y1, x1+128, y1+128))
            
            # Handle edge padding if patch is smaller than 128
            if patch.size != (128, 128):
                new_patch = Image.new("RGB", (128, 128))
                new_patch.paste(patch, (0, 0))
                patch = new_patch
            
            # Convert to numpy
            patch_arr = np.array(patch)
            
            # --- APPLY HISTOGRAM MATCHING TO CROP ---
            # This fixes the "blind map" issue on the high-res crop
            patch_matched = match_histograms(patch_arr, ref_stats, channel_axis=-1)
            
            batch_crops.append(patch_matched.astype(np.float32) / 255.0)
            batch_offsets.append((x1, y1))
            
        if batch_crops:
            preds = model.predict(np.array(batch_crops), verbose=0)
            
            for k, rays in enumerate(preds):
                ox, oy = batch_offsets[k] # Offset (Top-Left of crop)
                
                # Center of the crop is (64, 64) relative to the crop
                # Rays are 0.0-1.0, scaled by 128.0 (MAX_RAY_DIST)
                real_rays = rays * 128.0 
                
                poly_points = []
                angles = np.linspace(0, 360, NUM_RAYS, endpoint=False)
                
                for angle, dist in zip(angles, real_rays):
                    if dist < 2: continue
                    rad = np.deg2rad(angle)
                    
                    # Calculate point relative to crop center (64, 64)
                    px_local = 64 + dist * np.cos(rad)
                    py_local = 64 + dist * np.sin(rad)
                    
                    # Convert to GLOBAL coordinates
                    px_global = ox + px_local
                    py_global = oy + py_local
                    
                    poly_points.append([int(px_global), int(py_global)])
                
                poly_points = np.array(poly_points, dtype=np.int32)
                if len(poly_points) >= 3:
                    hull = cv2.convexHull(poly_points)
                    polygons_global.append(hull[:, 0, :])
                else:
                    polygons_global.append(poly_points)

    return polygons_global

# --- 5. MERGE ---

def merge_polygons(cv2_polygons, threshold=IOU_MERGE_THRESHOLD):
    print(f"Merging {len(cv2_polygons)} polygons...")
    shapely_polys = []
    for p in cv2_polygons:
        if len(p) >= 3:
            poly = Polygon(p).buffer(0)
            if poly.is_valid and not poly.is_empty:
                shapely_polys.append(poly)
    
    if not shapely_polys: return []
    
    tree = STRtree(shapely_polys)
    adj = {i: set() for i in range(len(shapely_polys))}
    
    # Query overlaps
    try:
        indices = tree.query(shapely_polys)
        pairs = zip(indices[0], indices[1])
    except:
        pairs = []
        geom_idx = {id(p): i for i, p in enumerate(shapely_polys)}
        for i, p in enumerate(shapely_polys):
            for res in tree.query(p):
                pairs.append((i, geom_idx[id(res)]))
                
    for i, j in pairs:
        if i >= j: continue
        pa, pb = shapely_polys[i], shapely_polys[j]
        if pa.intersects(pb):
            try:
                iou = pa.intersection(pb).area / pa.union(pb).area
                if iou > threshold:
                    adj[i].add(j); adj[j].add(i)
            except: pass

    # Cluster
    visited = [False] * len(shapely_polys)
    merged = []
    for i in range(len(shapely_polys)):
        if visited[i]: continue
        stack = [i]; visited[i] = True; cluster = []
        while stack:
            curr = stack.pop()
            cluster.append(shapely_polys[curr])
            for n in adj[curr]:
                if not visited[n]:
                    visited[n] = True; stack.append(n)
        merged.append(unary_union(cluster))
        
    final_coords = []
    for m in merged:
        if m.geom_type == 'MultiPolygon':
            final_coords.extend([np.array(g.exterior.coords) for g in m.geoms])
        else:
            final_coords.append(np.array(m.exterior.coords))
            
    return final_coords

# --- 6. EXPORT ---

def save_results(image_path, output_dir, polygons, heatmap, full_pil, scale_factor):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save JSON (Polygons are ALREADY in global coords, no scaling needed!)
    json_path = os.path.join(output_dir, f"{base_name}_predictions.json")
    print(f"Saving JSON to {json_path}...")
    
    feature_collection = {
        "type": "FeatureCollection", "name": base_name,
        "features": []
    }
    
    for i, poly in enumerate(polygons):
        points = poly.tolist()
        if points[0] != points[-1]: points.append(points[0])
        feature = {
            "type": "Feature", "properties": { "id": i },
            "geometry": { "type": "Polygon", "coordinates": [points] }
        }
        feature_collection["features"].append(feature)
        
    with open(json_path, "w") as f: json.dump(feature_collection, f, indent=4)
    
    # --- SAVE HEATMAP ---
    # The heatmap corresponds to the SCALED image dimensions.
    print("Saving Heatmap...")
    heatmap_norm = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_heatmap.jpg"), heatmap_color)

    # --- SAVE DEBUG VISUALS (DOWNSCALED) ---
    print("Saving Overlay Visuals (Downscaled)...")
    vis_w, vis_h = int(full_pil.width * 0.2), int(full_pil.height * 0.2)
    vis_pil = full_pil.resize((vis_w, vis_h))
    vis_img = np.array(vis_pil)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    
    # Scale polys down just for drawing
    draw_polys = [(p * 0.2).astype(np.int32) for p in polygons]
    cv2.drawContours(vis_img, draw_polys, -1, (0, 0, 255), 1)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_overlay.jpg"), vis_img)

# --- MAIN ---

def run_hybrid_inference(image_path, npy_path, output_dir):
    m1, m2 = load_models()
    scale = load_scale_from_metadata(npy_path)
    ref_stats = prepare_reference_stats(REFERENCE_IMAGE_PATH)
    
    print(f"Opening Massive Image: {image_path}")
    full_pil = Image.open(image_path).convert('RGB')
    orig_w, orig_h = full_pil.size
    
    # 1. Create Scaled Version for Stage 1
    target_w, target_h = int(orig_w * scale), int(orig_h * scale)
    print(f"Creating Scaled View for Detection: {target_w}x{target_h}")
    print(f"Used Scale Factor: {scale}")

    scaled_pil = full_pil.resize((target_w, target_h), Image.BILINEAR)
    scaled_arr = np.array(scaled_pil).astype(np.float32) / 255.0
    
    # 2. Histogram Match (Scaled)
    # Note: match_histograms needs uint8 usually for speed, we handle float conversion inside
    scaled_uint8 = (scaled_arr * 255).astype(np.uint8)
    matched_uint8 = match_histograms(scaled_uint8, ref_stats, channel_axis=-1)
    matched_scaled_arr = matched_uint8.astype(np.float32) / 255.0
    
    # 3. Detect Centers (On Scaled)
    centers_scaled, heatmap = get_centers_on_scaled(m1, matched_scaled_arr)
    
    # 4. Predict Polygons (On Full Res Crops)
    # We pass 'ref_stats' so we can fix the color of the high-res crops on the fly

    if scale > 1:
        raw_polys = predict_polygons_on_full(m2, scaled_pil, centers_scaled, scale, ref_stats)
    else:
        raw_polys = predict_polygons_on_full(m2, full_pil, centers_scaled, scale, ref_stats)
    
    # 5. Merge & Save
    merged_polys = merge_polygons(raw_polys)
    save_results(image_path, output_dir, merged_polys, heatmap, full_pil, scale)

if __name__ == "__main__":
    TEST_IMAGE = "data/Input/1889 Strathfield/Strathfield 1889-1894.jpg"
    METADATA   = 'data/Output/Strathefield_Testing/Predictions/npy_raw/Strathfield 1889-1894_metadata.npy'
    OUTPUT_DIR = "data/Output/Strathefield_Testing/Text_Detection"
    
    run_hybrid_inference(TEST_IMAGE, METADATA, OUTPUT_DIR)