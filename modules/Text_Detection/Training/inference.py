import os
import cv2
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from shapely.geometry import Polygon
from pdf2image import convert_from_path
from rtree import index  # Optimized spatial indexing

# --- CONFIGURATION & OPTIMIZATIONS ---
Image.MAX_IMAGE_PIXELS = None
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def get_patch_and_check(pil_img, x, y, patch_size):
    patch = pil_img.crop((x, y, x + patch_size, y + patch_size))
    patch_np = np.array(patch)
    if np.std(patch_np) < 8: return None
    
    if len(patch_np.shape) == 3:
        gray = cv2.cvtColor(patch_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = patch_np
        
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(bin_img, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0

# --- 1. SPATIAL DEDUPLICATION (R-TREE) ---

def spatial_nms_rtree(polys, iou_threshold=0.3):
    """Ultra-fast deduplication for high-density maps."""
    if not polys: return []
    idx = index.Index()
    data = {}
    for i, p in enumerate(polys):
        poly_obj = Polygon(p['points'])
        if not poly_obj.is_valid or poly_obj.area < 5: continue
        data[i] = {'poly': poly_obj, 'area': poly_obj.area, 'orig': p}
        idx.insert(i, poly_obj.bounds)
    
    sorted_ids = sorted(data.keys(), key=lambda k: data[k]['area'], reverse=True)
    keep, skipped = [], set()
    for i in sorted_ids:
        if i in skipped: continue
        curr = data[i]
        keep.append(curr['orig'])
        for j in list(idx.intersection(curr['poly'].bounds)):
            if j == i or j in skipped: continue
            inter = curr['poly'].intersection(data[j]['poly']).area
            if (inter / (curr['area'] + data[j]['area'] - inter)) > iou_threshold:
                skipped.add(j)
    return keep

def to_geojson(polys):
    """Convert custom polygon list to GeoJSON FeatureCollection."""
    features = []
    for i, p in enumerate(polys):
        # Ensure simplified polygon is closed
        pts = p['points']
        if pts[0] != pts[-1]:
            pts.append(pts[0])
            
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [pts]
            },
            "properties": {
                "id": i,
                "scale": p.get('scale', 1.0)
            }
        })
        
    return {
        "type": "FeatureCollection",
        "features": features
    }

# --- 2. INFERENCE & VISUALIZATION ---

def sliding_window_inference(model, pil_img, patch_size=256, stride=128, batch_size=64):
    w, h = pil_img.size
    prob_map = np.zeros((h, w, 2), dtype=np.float32)
    count_map = np.zeros((h, w, 1), dtype=np.float32)
    x_steps = range(0, max(1, w - patch_size + 1), stride)
    y_steps = range(0, max(1, h - patch_size + 1), stride)
    windows = [(x, y) for y in y_steps for x in x_steps]
    
    batch_imgs, batch_coords = [], []
    for (x, y) in tqdm(windows, desc="  Tiles", leave=False):
        inp = get_patch_and_check(pil_img, x, y, patch_size)
        if inp is None:
            count_map[y:y+patch_size, x:x+patch_size, 0] += 1
            continue
        batch_imgs.append(inp); batch_coords.append((x, y))
        if len(batch_imgs) >= batch_size:
            preds = model.predict_on_batch(np.array(batch_imgs))
            for i, (px, py) in enumerate(batch_coords):
                prob_map[py:py+patch_size, px:px+patch_size, :] += preds[i]
                count_map[py:py+patch_size, px:px+patch_size, 0] += 1
            batch_imgs, batch_coords = [], []

    if batch_imgs:
        preds = model.predict_on_batch(np.array(batch_imgs))
        for i, (px, py) in enumerate(batch_coords):
            prob_map[py:py+patch_size, px:px+patch_size, :] += preds[i]
            count_map[py:py+patch_size, px:px+patch_size, 0] += 1
            
    return prob_map / np.maximum(count_map, 1)

def run_inference(model_path, input_path, out_dir, npy_metadata_dir, batch_size=128):
    os.makedirs(out_dir, exist_ok=True)
    model = tf.keras.models.load_model(model_path, compile=False)
    

    scale_data = np.load(npy_metadata_dir, allow_pickle=True)
    scale = scale_data.item().get('scale')

    scales = []

    scales.append(scale)
   
    # PDF/Image Loading
    if input_path.lower().endswith('.pdf'):
        pages = convert_from_path(input_path, dpi=250, grayscale=True)
        p_img = pages[0].convert('RGB')
    else:
        p_img = Image.open(input_path).convert('RGB')

    orig_w, orig_h = p_img.size
    aggregated_results = []
    
    # Track temp files for cleanup
    temp_scaled_files = []

    for s in scales:
        print(f"Processing Scale {s}...")
        sw, sh = int(orig_w * s), int(orig_h * s)
        scaled_img = p_img.resize((sw, sh), Image.Resampling.BILINEAR)
        
        # Save temp scale file for reference (to be deleted later)
        temp_scale_path = os.path.join(out_dir, f"temp_scale_{s}.jpg")
        scaled_img.save(temp_scale_path)
        temp_scaled_files.append(temp_scale_path)
        
        p_map = sliding_window_inference(model, scaled_img, batch_size=batch_size)
        
        # Save Heatmap (Channel 0 = Text probability)
        heatmap = (p_map[:, :, 0] * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(out_dir, f"heatmap_scale_{s}.jpg"), heatmap_color)
        
        scale_polys = extract_polygons_from_map(p_map)
        for poly in scale_polys:
            pts = np.array(poly['points'])
            poly['points'] = (pts / s).tolist()
            aggregated_results.append(poly)

    # Deduplication
    print(f"Spatial Deduplication (R-Tree)...")
    final_results = spatial_nms_rtree(aggregated_results)

    # 3. Create Final Overlay Map
    print("Generating Overlay Map...")
    overlay_img = np.array(p_img)
    for poly in final_results:
        pts = np.array(poly['points'], np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay_img, [pts], True, (0, 255, 0), 2)
    
    cv2.imwrite(os.path.join(out_dir, "final_overlay.jpg"), cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

    # Save JSON
    results_path = os.path.join(out_dir, "detection_results.json")
    with open(results_path, "w") as f:
        json.dump(to_geojson(final_results), f, indent=2)
        
    # Cleanup: Delete temp scale files
    for f in temp_scaled_files:
        if os.path.exists(f): os.remove(f)
    
    print(f"Done! Cleaned up {len(temp_scaled_files)} temp scale files.")

# --- UTILITIES ---
def extract_polygons_from_map(prob_map, threshold=0.3):
    # 1. Generate Masks
    text_mask = (prob_map[:, :, 0] > threshold).astype(np.uint8)
    core_mask = (prob_map[:, :, 1] > threshold).astype(np.uint8)
    
    # --- NEW: Morphological Closing to bridge gaps ---
    # This connects nearby components in the core mask before labelling.
    # A (7, 7) kernel is usually sufficient to close gaps like the one in your image.
    # If gaps persist, increase this to (9, 9) or (11, 11).
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    core_mask = cv2.morphologyEx(core_mask, cv2.MORPH_CLOSE, kernel)
    # -------------------------------------------------

    # 2. Watershed Segmentation
    # Now that the core is connected, ndi.label will assign a single ID to the object
    markers, _ = ndi.label(core_mask)
    labels = watershed(-prob_map[:, :, 0], markers, mask=text_mask)
    objs = ndi.find_objects(labels)
    
    polys = []
    
    for i, loc in enumerate(objs):
        if loc is None: continue
        
        # Extract coordinate bounds with padding
        y1, y2 = loc[0].start, loc[0].stop
        x1, x2 = loc[1].start, loc[1].stop
        
        # Extract binary mask for this specific object instance
        roi_bool = (labels[max(0, y1-2):y2+2, max(0, x1-2):x2+2] == (i+1))
        
        # Fill holes to ensure we get a solid shape
        roi_filled = ndi.binary_fill_holes(roi_bool).astype(np.uint8) * 255
        
        # Find Contours
        cnts, _ = cv2.findContours(roi_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in cnts:
            # Shift contour coordinates back to global image space
            c[:, :, 0] += max(0, x1-2)
            c[:, :, 1] += max(0, y1-2)
            
            # Simplify polygon (epsilon 0.5% of arc length)
            approx = cv2.approxPolyDP(c, 0.005 * cv2.arcLength(c, True), True)
            
            # Ensure we have a valid polygon (at least 3 points)
            if len(approx) >= 3: 
                polys.append({"points": approx.reshape(-1, 2).tolist()})
                
    return polys

if __name__ == "__main__":
    MODEL_P = "data/Input/Model/Detection/Resnet_15-01/resnet_text_model.h5"
    INPUT_P = "data/Input/1890/e30067_0001_c.jpg"
    OUT_DIR = "data/Output/1890_Inference"
    Road_Scale_Data = "data/Output/1890_10-02-26/Predictions/npy_raw/e30067_0001_c_metadata.npy"
    run_inference(MODEL_P, INPUT_P, OUT_DIR, Road_Scale_Data, batch_size=128)