import os
import json
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tqdm import tqdm  # <--- NEW: Loading Bar
from config import *

# --- CONFIG ---
INFERENCE_BATCH_SIZE = 64
STRIDE = 224
CONFIDENCE_THRESHOLD = 0.4
MATCHING_RADIUS = 200

def load_ground_truth(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    gt_centers = []
    gt_polygons = []
    for ann in data.get('annotations', []):
        if len(ann['points']) < 3: continue
        pts = np.array(ann['points'])
        M = cv2.moments(pts.astype(np.float32))
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            gt_centers.append([cx, cy])
            gt_polygons.append(pts.astype(np.int32))
    return np.array(gt_centers), gt_polygons

def find_nearest_polygon(center, gt_polygons):
    cx, cy = center
    best_poly = None
    min_dist = float('inf')
    
    for poly in gt_polygons:
        if cv2.pointPolygonTest(poly, (float(cx), float(cy)), True) >= -20:
            return poly
        
        M = cv2.moments(poly)
        if M["m00"] > 0:
            pcx, pcy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            dist = np.sqrt((cx-pcx)**2 + (cy-pcy)**2)
            if dist < min_dist:
                min_dist = dist
                best_poly = poly
                
    if min_dist > MATCHING_RADIUS: return None
    return best_poly

def debug_stage_1(model, image_path, gt_centers):
    print("DEBUG: Running Stage 1 (Scanning Map)...")
    t0 = time.time()
    
    full_image = Image.open(image_path).convert('RGB')
    img_array = np.array(full_image).astype(np.float32) / 255.0
    h, w, c = img_array.shape
    
    # Pad
    pad_h = (PATCH_SIZE - h % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - w % PATCH_SIZE) % PATCH_SIZE
    img_padded = np.pad(img_array, ((0, pad_h), (0, pad_w), (0, 0)))
    final_h, final_w = img_padded.shape[:2]
    
    heatmap_canvas = np.zeros((final_h, final_w), dtype=np.float32)
    
    # 1. Pre-calculate all coordinates
    all_coords = []
    for y in range(0, final_h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, final_w - PATCH_SIZE + 1, STRIDE):
            all_coords.append((y, x))
            
    # 2. Process in Batches with Progress Bar
    # split into chunks of INFERENCE_BATCH_SIZE
    for i in tqdm(range(0, len(all_coords), INFERENCE_BATCH_SIZE), desc="Scanning Tiles"):
        batch_coords = all_coords[i : i + INFERENCE_BATCH_SIZE]
        batch_images = []
        
        for (y, x) in batch_coords:
            patch = img_padded[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            batch_images.append(patch)
            
        if len(batch_images) > 0:
            preds = model.predict(np.array(batch_images), verbose=0)
            
            # Place back on canvas
            for j, (py, px) in enumerate(batch_coords):
                heatmap_canvas[py:py+PATCH_SIZE, px:px+PATCH_SIZE] = np.maximum(
                    heatmap_canvas[py:py+PATCH_SIZE, px:px+PATCH_SIZE], 
                    preds[j, ..., 0]
                )
            
    # Crop & Threshold
    heatmap_final = heatmap_canvas[:h, :w]
    binary = (heatmap_final > CONFIDENCE_THRESHOLD).astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    
    centers_found = []
    for i in range(1, num):
        cx, cy = centroids[i]
        centers_found.append([int(cx), int(cy)])
        
    print(f"Stage 1 Done in {time.time()-t0:.2f}s. Found {len(centers_found)} centers.")
    return centers_found, full_image

def debug_stage_2(model, full_image, detected_centers, gt_polygons):
    print("DEBUG: Running Stage 2 (Sampled)...")
    
    if not detected_centers: return
    
    # Random Sample 16
    indices = np.arange(len(detected_centers))
    np.random.shuffle(indices)
    sample_indices = indices[:16]
    
    batch_imgs = []
    batch_indices = []
    
    for idx in sample_indices:
        cx, cy = detected_centers[idx]
        x1, y1 = int(cx - 64), int(cy - 64)
        patch = full_image.crop((x1, y1, x1+128, y1+128))
        if patch.size != (128, 128): patch = patch.resize((128, 128))
        batch_imgs.append(np.array(patch).astype(np.float32) / 255.0)
        batch_indices.append(idx)
        
    if batch_imgs:
        preds = model.predict(np.array(batch_imgs), verbose=0)
        
        plt.figure(figsize=(20, 20))
        plt.suptitle(f"Stage 2 Verification (Green=Pred, Blue=Truth)", fontsize=16)
        
        for k, rays in enumerate(preds):
            idx = batch_indices[k]
            cx, cy = detected_centers[idx]
            x1, y1 = int(cx - 64), int(cy - 64)
            
            # --- PREDICTION RECONSTRUCTION ---
            real_rays = rays * 64.0
            pred_points = []
            angles = np.linspace(0, 360, NUM_RAYS, endpoint=False)
            
            for angle, dist in zip(angles, real_rays):
                if dist < 2: continue # Ignore near-zero noise
                rad = np.deg2rad(angle)
                px = 64 + dist * np.cos(rad)
                py = 64 + dist * np.sin(rad)
                pred_points.append([int(px), int(py)])
            
            pred_points = np.array(pred_points, dtype=np.int32)
            
            # --- SMOOTHING FIX: CONVEX HULL ---
            # Instead of drawing the spiky raw points, we wrap them in a hull.
            if len(pred_points) >= 3:
                smooth_poly = cv2.convexHull(pred_points)
            else:
                smooth_poly = pred_points
            
            # --- GT MATCHING ---
            gt_poly_global = find_nearest_polygon((cx, cy), gt_polygons)
            gt_poly_local = None
            title = "No Match"
            
            vis_patch = (batch_imgs[k] * 255).astype(np.uint8)
            
            # Draw Smooth Prediction (Green)
            if len(smooth_poly) > 0:
                cv2.drawContours(vis_patch, [smooth_poly], -1, (0, 255, 0), 2)
            
            if gt_poly_global is not None:
                title = "GT Match"
                gt_poly_local = gt_poly_global.copy()
                gt_poly_local[:, 0] -= x1
                gt_poly_local[:, 1] -= y1
                cv2.drawContours(vis_patch, [gt_poly_local], -1, (0, 0, 255), 2)
                
            plt.subplot(4, 4, k+1)
            plt.imshow(vis_patch)
            plt.title(title)
            plt.axis('off')
            
        plt.savefig("debug_stage2.jpg")
        plt.show()

if __name__ == "__main__":
    # PATHS
    TEST_IMAGE = "data/Input/1877/nla.obj-231444908.tif"
    TEST_JSON = "data/Training Sets/Detection/ResNet_Box/1877_Text_Labels.json"
    
    path_1 = os.path.join(CHECKPOINT_DIR, "stage1_center_best.keras")
    model_1 = tf.keras.models.load_model(path_1, compile=False)
    
    path_2 = os.path.join(CHECKPOINT_DIR, "stage2_poly_best.keras")
    model_2 = tf.keras.models.load_model(path_2, compile=False)
    
    gt_centers, gt_polys = load_ground_truth(TEST_JSON)
    centers, img = debug_stage_1(model_1, TEST_IMAGE, gt_centers)
    debug_stage_2(model_2, img, centers, gt_polys)