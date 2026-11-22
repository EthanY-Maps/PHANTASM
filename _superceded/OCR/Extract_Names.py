import os
import cv2
import json
import math
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.validation import make_valid
from PIL import Image
from pdf2image import convert_from_path
from tqdm import tqdm
import warnings

# --- CONFIGURATION ---
warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()

# ==========================================
# 1. HELPER FUNCTIONS (UPDATED)
# ==========================================

def load_source_image(image_path, target_dpi=300):
    """Loads an image or converts PDF to OpenCV BGR image."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext == '.pdf':
        print(f"📄 Converting PDF: {image_path}")
        pages = convert_from_path(image_path, dpi=target_dpi, grayscale=False)
        return cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
    else:
        print(f"🖼️ Loading Image: {image_path}")
        pil_img = Image.open(image_path).convert('RGB')
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def simplify_polygon(poly, max_points=12, tolerance_step=1.0):
    """Simplifies polygon to remove jitter/curves."""
    if poly.is_empty: return poly
    current_poly = poly
    tolerance = tolerance_step
    for _ in range(10):
        if len(current_poly.exterior.coords) <= max_points: break
        current_poly = poly.simplify(tolerance, preserve_topology=True)
        tolerance += tolerance_step
    return current_poly

def get_pca_orientation(poly):
    """
    Calculates the primary and secondary direction vectors using PCA 
    and determines the smallest rotation angle to make the primary vector horizontal.
    """
    if poly.is_empty: return 0.0

    # 1. Extract coordinates (Remove the last point as it duplicates the first)
    pts = np.array(poly.exterior.coords)[:-1]
    if len(pts) < 2: return 0.0

    # 2. Perform PCA (Principal Component Analysis)
    # Center the data
    mean = np.mean(pts, axis=0)
    centered_data = pts - mean
    
    # Compute Covariance Matrix and Eigenvectors
    cov = np.cov(centered_data.T)
    evals, evecs = np.linalg.eig(cov)

    # 3. Sort eigenvectors by eigenvalues (Largest is Primary)
    sort_indices = np.argsort(evals)[::-1]
    primary_vec = evecs[:, sort_indices[0]]  # The direction of greatest spread
    
    # 4. Calculate Angle of Primary Vector
    # arctan2 returns angle in radians between -pi and pi
    angle_rad = np.arctan2(primary_vec[1], primary_vec[0])
    angle_deg = np.degrees(angle_rad)

    # 5. Smallest Rotation Logic
    # We want the text horizontal (0 degrees). 
    # The PCA vector has no "direction" (left vs right), just an axis.
    # If the vector points "left" (e.g., 170 deg), rotating -170 flips the image upside down?
    # No, we just want to align the LINE to the X-axis.
    
    # Normalize to [-90, 90] range to ensure "smallest rotation"
    # This prevents 180-degree flips (upside down text) if the vector happened to point left.
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180

    return angle_deg

def rotate_image_expanded(image, angle):
    """Rotates image without cropping corners."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    rad = math.radians(angle)
    sin = math.sin(rad)
    cos = math.cos(rad)

    n_w = int((h * abs(sin)) + (w * abs(cos)))
    n_h = int((h * abs(cos)) + (w * abs(sin)))

    # OpenCV rotate requires the negative of the angle for standard Cartesian rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0) 
    M[0, 2] += (n_w / 2) - center[0]
    M[1, 2] += (n_h / 2) - center[1]

    return cv2.warpAffine(image, M, (n_w, n_h), 
                         flags=cv2.INTER_CUBIC, 
                         borderMode=cv2.BORDER_CONSTANT, 
                         borderValue=(0, 0, 0))

def autocrop_black_borders(image, margin=2):
    """Trims black borders after rotation."""
    if image is None or image.size == 0: return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    if coords is None: return image
    
    x, y, w, h = cv2.boundingRect(coords)
    h_img, w_img = image.shape[:2]
    
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(w_img - x, w + 2*margin)
    h = min(h_img - y, h + 2*margin)
    
    return image[y:y+h, x:x+w]
# ==========================================
# 2. MAIN PIPELINE
# ==========================================

def extract_snippets_from_json(image_path, json_path, output_dir, buffer_px=10):
    os.makedirs(output_dir, exist_ok=True)
    
    image = load_source_image(image_path)
    h_img, w_img = image.shape[:2]
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    polygons = []
    metadata = []
    
    print("📐 Parsing polygons...")
    for entry in data.get('features', []):
        try:
            coords = entry['geometry']['coordinates'][0]
            poly = Polygon(coords)
            if not poly.is_valid: poly = make_valid(poly)
            polygons.append(poly)
            metadata.append(entry.get('properties', {}))
        except:
            continue

    print(f"✂️ Processing {len(polygons)} items...")
    manifest = []

    for idx, raw_poly in enumerate(tqdm(polygons)):
        if raw_poly.is_empty: continue

        # A. SIMPLIFY (Clean shape for angle calculation)
        simple_poly = simplify_polygon(raw_poly, max_points=10)

        # B. BUFFER (Capture context)
        buffered_poly = simple_poly.buffer(buffer_px)
        
        # C. CROP
        minx, miny, maxx, maxy = buffered_poly.bounds
        minx, miny = max(int(minx), 0), max(int(miny), 0)
        maxx, maxy = min(int(maxx), w_img), min(int(maxy), h_img)
        
        if minx >= maxx or miny >= maxy: continue
        crop = image[miny:maxy, minx:maxx].copy()
        if crop.size == 0: continue

        # D. MASK
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        parts = [buffered_poly] if buffered_poly.geom_type == 'Polygon' else list(buffered_poly.geoms)
        
        for part in parts:
            if part.is_empty: continue
            local_coords = np.array(part.exterior.coords) - [minx, miny]
            cv2.fillPoly(mask, [local_coords.astype(np.int32)], 255)

        masked_crop = cv2.bitwise_and(crop, crop, mask=mask)

        # E. ROTATE (Corrected Logic)
        rot_angle = get_pca_orientation(simple_poly)
        rotated_img = rotate_image_expanded(masked_crop, rot_angle)
        final_img = autocrop_black_borders(rotated_img)

        # F. SAVE
        if final_img is not None and final_img.size > 0:
            filename = f"snippet_{idx:05d}.png"
            cv2.imwrite(os.path.join(output_dir, filename), final_img)
            
            manifest.append({
                "filename": filename,
                "original_idx": idx,
                "rotation": rot_angle,
                **metadata[idx]
            })

    pd.DataFrame(manifest).to_csv(os.path.join(output_dir, "manifest.csv"), index=False)
    print("✅ Done.")

# ==========================================
# 3. RUN
# ==========================================
if __name__ == "__main__":
    # UPDATE PATHS
    INPUT_MAP = "data/Input/1889 Strathfield/1083313_1193-39-Strathfield.pdf"
    INPUT_JSON = "data/Output/Strathefield_Testing/1083313_1193-39-Strathfield_predictions.json"
    OUTPUT_DIR = "data/Output/Strathefield_Testing/OCR/clean_snippets"
    
    extract_snippets_from_json(INPUT_MAP, INPUT_JSON, OUTPUT_DIR)