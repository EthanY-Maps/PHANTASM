import os
import cv2
import json
import random
import numpy as np
from tqdm import tqdm
from pdf2image import convert_from_path
from PIL import Image
from training_utils import generate_db_targets
from config import *

# --- 1. CONFIGURATION ---
CACHE_DIR = 'data/Training Sets/Detection/dbnet_cache_256'
INPUT_SIZE = 256
OVERLAP_FACTOR = 5  # Adjust this: 15 is dense overlapping, 5 is sparse
VAL_SPLIT = 0.2      # 10% of generated crops go to validation

DATA_SOURCES = [
    (
        "data/Input/1877/nla.obj-231444908.tif",
        "data/Input/1877/1877_Text_Labels.json"
    ),
    (
        "data/Input/1889 Strathfield/Strathfield 1889-1894.jpg", 
        "data/Input/1889 Strathfield/Strathfield 1889-1894_Text.json"
    ),
    (
        "data/Input/Drummoyne/1083255_1193-13-Drummoyne.pdf", 
        "data/Input/Drummoyne/1083255_1193-13-Drummoyne_Text.json"
    ),
    (
        "data/Input/Endfield/1083257_1193-15-Enfield.tif",
        "data/Input/Endfield/1083257_1193-15-Enfield.json"
    ),
    (
        "data/Input/St Peters/1083312_1193-38-St Peters.pdf",
        "data/Input/St Peters/1083312_1193-38-St Peters.json"
    ),
]

# --- 2. SMART ANALYZER & GENERATOR ---
class SmartDBGenerator:
    def __init__(self, json_path, img_path, input_size=256):
        self.input_size = input_size
        
        # Clean paths
        json_path = json_path.replace("'", "").replace('"', "")
        img_path = img_path.replace("'", "").replace('"', "")
        
        # A. Load Image (PDF or Image)
        print(f"\nInitializing {os.path.basename(img_path)}...")
        if img_path.lower().endswith('.pdf'):
            print("  -> Converting PDF to Image...")
            images = convert_from_path(img_path, dpi=200) # 200 DPI is usually good balance
            self.full_image = images[0].convert('RGB')
        else:
            Image.MAX_IMAGE_PIXELS = None 
            self.full_image = Image.open(img_path).convert('RGB')
            
        orig_w, orig_h = self.full_image.size

        # B. Load Annotations
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        raw_anns = data.get('annotations', []) or data.get('shapes', [])
        
        # Extract Box Sizes for Analysis
        box_dims = []
        valid_polys_raw = []
        
        for ann in raw_anns:
            points = ann.get('points', [])
            if not points: continue
            pts = np.array(points, dtype=np.float32)
            
            # Bounding box dimensions
            w = np.max(pts[:,0]) - np.min(pts[:,0])
            h = np.max(pts[:,1]) - np.min(pts[:,1])
            box_dims.append((w, h))
            valid_polys_raw.append(pts)
            
        # C. SMART SCALING LOGIC
        self.scale_factor = 1.0
        
        if len(box_dims) > 0:
            box_dims = np.array(box_dims)
            # Check max dimension of each box
            max_dims = np.max(box_dims, axis=1) # Longest side of each box
            
            # We want the text to comfortably fit in the crop
            THRESHOLD_SIZE = input_size * 0.6 
            
            # How many boxes are "Too Big"?
            oversized_count = np.sum(max_dims > THRESHOLD_SIZE)
            oversized_ratio = oversized_count / len(max_dims)
            
            print(f"  -> Text Analysis: {oversized_count}/{len(max_dims)} boxes are larger than {THRESHOLD_SIZE}px")
            
            if oversized_ratio > 0.3: # If >30% of text is huge
                # Find the median size of the huge text
                huge_dims = max_dims[max_dims > THRESHOLD_SIZE]
                median_huge = np.median(huge_dims)
                
                # Target size: We want that median huge text to be roughly 80px
                target_size = 80.0
                self.scale_factor = target_size / median_huge
                
                # Hard limit: Don't scale below 10% (0.1)
                self.scale_factor = max(self.scale_factor, 0.1)
                
                print(f"  -> Majority Too Big. Calculated Auto-Scale: {self.scale_factor:.3f}")
            else:
                print(f"  -> Majority fits. Keeping Scale: 1.0")
        
        # D. Apply Scale
        if self.scale_factor != 1.0:
            new_w = int(orig_w * self.scale_factor)
            new_h = int(orig_h * self.scale_factor)
            print(f"  -> Resizing Map: {orig_w}x{orig_h} -> {new_w}x{new_h}")
            self.full_image = self.full_image.resize((new_w, new_h), Image.BILINEAR)
            
            # Scale Polygons
            self.polys = [p * self.scale_factor for p in valid_polys_raw]
        else:
            self.polys = valid_polys_raw

    def get_sample(self):
        """
        Gets a single crop (X) and targets (y).
        Uses 'Smart Cropping' to focus on text regions 70% of the time.
        """
        w, h = self.full_image.size
        
        # --- 1. Determine Crop Coordinates ---
        if len(self.polys) > 0 and random.random() < 0.7:
            # Pick a random polygon
            target_poly = random.choice(self.polys)
            
            # Get bounds
            min_x, min_y = np.min(target_poly, axis=0)
            max_x, max_y = np.max(target_poly, axis=0)
            
            # Pick a random point INSIDE or NEAR that box
            center_x = random.uniform(min_x, max_x)
            center_y = random.uniform(min_y, max_y)
            
            # Calculate top-left of crop with random jitter
            jitter = self.input_size // 3
            crop_x = int(center_x - (self.input_size / 2) + random.randint(-jitter, jitter))
            crop_y = int(center_y - (self.input_size / 2) + random.randint(-jitter, jitter))
            
        else:
            # Random Background
            crop_x = random.randint(0, w - self.input_size)
            crop_y = random.randint(0, h - self.input_size)

        # Clamp to image bounds
        crop_x = max(0, min(crop_x, w - self.input_size))
        crop_y = max(0, min(crop_y, h - self.input_size))
        
        # --- 2. Perform Crop ---
        patch = self.full_image.crop((crop_x, crop_y, crop_x+self.input_size, crop_y+self.input_size))
        img_np = np.array(patch)
        
        # --- 3. Filter Polygons ---
        valid_polys_crop = []
        for poly in self.polys:
            poly_shifted = poly.copy()
            poly_shifted[:, 0] -= crop_x
            poly_shifted[:, 1] -= crop_y
            
            p_min = np.min(poly_shifted, axis=0)
            p_max = np.max(poly_shifted, axis=0)
            
            if (p_max[0] >= 0 and p_min[0] < self.input_size and 
                p_max[1] >= 0 and p_min[1] < self.input_size):
                valid_polys_crop.append(poly_shifted)

        # --- 4. Generate Targets ---
        X = img_np.astype(np.float32) / 255.0
        prob, thresh, mask = generate_db_targets((self.input_size, self.input_size), valid_polys_crop)
        y = np.concatenate([prob, thresh, mask], axis=-1)
        
        return X, y

# --- 3. SMART AREA-BASED CACHING LOOP ---
def process_map(img_path, json_path, overlap_factor=15, val_split=0.1):
    """
    Dynamically calculates how many crops a map needs based on its physical size.
    """
    map_name = os.path.splitext(os.path.basename(img_path))[0]
    
    # 1. Initialize Generator (Only happens ONCE per map now)
    try:
        gen = SmartDBGenerator(json_path, img_path, input_size=INPUT_SIZE)
    except Exception as e:
        print(f"Skipping {map_name}: {e}")
        return

    # 2. Calculate Area and Required Crops
    w, h = gen.full_image.size
    map_area = w * h
    crop_area = INPUT_SIZE * INPUT_SIZE
    
    # Base number of crops to cover the map exactly once
    base_crops = map_area / crop_area
    
    # Total samples to generate (with dense overlap for data augmentation)
    total_samples = int(base_crops * overlap_factor)
    total_samples = max(total_samples, 100) # Safety net for tiny images
    
    train_samples = int(total_samples * (1.0 - val_split))
    val_samples = total_samples - train_samples
    
    print(f"\n[{map_name}] Dimensions: {w}x{h} | Generating {total_samples} total crops ({train_samples} Train, {val_samples} Val)")

    # 3. Helper to save the splits
    def save_split(num_samples, split_name):
        save_path = os.path.join(CACHE_DIR, split_name)
        os.makedirs(save_path, exist_ok=True)
        
        success_count = 0
        pbar = tqdm(range(num_samples), desc=f"Caching {map_name} ({split_name})")
        
        for i in pbar:
            try:
                X, y = gen.get_sample()
                filename = f"{map_name}_{i}.npz"
                np.savez_compressed(os.path.join(save_path, filename), X=X, y=y)
                success_count += 1
            except Exception as e:
                pass
                
    # 4. Execute Caching
    save_split(train_samples, 'train')
    save_split(val_samples, 'val')
    
    # 5. Save Metadata
    meta_path = os.path.join(CACHE_DIR, f"{map_name}_metadata.txt")
    with open(meta_path, "w") as f:
        f.write(f"scale_used: {gen.scale_factor}\n")
        f.write(f"original_size: {w}x{h}\n")
        f.write(f"overlap_factor: {overlap_factor}\n")
        f.write(f"train_crops: {train_samples}\n")
        f.write(f"val_crops: {val_samples}\n")

if __name__ == "__main__":
    if not DATA_SOURCES:
        print("Error: DATA_SOURCES empty.")
        exit()
    
    print(f"Starting Smart Area-Based Caching to {CACHE_DIR}...")
    
    for img_p, json_p in DATA_SOURCES:
        clean_img = img_p.replace("'", "").replace('"', "")
        if not os.path.exists(clean_img):
            print(f"MISSING FILE: {clean_img}")
            continue
            
        # Execute the process map function
        process_map(img_p, json_p, overlap_factor=OVERLAP_FACTOR, val_split=VAL_SPLIT)
        
    print("\nSmart Caching Complete. Ready for TFRecord conversion!")