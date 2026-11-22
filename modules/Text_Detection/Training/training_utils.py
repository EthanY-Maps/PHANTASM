# training_utils.py
import math
import json
import random
import numpy as np
import cv2
import io
from PIL import Image, ImageEnhance, ImageFilter,ImageOps
import tensorflow as tf
import tensorflow.keras.backend as K
from config import *


def apply_heavy_augmentations(img_pil, stage=1):
    """
    The 'Scorched Earth' Augmentation Suite.
    Destroys domain-specific features (color, texture) to force shape learning.
    """
    # Convert to CV2 format for geometric transforms
    img_np = np.array(img_pil)

    # ==================================================
    # 1. GEOMETRIC CHAOS (The "Wrinkled Map" Effect)
    # ==================================================
    
    # Elastic Transform (Simulate water damage / warping)
    if random.random() < 0.4:
        alpha = random.uniform(30, 60) # Intensity
        sigma = random.uniform(4, 6)   # Smoothness
        
        shape = img_np.shape
        dx = cv2.GaussianBlur((np.random.rand(*shape[:2]) * 2 - 1), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(*shape[:2]) * 2 - 1), (0, 0), sigma) * alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x, map_y = (x + dx).astype('float32'), (y + dy).astype('float32')
        
        # Remap image
        img_np = cv2.remap(img_np, map_x, map_y, interpolation=cv2.INTER_BILINEAR, borderMode=cv2.BORDER_REFLECT)

    # ==================================================
    # 2. DOMAIN DESTRUCTION (The "Bad Scanner" Effect)
    # ==================================================
    
    # Convert back to PIL for color ops
    img_pil = Image.fromarray(img_np)

    # A. Color Channel Shuffle (The most important one for blind maps)
    # Text is dark, background is light. It shouldn't matter if paper is green or pink.
    if random.random() < 0.5:
        r, g, b = img_pil.split()
        channels = [r, g, b]
        random.shuffle(channels)
        img_pil = Image.merge('RGB', channels)

    # B. Solarize (Invert high values) - partial inversion
    if random.random() < 0.2:
        img_pil = ImageOps.solarize(img_pil, threshold=128)

    # C. Grayscale (Force shape learning)
    if random.random() < 0.3:
        img_pil = ImageOps.grayscale(img_pil).convert('RGB')
        
    # D. Invert (Negative photostat style)
    if random.random() < 0.1:
        img_pil = ImageOps.invert(img_pil)

    # ==================================================
    # 3. QUALITY DEGRADATION (The "Low-Res" Effect)
    # ==================================================

    # A. JPEG Compression Artifacts
    # We save to a buffer with low quality and read back
    if random.random() < 0.4:
        import io
        buffer = io.BytesIO()
        # Random quality between 10 (terrible) and 80 (okay)
        q = random.randint(10, 70)
        img_pil.save(buffer, format='JPEG', quality=q)
        buffer.seek(0)
        img_pil = Image.open(buffer)

    # B. Downscale-Upscale (Simulate low DPI scan)
    if random.random() < 0.4:
        w, h = img_pil.size
        factor = random.uniform(0.3, 0.7)
        small_size = (int(w * factor), int(h * factor))
        img_pil = img_pil.resize(small_size, resample=Image.BILINEAR)
        img_pil = img_pil.resize((w, h), resample=Image.NEAREST) # Nearest neighbor adds "pixelation"

    # ==================================================
    # 4. STANDARD PHOTOMETRIC (Lighting)
    # ==================================================
    
    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    # Great for normalizing lighting variations across the map
    img_np = np.array(img_pil)
    if random.random() < 0.3:
        # Convert to LAB color space
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        img_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # Final standard jitter
    if random.random() < 0.5:
        # Brightness / Contrast
        alpha = random.uniform(0.5, 1.5) # Contrast control
        beta = random.uniform(-30, 30)   # Brightness control
        img_np = cv2.convertScaleAbs(img_np, alpha=alpha, beta=beta)
        
    return img_np

# ==========================================
# 2. MATH & HELPERS
# ==========================================

def calculate_ray_distances_math(contour, center, num_rays, max_dist):
    cx, cy = center
    distances = np.zeros(num_rays, dtype=np.float32)
    angles = np.linspace(0, 360, num_rays, endpoint=False)
    for i, angle in enumerate(angles):
        angle_rad = np.deg2rad(angle)
        unit_dx, unit_dy = np.cos(angle_rad), np.sin(angle_rad)
        low, high, dist = 0.0, float(max_dist), 0.0
        for _ in range(8): 
            mid = (low + high) / 2
            px, py = cx + unit_dx * mid, cy + unit_dy * mid
            if cv2.pointPolygonTest(contour, (float(px), float(py)), False) >= 0:
                dist, low = mid, mid
            else: high = mid
        distances[i] = dist
    return distances

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    sigma = diameter / 6
    x, y = int(center[0]), int(center[1])
    h, w = heatmap.shape[0:2]
    dx = np.arange(-radius, radius + 1)
    dy = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(dx, dy)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    left, right = max(0, x - radius), min(w, x + radius + 1)
    top, bottom = max(0, y - radius), min(h, y + radius + 1)
    k_left, k_right = radius - (x - left), radius + (right - x)
    k_top, k_bottom = radius - (y - top), radius + (bottom - y)
    masked_heatmap = heatmap[top:bottom, left:right]
    masked_kernel = kernel[k_top:k_bottom, k_left:k_right]
    np.maximum(masked_heatmap, masked_kernel * k, out=masked_heatmap)
    return heatmap

# ==========================================
# 3. LOSS FUNCTIONS
# ==========================================

def f1_metric(y_true, y_pred):
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    y_pred_binary = K.cast(K.greater(y_pred, 0.3), 'float32')
    y_true_binary = K.cast(K.greater(y_true, 0.5), 'float32')
    tp = K.sum(y_true_binary * y_pred_binary)
    return 2 * ((tp / (K.sum(y_pred_binary) + K.epsilon()) * tp / (K.sum(y_true_binary) + K.epsilon())) / (tp / (K.sum(y_pred_binary) + K.epsilon()) + tp / (K.sum(y_true_binary) + K.epsilon()) + K.epsilon()))

def hybrid_blob_loss(y_true, y_pred):
    """
    Combines Weighted Focal Loss with Soft Dice Loss.
    Focal handles the sparsity; Dice handles the shape/blob accuracy.
    """
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    
    # 1. Weighted Focal Loss (The 75x Hammer)
    pos_mask = tf.cast(tf.equal(y_true, 1.0), tf.float32)
    neg_mask = tf.cast(tf.less(y_true, 1.0), tf.float32)
    
    pos_loss = -K.log(y_pred) * K.pow(1 - y_pred, 2) * pos_mask * 75.0
    neg_loss = -K.log(1 - y_pred) * K.pow(y_pred, 2) * neg_mask * K.pow(1 - y_true, 4)
    focal = tf.math.divide_no_nan(K.sum(pos_loss) + K.sum(neg_loss), K.sum(pos_mask) + 1e-4)
    
    # 2. Soft Dice Loss (Forces blobs to overlap perfectly)
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(K.square(y_true) + K.square(y_pred))
    dice = 1 - (2. * intersection + 1e-4) / (sum_ + 1e-4)
    
    return focal + (dice * 10.0) # Dice is scaled to match focal magnitude

def vector_iou_loss(y_true, y_pred):
    inter = tf.reduce_sum(tf.square(tf.minimum(y_true, y_pred)), axis=-1)
    union = tf.reduce_sum(tf.square(y_true), axis=-1) + tf.reduce_sum(tf.square(y_pred), axis=-1) - inter
    return 1.0 - tf.reduce_mean((inter + 1e-6) / (union + 1e-6))

# ==========================================
# 4. GENERATORS
# ==========================================

class CenterGenerator(tf.keras.utils.Sequence):
    def __init__(self, json_path, image_path, usage='train', split_ratio=0.8):
        with open(json_path, 'r') as f: 
            data = json.load(f)
        
        # Filter for valid polygons
        self.all_anns = [a for a in data.get('annotations', []) if len(a['points']) >= 3]
        self.full_image = Image.open(image_path).convert('RGB')
        self.img_w, self.img_h = self.full_image.size
        self.usage = usage
        
        # Tiling logic
        stride = 128
        all_tiles = []
        for y in range(0, self.img_h - PATCH_SIZE + 1, stride):
            for x in range(0, self.img_w - PATCH_SIZE + 1, stride):
                tile_box = [x, y, x + PATCH_SIZE, y + PATCH_SIZE]
                # Check if this tile contains any text centers
                has_text = any([
                    tile_box[0] <= (min([p[0] for p in a['points']]) + max([p[0] for p in a['points']]))/2 < tile_box[2] and 
                    tile_box[1] <= (min([p[1] for p in a['points']]) + max([p[1] for p in a['points']]))/2 < tile_box[3] 
                    for a in self.all_anns
                ])
                all_tiles.append({'coords': (x, y), 'label': has_text})
        
        # Split into Positive and Negative pools
        pos, neg = [t for t in all_tiles if t['label']], [t for t in all_tiles if not t['label']]
        random.seed(42)
        random.shuffle(pos)
        random.shuffle(neg)
        
        split_p, split_n = int(len(pos)*split_ratio), int(len(neg)*split_ratio)
        
        if usage == 'train':
            self.pos_pool = pos[:split_p]
            self.neg_pool = neg[:split_n]
        else:
            self.pos_pool = pos[split_p:]
            self.neg_pool = neg[split_n:]
            
        self.n_batches = max(1, min(len(self.pos_pool), len(self.neg_pool)) * 2 // BATCH_SIZE)

    def __len__(self): 
        return self.n_batches

    def __getitem__(self, index):
        X = np.zeros((BATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 3), dtype=np.float32)
        y = np.zeros((BATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1), dtype=np.float32)
        
        for i in range(BATCH_SIZE):
            # Balanced sampling: half positive, half negative
            pool = self.pos_pool if i < BATCH_SIZE // 2 else self.neg_pool
            tile = random.choice(pool)
            tx, ty = tile['coords']
            
            patch = self.full_image.crop((tx, ty, tx+PATCH_SIZE, ty+PATCH_SIZE))
            
            # PHOTOMETRIC ONLY for Stage 1 to avoid shifting center coords
            img_aug = np.array(patch)
            if self.usage == 'train':
                # Photometric Jitter
                enhancer = ImageEnhance.Color(patch)
                patch = enhancer.enhance(random.uniform(0.5, 1.5))
                if random.random() < 0.3: 
                    patch = patch.filter(ImageFilter.GaussianBlur(1))
                img_aug = np.array(patch)
                # Add pixel noise
                if random.random() < 0.2:
                    noise = np.random.normal(0, 10, img_aug.shape).astype(np.uint8)
                    img_aug = cv2.add(img_aug, noise)

            X[i] = img_aug.astype(np.float32) / 255.0
            
            heatmap = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
            for a in self.all_anns:
                pts = np.array([[p[0]-tx, p[1]-ty] for p in a['points']], dtype=np.float32)
                if np.any((pts >= 0) & (pts < PATCH_SIZE)):
                    M = cv2.moments(pts)
                    if M["m00"] > 0:
                        cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                        if 0 <= cx < PATCH_SIZE and 0 <= cy < PATCH_SIZE:
                            # Using 8px radius as discussed for a bigger target
                            draw_gaussian(heatmap, (cx, cy), 8)
            y[i, ..., 0] = heatmap
        return X, y

class PolygonGenerator(tf.keras.utils.Sequence):
    def __init__(self, json_path, image_path, usage='train', split_ratio=0.8):
        with open(json_path, 'r') as f: data = json.load(f)
        self.all_anns = [a for a in data.get('annotations', []) if len(a['points']) >= 3]
        indices = np.arange(len(self.all_anns)); random.seed(42); random.shuffle(indices)
        split_idx = int(len(indices) * split_ratio)
        self.dataset_indices = indices[:split_idx] if usage == 'train' else indices[split_idx:]
        self.full_image = Image.open(image_path).convert('RGB')
        self.usage = usage
        self.net_input_size = INPUT_SHAPE_POLY 
        self.n_batches = max(1, len(self.dataset_indices) // BATCH_SIZE)

    def __len__(self): return self.n_batches
    def __getitem__(self, index):
        X = np.zeros((BATCH_SIZE, self.net_input_size, self.net_input_size, 3), dtype=np.float32)
        y = np.zeros((BATCH_SIZE, NUM_RAYS), dtype=np.float32)
        for i in range(BATCH_SIZE):
            idx = random.choice(self.dataset_indices)
            ann = self.all_anns[idx]
            pts = np.array(ann['points'])
            min_x, max_x, min_y, max_y = np.min(pts[:,0]), np.max(pts[:,0]), np.min(pts[:,1]), np.max(pts[:,1])
            cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
            
            # Crop Larger for Rotation Safety
            obj_size = max(max_x - min_x, max_y - min_y, 20)
            # Increased padding to 2.5 min to ensure rotation doesn't clip
            padding = random.uniform(2.5, 4.0) if self.usage == 'train' else 2.5 
            crop_w = int(obj_size * padding)
            tx, ty = int(cx - crop_w // 2), int(cy - crop_w // 2)
            
            patch = self.full_image.crop((tx, ty, tx+crop_w, ty+crop_w))
            patch = patch.resize((256, 256), Image.BILINEAR) # Resize to temp large buffer
            
            scale_temp = 256.0 / crop_w
            local_pts = np.array([[(p[0]-tx)*scale_temp, (p[1]-ty)*scale_temp] for p in pts], dtype=np.float32)
            
            if self.usage == 'train':
                img_aug, local_pts = apply_heavy_augmentations(patch, local_pts, stage=2)
            else:
                img_aug = np.array(patch)

            # Center Crop to final 128x128
            start = (256 - self.net_input_size) // 2
            final_img = img_aug[start:start+self.net_input_size, start:start+self.net_input_size]
            
            local_pts[:, 0] -= start
            local_pts[:, 1] -= start
            
            X[i] = final_img.astype(np.float32) / 255.0
            
            # Rays (FIXED NORMALIZATION)
            # Center of the crop is (64, 64)
            center_crop = (self.net_input_size // 2, self.net_input_size // 2)
            rays = calculate_ray_distances_math(local_pts.astype(np.int32), center_crop, NUM_RAYS, max_dist=MAX_RAY_DIST)
            
            # Normalize by MAX_RAY_DIST (128.0), not 64.0
            y[i] = rays / MAX_RAY_DIST
            
        return X, y


hybrid_blob_loss.__name__ = "hybrid_blob_loss"
f1_metric.__name__ = "f1_metric"