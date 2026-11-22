import math
import json
import random
import cv2
import numpy as np
import pyclipper
import os
from shapely.geometry import Polygon
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
# from config import * # Uncomment if you have a config file

# ==========================================
# 1. DBNet TARGET GENERATION
# ==========================================

def shrink_polygon_pyclipper(polygon, shrink_ratio=0.4):
    """
    Shrinks a polygon by 'shrink_ratio' using the Vatti clipping algorithm.
    Used to generate the 'Probability Map' (the core of the text).
    """
    poly = Polygon(polygon)
    area = poly.area
    perimeter = poly.length
    
    if perimeter == 0: 
        return []
    
    # Calculate offset (Vatti formula)
    offset = area * (1 - shrink_ratio ** 2) / perimeter
    
    # Use Pyclipper to shrink
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(polygon, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    
    # Negative offset means shrink
    shrunk = pco.Execute(-offset)
    return shrunk

def generate_db_targets(img_size, polygons, shrink_ratio=0.4):
    """
    Creates the 3 Ground Truth maps for DBNet:
    1. Probability Map (G): The text region, shrunk by 0.4.
    2. Threshold Map (T): The gradient border zone around the text.
    3. Training Mask: Ignores "Don't Care" regions (too small/illegible).
    """
    h, w = img_size
    prob_map = np.zeros((h, w), dtype=np.float32)
    thresh_map = np.zeros((h, w), dtype=np.float32)
    mask = np.ones((h, w), dtype=np.float32)
    
    for poly_points in polygons:
        poly_points = np.array(poly_points).astype(np.int32)
        poly_shapely = Polygon(poly_points)
        
        if poly_shapely.area <= 0 or poly_shapely.length <= 0:
            continue
            
        # 1. Draw Probability Map (The Shrunk Core)
        # -----------------------------------------
        shrunk_polys = shrink_polygon_pyclipper(poly_points, shrink_ratio)
        for shrunk in shrunk_polys:
            shrunk = np.array(shrunk).astype(np.int32)
            cv2.fillPoly(prob_map, [shrunk], 1.0)
            
        # 2. Draw Threshold Map (The Gradient Border Zone)
        # ---------------------------------------
        offset = poly_shapely.area * (1 - shrink_ratio ** 2) / poly_shapely.length
        
        # Expand the polygon to find the outer boundary of the threshold area
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(poly_points.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded_polys = pco.Execute(offset)
        
        for expanded in expanded_polys:
            expanded = np.array(expanded).astype(np.int32)
            
            # Create a local bounding box to speed up distance transform
            xmin, ymin = np.clip(np.min(expanded, axis=0), 0, [w-1, h-1])
            xmax, ymax = np.clip(np.max(expanded, axis=0), 0, [w-1, h-1])
            
            if xmax <= xmin or ymax <= ymin:
                continue
                
            # Create a canvas for the original un-shrunk polygon edges
            canvas = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
            local_poly = poly_points.copy()
            local_poly[:, 0] -= xmin
            local_poly[:, 1] -= ymin
            
            # Draw the edges of the original polygon
            cv2.polylines(canvas, [local_poly], isClosed=True, color=1, thickness=1)
            
            # Calculate distance from the edges
            # 1 - canvas inverses it so the edges are 0, allowing distanceTransform to work
            distance_map = cv2.distanceTransform(1 - canvas, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            
            # Convert distance to a gradient between 0 and 1
            # Pixels exactly on the edge are 1.0, fading to 0.0 as they reach the 'offset' distance
            gradient = 1.0 - np.clip(distance_map / offset, 0, 1.0)
            
            # Create a mask of the expanded polygon so we only apply the gradient inside it
            local_expanded = expanded.copy()
            local_expanded[:, 0] -= xmin
            local_expanded[:, 1] -= ymin
            expanded_mask = np.zeros_like(canvas)
            cv2.fillPoly(expanded_mask, [local_expanded], 1)
            
            gradient = gradient * expanded_mask
            
            # Scale the gradient to the 0.3 - 0.7 range required by DBNet
            gradient = gradient * 0.4 + 0.3
            
            # Update the global threshold map (keeping the maximum value if polygons overlap)
            thresh_map[ymin:ymax+1, xmin:xmax+1] = np.maximum(
                thresh_map[ymin:ymax+1, xmin:xmax+1], 
                gradient
            )
            
    return prob_map[..., None], thresh_map[..., None], mask[..., None]

# ==========================================
# 2. DBNet LOSS FUNCTION
# ==========================================

def balance_bce_loss(y_true, y_pred, mask, alpha=0.25, gamma=2.0):
    """
    Replaces OHEM sorting with purely vectorized Focal Loss.
    Massively reduces GPU memory movement and step time.
    """
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    focal_weight = tf.math.pow((1.0 - p_t), gamma)
    
    alpha_weight = y_true * alpha + (1 - y_true) * (1 - alpha)
    
    weighted_loss = bce * focal_weight * alpha_weight * mask
    
    return tf.reduce_sum(weighted_loss) / (tf.reduce_sum(mask) + 1e-5)

def db_loss(y_true, y_pred):
    """
    Standard DBNet Loss.
    Combines Probability map loss, Threshold map loss, and Binary map loss.
    """
    gt_prob = y_true[..., 0:1]
    gt_thresh = y_true[..., 1:2]
    gt_mask = y_true[..., 2:3]
    
    pred_prob = y_pred[..., 0:1]
    pred_thresh = y_pred[..., 1:2]
    pred_binary = y_pred[..., 2:3]
    
    # Loss for Probability Map
    l_prob = balance_bce_loss(gt_prob, pred_prob, gt_mask)
    
    # Loss for Threshold Map (L1 loss only applied where GT Prob is high, weighted by 10)
    l_thresh = tf.reduce_mean(tf.abs(pred_thresh - gt_thresh) * gt_prob * 10.0)
    
    # Loss for Approximate Binary Map (Dice Loss)
    intersection = tf.reduce_sum(pred_binary * gt_prob * gt_mask)
    union = tf.reduce_sum(pred_binary * gt_mask) + tf.reduce_sum(gt_prob * gt_mask) + 1e-6
    l_binary = 1 - (2. * intersection / union)
    
    return l_prob + 10.0 * l_thresh + 5.0 * l_binary

def l1_loss(y_true, y_pred, mask):
    """Simple L1 loss for the Threshold Map"""
    abs_diff = tf.abs(y_true - y_pred)
    return tf.reduce_sum(abs_diff * mask) / (tf.reduce_sum(mask) + 1e-6)

def dice_loss(y_true, y_pred, mask):
    """Dice Loss for the final Binary Map"""
    intersection = tf.reduce_sum(y_true * y_pred * mask)
    union = tf.reduce_sum(y_true * mask) + tf.reduce_sum(y_pred * mask) + 1e-6
    return 1 - (2. * intersection / union)

# ==========================================
# 3. GENERATORS
# ==========================================

class DBGenerator(tf.keras.utils.Sequence):
    def __init__(self, json_path, image_path, usage='train', split_ratio=0.8, input_size=640, batch_size=4, **kwargs):
        super().__init__(**kwargs)
        
        with open(json_path, 'r') as f: 
            data = json.load(f)
        
        self.all_anns = [a for a in data.get('annotations', []) if len(a['points']) >= 3]
        self.full_image = Image.open(image_path).convert('RGB')
        self.usage = usage
        self.input_size = input_size
        self.batch_size = batch_size
        self.n_batches = 100 

    def __len__(self): 
        return self.n_batches

    def __getitem__(self, index):
        X = np.zeros((self.batch_size, self.input_size, self.input_size, 3), dtype=np.float32)
        y = np.zeros((self.batch_size, self.input_size, self.input_size, 3), dtype=np.float32)
        
        for i in range(self.batch_size):
            w, h = self.full_image.size
            if w <= self.input_size or h <= self.input_size:
                x, y_coord = 0, 0
            else:
                if self.usage == 'train':
                    x = random.randint(0, w - self.input_size)
                    y_coord = random.randint(0, h - self.input_size)
                else:
                    x = (w - self.input_size) // 2
                    y_coord = (h - self.input_size) // 2
                
            patch = self.full_image.crop((x, y_coord, x+self.input_size, y_coord+self.input_size))
            img_np = np.array(patch)
            
            valid_polys = []
            for ann in self.all_anns:
                pts = np.array(ann['points']).astype(np.float32)
                pts[:, 0] -= x
                pts[:, 1] -= y_coord
                
                # Check if polygon is inside the cropped patch
                if np.min(pts[:, 0]) >= 0 and np.max(pts[:, 0]) < self.input_size and \
                   np.min(pts[:, 1]) >= 0 and np.max(pts[:, 1]) < self.input_size:
                    valid_polys.append(pts)
            
            prob, thresh, mask = generate_db_targets((self.input_size, self.input_size), valid_polys)
            
            X[i] = img_np.astype(np.float32) / 255.0
            y[i, ..., 0:1] = prob
            y[i, ..., 1:2] = thresh
            y[i, ..., 2:3] = mask
            
        return X, y

class CachedDBGenerator(tf.keras.utils.Sequence):
    def __init__(self, cache_dir, batch_size=4, shuffle=True):
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.files = [f for f in os.listdir(cache_dir) if f.endswith('.npz')]
        self.indexes = np.arange(len(self.files))
        
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        idxs = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
        batch_files = [self.files[k] for k in idxs]

        X, y = [], []
        for f in batch_files:
            data = np.load(os.path.join(self.cache_dir, f))
            X.append(data['X'])
            y.append(data['y'])

        # Squeeze out any extra batch dimensions if your cached files were saved individually
        return np.array(X).squeeze(), np.array(y).squeeze()
        
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)