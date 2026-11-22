#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
U-Net Training from Large Images - Aligned with TTA Prediction Pipeline.

Features:
- Takes large image + mask pairs directly
- Strict Spatial Splits (Train/Val) to prevent data leakage
- Samples random patches during training (better coverage, road biased)
- Fast C++ backed Albumentations pipeline (TTA + Elastic + Zoom + Noise)
"""

import os
import gc
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.utils import Sequence
from unet import unet
import tensorflow.keras.backend as K
import albumentations as A


# =============================================================================
# CONFIGURATION
# =============================================================================
PATCH_SIZE = (256, 256)
INPUT_SHAPE = PATCH_SIZE + (3,)
PATCHES_PER_EPOCH = 2000      # How many patches to sample per epoch
VALIDATION_PATCHES = 400    # Fixed validation set size
EPOCHS = 500
BATCH_SIZE = 64
SEED = 820

# Large image/mask pairs
IMAGE_MASK_PAIRS = [
    {
        'image': 'data/Input/1877/nla.obj-231444908.tif',
        'mask': 'data/Input/1877/nla.obj-231444908_mask.png',
        'weight': 1.0
    },
    {
        'image': 'data/Input/1886 Atlas of Sydney/Annandale/1083235_1193-01A-Annandale.tif',
        'mask': 'data/Input/1886 Atlas of Sydney/Annandale/1083235_1193-01A-Annandale.png',
        'weight': 1.0
    },
    {
        'image': 'data/Input/1886 Atlas of Sydney/Endfield/1083257_1193-15-Enfield.tif',
        'mask': 'data/Input/1886 Atlas of Sydney/Endfield/1083257_1193-15-Enfield.png',
        'weight': 1.0
    },
    {
        'image': 'data/Input/Melbourne/vc000868-001.tif',
        'mask': 'data/Input/Melbourne/Melbourne_Truth.png',
        'weight': 1.0
    },
]

MODEL_DIR = 'data/Input/Model/Road_Extraction'
os.makedirs(MODEL_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =============================================================================
# CUSTOM METRICS
# =============================================================================
@register_keras_serializable()
def dice_loss(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(K.cast(y_pred, 'float32'))
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))

@register_keras_serializable()
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(K.cast(y_pred, 'float32'))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# =============================================================================
# GPU SETUP
# =============================================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU available and configured.")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("No GPU found. Using CPU.")


# =============================================================================
# LARGE IMAGE HANDLER (With Strict Spatial Splitting)
# =============================================================================
class LargeImageHandler:
    def __init__(self, image_mask_pairs, patch_size=(256, 256), border_margin=0.05, 
                 mode='train', val_split=0.2):
        self.patch_size = patch_size
        self.border_margin = border_margin
        self.mode = mode
        self.sources = []
        
        total_weight = sum(p.get('weight', 1.0) for p in image_mask_pairs)
        
        for pair in image_mask_pairs:
            print(f"Loading {mode.upper()} split for: {os.path.basename(pair['image'])}")
            
            image = cv2.imread(pair['image'])
            if image is None:
                raise ValueError(f"Could not load image: {pair['image']}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            mask = cv2.imread(pair['mask'], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not load mask: {pair['mask']}")
            
            if mask.max() > 1:
                mask = (mask > 127).astype(np.float32)
            else:
                mask = mask.astype(np.float32)
            
            h, w = image.shape[:2]
            
            margin_h = int(h * border_margin)
            margin_w = int(w * border_margin)
            split_y = int(h * (1.0 - val_split))
            
            if mode == 'train':
                valid_region = {
                    'y_min': margin_h,
                    'y_max': split_y - patch_size[0],
                    'x_min': margin_w,
                    'x_max': w - margin_w - patch_size[1]
                }
            elif mode == 'val':
                valid_region = {
                    'y_min': split_y,
                    'y_max': h - margin_h - patch_size[0],
                    'x_min': margin_w,
                    'x_max': w - margin_w - patch_size[1]
                }
            else:
                raise ValueError("mode must be 'train' or 'val'")
                
            if valid_region['y_min'] >= valid_region['y_max'] or valid_region['x_min'] >= valid_region['x_max']:
                print(f"  WARNING: {mode} region for {os.path.basename(pair['image'])} is too small. Skipping.")
                continue
            
            weight = pair.get('weight', 1.0) / total_weight
            
            self.sources.append({
                'image': image,
                'mask': mask,
                'shape': (h, w),
                'valid_region': valid_region,
                'weight': weight,
                'name': os.path.basename(pair['image'])
            })
            
            print(f"  Shape: {w}x{h}, Valid region: {valid_region}")
        
        if not self.sources:
            raise ValueError(f"No valid images loaded for {mode} mode!")
            
        print(f"Loaded {len(self.sources)} image/mask pairs for {mode}.")
    
    def sample_patch(self, prefer_roads=False, road_bias=0.7):
        weights = [s['weight'] for s in self.sources]
        source = random.choices(self.sources, weights=weights, k=1)[0]
        
        region = source['valid_region']
        max_attempts = 50 if prefer_roads else 1
        
        for _ in range(max_attempts):
            y = random.randint(region['y_min'], region['y_max'])
            x = random.randint(region['x_min'], region['x_max'])
            
            img_patch = source['image'][y:y+self.patch_size[0], x:x+self.patch_size[1]]
            mask_patch = source['mask'][y:y+self.patch_size[0], x:x+self.patch_size[1]]
            
            if prefer_roads:
                has_roads = mask_patch.mean() > 0.01
                if has_roads or random.random() > road_bias:
                    break
            else:
                break
        
        # Returns raw uint8 image for faster Albumentations processing
        return img_patch, mask_patch
    
    def get_road_coverage_stats(self, n_samples=100):
        coverages = []
        for _ in range(n_samples):
            _, mask = self.sample_patch(prefer_roads=False)
            coverages.append(mask.mean() * 100)
        
        return {
            'mean': np.mean(coverages), 'std': np.std(coverages),
            'min': np.min(coverages), 'max': np.max(coverages)
        }


# =============================================================================
# ALBUMENTATIONS PIPELINE
# =============================================================================
def get_training_augmentation():
    """
    Highly optimized C++ augmentation pipeline replacing SciPy/CV2 loops.
    Updated with extreme color shifts to make the model invariant to paper color.
    """
    return A.Compose([
        # TTA-style augmentations (D4 symmetry group)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # Zoom / Scaling / Slight Rotations
        A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=15, 
                           border_mode=cv2.BORDER_REFLECT_101, p=0.3),
        
        # Elastic Deformation (Simulates hand-drawn variations & paper warping)
        A.ElasticTransform(alpha=34, sigma=4, alpha_affine=0, 
                           border_mode=cv2.BORDER_REFLECT_101, p=0.3),
        
        # EXTREME COLOR AUGMENTATION
        # 1. Full-spectrum hue rotation (180 = full color wheel) + saturation changes
        A.HueSaturationValue(hue_shift_limit=180, sat_shift_limit=40, val_shift_limit=30, p=0.8),
        
        # 2. RGB Shift to simulate heavy color tints (like green paper or blue ink)
        A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=0.6),
        
        # 3. Broad contrast/brightness changes for faded vs. dark scans
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        
        # Sensor noise
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    ])

# =============================================================================
# DATA GENERATORS
# =============================================================================
class LargeImageGenerator(Sequence):
    def __init__(self, image_handler, patches_per_epoch, batch_size,
                 augment=True, prefer_roads=True, road_bias=0.6):
        self.handler = image_handler
        self.patches_per_epoch = patches_per_epoch
        self.batch_size = batch_size
        self.augment = augment
        self.prefer_roads = prefer_roads
        self.road_bias = road_bias
        self.transform = get_training_augmentation() if augment else None
    
    def __len__(self):
        return self.patches_per_epoch // self.batch_size
    
    def __getitem__(self, idx):
        batch_images = []
        batch_masks = []
        
        for _ in range(self.batch_size):
            img, mask = self.handler.sample_patch(
                prefer_roads=self.prefer_roads,
                road_bias=self.road_bias
            )
            
            if self.augment and self.transform:
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
            
            # Normalize to [0, 1] AFTER augmentation
            img = img.astype(np.float32) / 255.0
            
            batch_images.append(img)
            batch_masks.append(mask[..., np.newaxis])
        
        return np.array(batch_images), np.array(batch_masks)
    
    def on_epoch_end(self):
        pass


class FixedValidationSet:
    def __init__(self, image_handler, n_patches):
        print(f"Creating fixed validation set ({n_patches} patches)...")
        self.images = []
        self.masks = []
        
        for _ in range(n_patches):
            img, mask = image_handler.sample_patch(prefer_roads=True, road_bias=0.5)
            # Normalize to [0, 1] since validation bypasses the generator
            img = img.astype(np.float32) / 255.0
            
            self.images.append(img)
            self.masks.append(mask[..., np.newaxis])
        
        self.images = np.array(self.images)
        self.masks = np.array(self.masks)
        print(f"  Validation set shape: {self.images.shape}")
    
    def get_data(self):
        return self.images, self.masks


# =============================================================================
# VISUALISATION
# =============================================================================
def visualise_samples(image_handler, n_samples=8, save_path=None):
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2.5, 5))
    for i in range(n_samples):
        img, mask = image_handler.sample_patch(prefer_roads=True)
        # Display uint8 images directly
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Patch {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title(f'{mask.mean()*100:.1f}% road')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def visualise_augmentations(image_handler, save_path=None):
    img, mask = image_handler.sample_patch(prefer_roads=True)
    transform = get_training_augmentation()
    
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(mask, cmap='gray')
    axes[1, 0].set_title('Original Mask')
    axes[1, 0].axis('off')
    
    for i in range(1, 6):
        augmented = transform(image=img, mask=mask)
        aug_img = augmented['image']
        aug_mask = augmented['mask']
        
        axes[0, i].imshow(aug_img)
        axes[0, i].set_title(f'Random Aug {i}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(aug_mask, cmap='gray')
        axes[1, i].set_title('Augmented Mask')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def plot_training_history(history, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['dice_coefficient'], 'r-', label='Train')
    axes[0].plot(history.history['val_dice_coefficient'], 'b-', label='Val')
    axes[0].set_title('Dice Coefficient')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Dice')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['loss'], 'r-', label='Train')
    axes[1].plot(history.history['val_loss'], 'b-', label='Val')
    axes[1].set_title('Dice Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


# =============================================================================
# MAIN TRAINING
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("U-Net Training from Large Images")
    print("=" * 60)
    
    # Load large images for TRAINING (Top 80%)
    train_handler = LargeImageHandler(
        IMAGE_MASK_PAIRS,
        patch_size=PATCH_SIZE,
        border_margin=0.05,
        mode='train',
        val_split=0.2
    )
    
    # Load large images for VALIDATION (Bottom 20%)
    val_handler = LargeImageHandler(
        IMAGE_MASK_PAIRS,
        patch_size=PATCH_SIZE,
        border_margin=0.05,
        mode='val',
        val_split=0.2
    )
    
    # Visualise samples
    visualise_samples(
        train_handler, n_samples=8,
        save_path=os.path.join(MODEL_DIR, 'sample_patches.png')
    )
    visualise_augmentations(
        train_handler,
        save_path=os.path.join(MODEL_DIR, 'augmentation_samples.png')
    )
    
    # Create training generator
    train_gen = LargeImageGenerator(
        train_handler,
        patches_per_epoch=PATCHES_PER_EPOCH,
        batch_size=BATCH_SIZE,
        augment=True,
        prefer_roads=True,
        road_bias=0.6
    )
    
    # Fixed validation set
    val_set = FixedValidationSet(val_handler, n_patches=VALIDATION_PATCHES)
    val_images, val_masks = val_set.get_data()
    
    print(f"\nTraining config:")
    print(f"  Patches per epoch: {PATCHES_PER_EPOCH}")
    print(f"  Validation patches: {VALIDATION_PATCHES}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Steps per epoch: {len(train_gen)}")
    
    # Build model
    model = unet(INPUT_SHAPE, output_layer=1)
    
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=dice_loss,
        metrics=[dice_coefficient, tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)]
    )
    
    model.summary()
    
    callbacks = [
        ModelCheckpoint(
            os.path.join(MODEL_DIR, 'best_model.keras'),
            save_best_only=True,
            verbose=1,
            monitor='val_dice_coefficient',
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_dice_coefficient',
            factor=0.5,
            patience=15,
            verbose=1,
            mode='max',
            min_lr=1e-6
        ),
        EarlyStopping(
            monitor='val_dice_coefficient',
            patience=30,
            verbose=1,
            mode='max',
            restore_best_weights=True
        )
    ]
    
    # Train
    print("\nStarting training...")
    history = model.fit(
        train_gen,
        validation_data=(val_images, val_masks),
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # Plot history
    plot_training_history(
        history,
        save_path=os.path.join(MODEL_DIR, 'training_history.png')
    )
    
    # Save final model
    model.save(os.path.join(MODEL_DIR, 'final_model.keras'))
    
    # Final evaluation
    print("\nFinal evaluation on validation set:")
    results = model.evaluate(val_images, val_masks, verbose=1)
    print(f"  Loss: {results[0]:.4f}")
    print(f"  Dice: {results[1]:.4f}")
    print(f"  IoU:  {results[2]:.4f}")
    
    print("\nTraining complete!")