#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized overlap-aware tile stitcher for binary segmentation tiles.

Optimizations:
- Better error handling and logging
- Uses model parameters
- More efficient memory usage
- Cleaner code structure
"""

import os
import re
from pathlib import Path
from typing import Optional, Callable
from PIL import Image, ImageFile
try:
    import numpy as np
except ImportError as e:
    raise ImportError(
        "NumPy is required but could not be imported. "
        "Please install it with: pip install numpy"
    ) from e
except Exception as e:
    raise ImportError(
        f"NumPy import failed with error: {e}. "
        "This may indicate a corrupted NumPy installation. "
        "Try reinstalling: pip uninstall numpy && pip install numpy"
    ) from e

# Disable PIL image size limits
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Constants
TILE_REGEX = re.compile(r'(?:.*?[^a-zA-Z0-9])?tile_(\d+)_(\d+)', re.IGNORECASE)
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def _has_tile_images(folder_path: str) -> bool:
    """Check if folder contains at least one file matching tile pattern."""
    try:
        for fn in os.listdir(folder_path):
            if os.path.splitext(fn.lower())[1] in IMG_EXTS and TILE_REGEX.search(fn):
                return True
    except (FileNotFoundError, PermissionError):
        return False
    return False


def _find_reference_image(reference_folder_path: str) -> Optional[str]:
    """Find a reference image from a folder. Returns path or None."""
    if not reference_folder_path or not os.path.exists(reference_folder_path):
        return None
    try:
        image_files = [
            fn for fn in os.listdir(reference_folder_path)
            if os.path.splitext(fn.lower())[1] in IMG_EXTS
        ]
        if not image_files:
            return None
        image_files.sort()
        return os.path.join(reference_folder_path, image_files[0])
    except Exception:
        return None


def _load_image(img_path: str, color_mode: bool = False, thresh: int = 127) -> np.ndarray:
    """
    Load an image as RGB or binary mask.
    
    Args:
        img_path: Path to image file
        color_mode: If True, return RGB. If False, return binary mask.
        thresh: Threshold for binarization (only used when color_mode=False)
    
    Returns:
        numpy array: RGB (H,W,3) or binary mask (H,W) uint8
    """
    with Image.open(img_path) as im:
        if color_mode:
            im = im.convert('RGB')
            return np.array(im, dtype=np.uint8)
        else:
            im = im.convert('L')
            arr = np.array(im, dtype=np.uint8)
            return np.where(arr > thresh, 255, 0).astype(np.uint8)


def _calculate_tile_position(
    x_idx: int, y_idx: int, max_x: int, max_y: int,
    tile_w: int, tile_h: int,
    stride_x: int, stride_y: int,
    original_w: int, original_h: int
) -> tuple:
    """
    Calculate tile position with edge-snapping logic.
    
    Edge tiles are snapped to image boundaries.
    Interior tiles use uniform stride spacing.
    """
    # X position
    xs = (original_w - tile_w) if x_idx == max_x else (x_idx * stride_x)
    
    # Y position
    ys = (original_h - tile_h) if y_idx == max_y else (y_idx * stride_y)
    
    return xs, ys


def stitch_images(
    folder_path: str,
    output_dir: str,
    reference_folder_path: str,
    overlap_frac: float = 0.10,
    thresh: int = 127,
    color_mode: bool = False,
    log: Optional[Callable[[str], None]] = None
) -> str:
    """
    Stitch image tiles back together with fixed overlap and OR-merge overlapping areas.
    
    Args:
        folder_path: Path to folder containing tiles named like tile_X_Y.png
        output_dir: Directory to save stitched image
        reference_folder_path: Folder with the original image (REQUIRED)
        overlap_frac: Fractional overlap per axis (e.g., 0.10 for 10%)
        thresh: Threshold to binarize input tiles (only used when color_mode=False)
        color_mode: If True, stitch as RGB. If False, stitch as binary mask
        log: Optional logging function
    
    Returns:
        Path to saved stitched PNG
    """
    if log is None:
        log = print
    
    pattern = TILE_REGEX
    image_tiles = {}  # (x,y) -> image array
    
    # Load tiles
    log(f"[STITCH] Loading tiles from: {folder_path}")
    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename.lower())[1]
        if ext in IMG_EXTS:
            match = pattern.search(filename)
            if match:
                x_coord = int(match.group(1))
                y_coord = int(match.group(2))
                img_path = os.path.join(folder_path, filename)
                try:
                    img_array = _load_image(img_path, color_mode=color_mode, thresh=thresh)
                    
                    if color_mode:
                        if img_array.ndim != 3 or img_array.shape[2] != 3:
                            raise ValueError(f"{filename} did not load as RGB. Shape: {img_array.shape}")
                    else:
                        if img_array.ndim != 2:
                            raise ValueError(f"{filename} did not load as single-channel. Shape: {img_array.shape}")
                    
                    image_tiles[(x_coord, y_coord)] = img_array
                except Exception as e:
                    log(f"[WARNING] Failed to load {filename}: {e}")
    
    if not image_tiles:
        raise ValueError(f"No image tiles matched the pattern in: {folder_path}")
    
    # Get reference image dimensions (REQUIRED)
    reference_image_path = _find_reference_image(reference_folder_path)
    if not reference_image_path:
        raise ValueError(
            f"Reference image REQUIRED for accurate stitching!\n"
            f"The tiling script snaps edge tiles to image boundaries, so we need "
            f"the original dimensions to place tiles correctly.\n"
            f"No image found in: {reference_folder_path}"
        )
    
    try:
        with Image.open(reference_image_path) as ref_img:
            original_w, original_h = ref_img.size
        log(f"[STITCH] Reference image size: {original_w} x {original_h}")
    except Exception as e:
        raise ValueError(f"Could not load reference image: {e}")
    
    # Grid size & tile size
    max_x = max(x for x, _ in image_tiles.keys())
    max_y = max(y for _, y in image_tiles.keys())
    log(f"[STITCH] Grid size: {max_x + 1} x {max_y + 1} tiles")
    log(f"[STITCH] Color mode: {'RGB' if color_mode else 'Binary'}")
    
    sample_tile = next(iter(image_tiles.values()))
    tile_h, tile_w = sample_tile.shape[:2]
    log(f"[STITCH] Tile size: {tile_w} x {tile_h}")
    
    # Validate all tiles are same size
    for coord, img_array in image_tiles.items():
        expected_shape = (tile_h, tile_w, 3) if color_mode else (tile_h, tile_w)
        if img_array.shape != expected_shape:
            raise ValueError(f"Tile {coord} has inconsistent size: {img_array.shape} vs expected {expected_shape}")
    
    # Compute stride from overlap
    ovx = int(round(tile_w * overlap_frac))
    ovy = int(round(tile_h * overlap_frac))
    stride_x = tile_w - ovx
    stride_y = tile_h - ovy
    
    if stride_x <= 0 or stride_y <= 0:
        raise ValueError(f"Invalid overlap leading to non-positive stride: ({stride_x}, {stride_y})")
    
    log(f"[STITCH] Overlap: {ovx} x {ovy} pixels ({overlap_frac*100:.1f}%)")
    log(f"[STITCH] Stride: {stride_x} x {stride_y} pixels")
    
    # Create canvas using original dimensions
    if color_mode:
        stitched = np.zeros((original_h, original_w, 3), dtype=np.uint8)
    else:
        stitched = np.zeros((original_h, original_w), dtype=np.uint8)
    log(f"[STITCH] Canvas size: {original_w} x {original_h}")
    
    # Place tiles with proper edge-snapping logic
    placed_count = 0
    for (x, y), img_array in sorted(image_tiles.items()):
        # Calculate position using edge-snapping logic
        xs, ys = _calculate_tile_position(
            x, y, max_x, max_y,
            tile_w, tile_h,
            stride_x, stride_y,
            original_w, original_h
        )
        
        xe = xs + tile_w
        ye = ys + tile_h
        
        # Clamp to canvas bounds
        xe_clamp = min(xe, original_w)
        ye_clamp = min(ye, original_h)
        xs_clamp = max(xs, 0)
        ys_clamp = max(ys, 0)
        
        w_place = xe_clamp - xs_clamp
        h_place = ye_clamp - ys_clamp
        
        if w_place <= 0 or h_place <= 0:
            log(f"[WARNING] Tile ({x},{y}) outside canvas bounds, skipping.")
            continue
        
        # Calculate source region from tile
        tile_x_start = xs_clamp - xs
        tile_y_start = ys_clamp - ys
        tile_x_end = tile_x_start + w_place
        tile_y_end = tile_y_start + h_place
        
        # Merge into canvas (OR == max for binary {0,255}, also works for RGB)
        canvas_region = stitched[ys_clamp:ye_clamp, xs_clamp:xe_clamp]
        tile_region = img_array[tile_y_start:tile_y_end, tile_x_start:tile_x_end]
        
        np.maximum(canvas_region, tile_region, out=canvas_region)
        placed_count += 1
    
    log(f"[STITCH] Successfully placed {placed_count}/{len(image_tiles)} tiles")
    
    # Save as PNG
    os.makedirs(output_dir, exist_ok=True)
    if color_mode:
        out_pil = Image.fromarray(stitched, mode='RGB')
    else:
        out_pil = Image.fromarray(stitched, mode='L')
    
    output_path = os.path.join(output_dir, 'Stitched.png')
    out_pil.save(output_path, optimize=True)
    
    log(f"[STITCH] Saved stitched image to: {output_path}")
    log(f"[STITCH] Final image size: {out_pil.size}")
    
    return output_path

