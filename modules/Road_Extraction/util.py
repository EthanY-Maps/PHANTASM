"""
Utility functions for Road Extraction pipeline.

Includes:
- Data loading and augmentation for training
- Scale optimization for inference on maps of varying scales
- Road segmentation scoring metrics
- Visualisation tools
"""
import numpy as np
import os
import random
import cv2
import scipy.ndimage
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from scipy.ndimage import label, binary_erosion



# ── DATA AUGMENTATION ──

def zoom_image(image, mask, zoom_range=(0.8, 1.2)):
    """
    Apply random zoom augmentation to image and mask.
    
    Args:
        image: Input image array
        mask: Input mask array
        zoom_range: Tuple of (min_zoom, max_zoom)
        
    Returns:
        Tuple of (zoomed_image, zoomed_mask)
    """
    zoom_factor = np.random.uniform(*zoom_range)
    height, width = image.shape[:2]
    
    new_height = int(height * zoom_factor)
    new_width = int(width * zoom_factor)
    
    y1 = max(0, (new_height - height) // 2)
    x1 = max(0, (new_width - width) // 2)
    y2 = y1 + height
    x2 = x1 + width
   
    mask = img_to_array(mask).astype(np.float32)
    
    image = tf.image.resize(image[y1:y2, x1:x2], (height, width))
    mask = tf.image.resize(mask[y1:y2, x1:x2], (height, width))
    mask = img_to_array(mask, dtype=np.bool_)
    
    return image, mask


def elastic_transform(image, mask, alpha, sigma, random_state=None):
    """
    Apply elastic deformation to image and mask.
    
    Args:
        image: Input image array
        mask: Input mask array
        alpha: Scaling factor for deformation intensity
        sigma: Gaussian filter smoothing factor
        random_state: Optional RandomState for reproducibility
        
    Returns:
        Tuple of (deformed_image, deformed_mask)
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
        
    mask = img_to_array(mask).astype(np.float32)

    shape = image.shape
    shape_m = mask.shape
    
    # Generate displacement fields for image
    dx = scipy.ndimage.gaussian_filter(
        (random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
    ) * alpha
    dy = scipy.ndimage.gaussian_filter(
        (random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
    ) * alpha

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    indices = (
        np.reshape(x + dx, (-1, 1)),
        np.reshape(y + dy, (-1, 1)),
        np.reshape(z, (-1, 1))
    )

    distorted_image = scipy.ndimage.map_coordinates(
        image, indices, order=1, mode='reflect'
    ).reshape(shape)
    
    # Generate displacement fields for mask
    dx = scipy.ndimage.gaussian_filter(
        (random_state.rand(*shape_m) * 2 - 1), sigma, mode="constant", cval=0
    ) * alpha
    dy = scipy.ndimage.gaussian_filter(
        (random_state.rand(*shape_m) * 2 - 1), sigma, mode="constant", cval=0
    ) * alpha

    x, y, z = np.meshgrid(np.arange(shape_m[0]), np.arange(shape_m[1]), np.arange(shape_m[2]))
    indices = (
        np.reshape(x + dx, (-1, 1)),
        np.reshape(y + dy, (-1, 1)),
        np.reshape(z, (-1, 1))
    )
    
    distorted_mask = scipy.ndimage.map_coordinates(
        mask, indices, order=1, mode='reflect'
    ).reshape(shape_m)
    
    distorted_image = img_to_array(distorted_image) / 255.0
    distorted_mask = img_to_array(distorted_mask, dtype=np.bool_)

    return distorted_image, distorted_mask


def preprocess_data(img_path, mask_path, sorted_images, sorted_masks, input_size, augmented=False):
    """
    Load and preprocess image/mask pairs with optional augmentation.
    
    Args:
        img_path: Directory containing images
        mask_path: Directory containing masks
        sorted_images: List of image filenames
        sorted_masks: List of mask filenames
        input_size: Tuple of (height, width) for resizing
        augmented: Whether to apply augmentation
        
    Returns:
        Tuple of (images_array, masks_array)
    """
    def load_and_preprocess_image(img_file, mask_file):
        img_file = img_file.numpy().decode('utf-8')
        mask_file = mask_file.numpy().decode('utf-8')
        
        img = load_img(
            os.path.join(img_path, img_file),
            target_size=input_size,
            color_mode='rgb'
        )
        mask = load_img(
            os.path.join(mask_path, mask_file),
            target_size=input_size,
            color_mode='grayscale'
        )
        
        img_array = img_to_array(img) / 255.0
        mask_array = img_to_array(mask, dtype=np.bool_)

        if augmented:
            # Elastic deformation
            alpha = 34
            sigma = 4
            img_array, mask_array = elastic_transform(img_array, mask_array, alpha, sigma)

            # Random horizontal flip
            if random.randint(0, 1):
                img_array = tf.image.flip_left_right(img_array)
                mask_array = tf.image.flip_left_right(mask_array)
            
            # Random vertical flip
            if random.randint(0, 1):
                img_array = tf.image.flip_up_down(img_array)
                mask_array = tf.image.flip_up_down(mask_array)

            # Random rotation (90, 180, or 270 degrees)
            k = random.randint(1, 3)
            img_array = tf.image.rot90(img_array, k)
            mask_array = tf.image.rot90(mask_array, k)
            
            # Random shift
            shift_value = random.uniform(-0.1, 0.1)
            shift_pixels = int(shift_value * input_size[0])
            img_array = tf.roll(img_array, shift=shift_pixels, axis=-1)
            mask_array = tf.roll(mask_array, shift=shift_pixels, axis=-1)

            # Zoom augmentation
            img_array, mask_array = zoom_image(img_array, mask_array)

            # Colour augmentations (image only)
            img_array = tf.image.random_brightness(img_array, max_delta=0.6)
            img_array = tf.image.random_contrast(img_array, lower=0.5, upper=1.0)
            img_array = tf.image.per_image_standardization(img_array)
            img_array = tf.image.random_hue(img_array, max_delta=0.1)
            
            # Noise injection
            noise = tf.random.normal(shape=tf.shape(img_array), mean=0.0, stddev=0.05)
            img_array = img_array + noise

        return img_array, mask_array
    
    dataset = tf.data.Dataset.from_tensor_slices((sorted_images, sorted_masks))
    dataset = dataset.map(
        lambda img, mask: tf.py_function(
            load_and_preprocess_image, [img, mask], [tf.float32, tf.bool]
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    
    images, masks = [], []
    for img_batch, mask_batch in tqdm(dataset, desc="Preprocessing data"):
        images.extend(img_batch.numpy())
        masks.extend(mask_batch.numpy())
    
    return np.array(images), np.array(masks)




# ── ROAD SEGMENTATION SCORING ──

def score_road_segmentation(pred_mask, threshold=0.5, expected_coverage=(0.05, 0.8), detailed=False):
    """
    Score a road segmentation prediction - optimised for historic maps with thick roads.
    
    Focuses on:
    1. Connectivity (primary - wrong scale causes fragmentation)
    2. Coverage in expected range
    3. Compactness (roads shouldn't be overly blobby)
    
    Args:
        pred_mask: Predicted segmentation mask (values 0-1 or 0-255)
        threshold: Binarisation threshold (applied after normalisation)
        expected_coverage: Tuple of (min, max) expected road coverage
        detailed: If True, return (score, details_dict)
        
    Returns:
        Score value (higher is better), or -1.0 for invalid predictions
        If detailed=True, returns (score, details_dict)
    """
    from scipy.ndimage import binary_erosion, label
    
    # Normalise if needed
    if pred_mask.max() > 1:
        pred_mask = pred_mask / 255.0
        
    binary = pred_mask > threshold
    
    # Coverage check
    coverage = binary.mean()
    
    invalid_details = None
    if coverage < 0.005 or coverage > 0.6:
        invalid_details = {'valid': False, 'reason': 'coverage out of range', 'coverage_pct': coverage * 100}
    
    road_area = binary.sum()
    if road_area == 0 and invalid_details is None:
        invalid_details = {'valid': False, 'reason': 'no roads detected'}
    
    if invalid_details is None:
        labeled, num_components = label(binary)
        if num_components == 0:
            invalid_details = {'valid': False, 'reason': 'no components'}
    
    if invalid_details:
        if detailed:
            return -1.0, invalid_details
        return -1.0
    
    # 1. CONNECTIVITY (weight: 4.0) - most important for scale detection
    # At wrong scale, roads fragment into many disconnected pieces
    component_sizes = np.bincount(labeled.flatten())[1:]
    largest_component_ratio = component_sizes.max() / road_area
    
    # Strong penalty for fragmentation
    # 1 comp = 1.0, 2 = 0.7, 5 = 0.35, 10 = 0.18
    connectivity_score = largest_component_ratio * (0.7 ** np.log2(max(1, num_components)))
    
    # 2. COVERAGE (weight: 2.0) - should be in expected range
    min_cov, max_cov = expected_coverage
    if min_cov <= coverage <= max_cov:
        # Normalise to 0-1 within range, peak at middle
        mid = (min_cov + max_cov) / 2
        coverage_score = 1.0 - abs(coverage - mid) / (max_cov - min_cov)
    else:
        coverage_score = 0.3  # Outside range but not invalid
    
    # 3. COMPACTNESS (weight: 1.0) - penalise very blobby predictions
    # Use perimeter-to-area ratio
    eroded = binary_erosion(binary)
    perimeter = np.sum(binary & ~eroded)
    perimeter_ratio = perimeter / road_area if road_area > 0 else 0
    
    # Ideal range roughly 0.03-0.20 for road networks
    if 0.03 <= perimeter_ratio <= 0.20:
        compactness_score = 1.0 - abs(perimeter_ratio - 0.10) / 0.10
        compactness_score = max(0.3, compactness_score)
    else:
        compactness_score = 0.3
    
    # Combined score (max = 7.0)
    score = (
        connectivity_score * 4.0 +
        coverage_score * 2.0 +
        compactness_score * 1.0
    )
    
    if detailed:
        details = {
            'valid': True,
            'connectivity': connectivity_score,
            'coverage': coverage_score,
            'compactness': compactness_score,
            'num_components': num_components,
            'largest_component_pct': largest_component_ratio * 100,
            'coverage_pct': coverage * 100,
            'perimeter_ratio': perimeter_ratio
        }
        return score, details
    
    return score



# ── SCALE OPTIMIZATION HELPERS ──

def _sample_patches(image, patch_size, num_samples, min_std=10.0, border_margin=0.1,
                    grid_scatter=True, context_factor=4.0):
    """
    Sample high-variance patches from the interior of an image.

    Patches are sampled at `patch_size * context_factor` so they carry enough
    surrounding context for the scale-optimisation step to faithfully simulate
    what `predict_large_map_tta` will see at any scale in
    `[1/context_factor, ∞)`. Falls back to plain `patch_size` if the image is
    too small to provide that context.

    Args:
        image: Full image array (RGB)
        patch_size: Tuple of (height, width) — the model input size, NOT the
            sampled patch size. Sampled patches are larger when context_factor > 1.
        num_samples: Maximum number of patches to return
        min_std: Minimum standard deviation to consider patch valid
        border_margin: Fraction of image to exclude from edges (0.1 = 10% margin)
        grid_scatter: If True, divide interior into grid and sample from each cell
        context_factor: Patches are this many times bigger than `patch_size`. 4.0
            covers scale_val ≥ 0.25 faithfully; raise it for sub-quarter scales.

    Returns:
        List of patch arrays of size (patch_size[0]*context_factor, patch_size[1]*context_factor),
        or the smaller plain `patch_size` if the image was too small.
    """
    h, w = image.shape[:2]
    ctx_h = int(patch_size[0] * context_factor)
    ctx_w = int(patch_size[1] * context_factor)

    # Fall back to plain patch_size if the image isn't big enough to hold the
    # context window comfortably (need ≥ 1.5x the context dimension to leave
    # room for sampling without hugging boundaries).
    if h < ctx_h * 1.5 or w < ctx_w * 1.5:
        ctx_h, ctx_w = patch_size

    patch_h, patch_w = ctx_h, ctx_w
    
    # Calculate interior region (exclude borders)
    margin_y = int(h * border_margin)
    margin_x = int(w * border_margin)
    
    # Ensure margins are at least patch_size to avoid edge patches
    margin_y = max(margin_y, patch_h)
    margin_x = max(margin_x, patch_w)
    
    # Define interior bounds
    y_start = margin_y
    y_end = h - margin_y - patch_h
    x_start = margin_x
    x_end = w - margin_x - patch_w
    
    # Check if image is large enough
    if y_end <= y_start or x_end <= x_start:
        # Image too small for margins, fall back to centre region
        y_start = h // 4
        y_end = 3 * h // 4 - patch_h
        x_start = w // 4
        x_end = 3 * w // 4 - patch_w
        
        if y_end <= y_start or x_end <= x_start:
            # Still too small, just sample from centre
            cy, cx = h // 2 - patch_h // 2, w // 2 - patch_w // 2
            patch = image[cy:cy+patch_h, cx:cx+patch_w]
            return [patch.copy()] if np.std(patch) > min_std else []
    
    candidates = []
    
    if grid_scatter:
        # Divide interior into grid cells and sample best patch from each
        grid_size = int(np.ceil(np.sqrt(num_samples * 2)))  # More cells than needed
        
        cell_h = (y_end - y_start) // grid_size
        cell_w = (x_end - x_start) // grid_size
        
        # Ensure cells are at least patch size
        cell_h = max(cell_h, patch_h)
        cell_w = max(cell_w, patch_w)
        
        for grid_y in range(grid_size):
            for grid_x in range(grid_size):
                # Cell bounds
                cell_y_start = y_start + grid_y * cell_h
                cell_y_end = min(cell_y_start + cell_h, y_end)
                cell_x_start = x_start + grid_x * cell_w
                cell_x_end = min(cell_x_start + cell_w, x_end)
                
                # Skip if cell too small
                if cell_y_end - cell_y_start < patch_h or cell_x_end - cell_x_start < patch_w:
                    continue
                
                # Find best patch in this cell (sample a few positions)
                best_std = 0
                best_patch = None
                
                # Sample several positions within the cell
                for _ in range(5):
                    y = np.random.randint(cell_y_start, cell_y_end - patch_h + 1)
                    x = np.random.randint(cell_x_start, cell_x_end - patch_w + 1)
                    
                    patch = image[y:y+patch_h, x:x+patch_w]
                    std_dev = np.std(patch)
                    
                    if std_dev > best_std and std_dev > min_std:
                        best_std = std_dev
                        best_patch = patch.copy()
                
                if best_patch is not None:
                    candidates.append((best_std, best_patch))
    else:
        # Original approach but restricted to interior
        step_y = patch_h // 2
        step_x = patch_w // 2
        
        for y in range(y_start, y_end, step_y):
            for x in range(x_start, x_end, step_x):
                patch = image[y:y+patch_h, x:x+patch_w]
                std_dev = np.std(patch)
                if std_dev > min_std:
                    candidates.append((std_dev, patch.copy()))
    
    # Sort by variance (descending) and take top samples
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [p[1] for p in candidates[:num_samples]]


def _prepare_patch_for_inference(raw_patch, scale_val, patch_size):
    """
    Simulate what `predict_large_map_tta` feeds the model when running at
    `scale_val`. Real inference resizes the full image by `scale_val` then
    tiles into `patch_size` chunks, so each model input represents a real-world
    region of `patch_size / scale_val`. We replicate that by centre-cropping
    `raw_patch` to that real-world region and resizing to `patch_size` — never
    pad with white, which is what the old version did and why scoring was
    unreliable on busy maps.

    Args:
        raw_patch: Sampled patch array (typically `patch_size * context_factor`).
        scale_val: Inference scale factor to simulate.
        patch_size: Target model-input size tuple (height, width).

    Returns:
        Prepared patch array normalised to [0, 1].
    """
    h, w = raw_patch.shape[:2]
    target_h = max(1, int(round(patch_size[0] / scale_val)))
    target_w = max(1, int(round(patch_size[1] / scale_val)))

    if target_h > h or target_w > w:
        # Patch was sampled with insufficient context for this scale (typically
        # very small `scale_val` on a small image). Best we can do without
        # introducing white padding artefacts is to use the full patch.
        sub = raw_patch
    else:
        sy = (h - target_h) // 2
        sx = (w - target_w) // 2
        sub = raw_patch[sy:sy + target_h, sx:sx + target_w]

    # Match the interpolation choice of `predict_large_map_tta`'s full-image
    # resize so the simulation tracks production behaviour byte-for-byte.
    interp = cv2.INTER_AREA if sub.shape[0] > patch_size[0] else cv2.INTER_LINEAR
    resized = cv2.resize(sub, (patch_size[1], patch_size[0]), interpolation=interp)
    return resized.astype(np.float32) / 255.0



# ── SCALE OPTIMIZATION ──

def get_optimal_scale(
    model_or_path,
    image_path,
    patch_size=(256, 256),
    num_samples=50,
    scale_range=(0.1, 3.0),
    coarse_step=0.1,
    fine_step=0.02,
    custom_objects=None,
    verbose=True,
    save_patches=None,
    border_margin=0.1,
    log=None,
    prior_sigma_log=0.7,
):
    """
    Determine the optimal scaling factor for road segmentation inference.

    Args:
        model_or_path: Loaded Keras model or path to saved model.
        image_path: Path to the image to analyse.
        patch_size: Model input size tuple (height, width).
        num_samples: Number of patches to sample for evaluation.
        scale_range: Tuple of (min_scale, max_scale) to search.
        coarse_step: Step size for initial coarse sweep.
        fine_step: Step size for fine-tuning around best scale.
        custom_objects: Custom objects dict for model loading.
        verbose: Whether to show tqdm progress bars.
        save_patches: Directory path to save sampled patches (None to skip).
        border_margin: Fraction of image to exclude from edges.
        log: Optional logging callback. Falls back to print when verbose=True.

    Returns:
        Optimal scale factor (float).
    """
    from tensorflow.keras.models import load_model as keras_load_model

    if log is None:
        log = print if verbose else lambda *a, **k: None

    log(f"[INFO] Optimising scale for: {os.path.basename(image_path)}")

    # Load model if path provided
    if isinstance(model_or_path, str):
        log("[INFO] Loading model...")
        try:
            model = keras_load_model(model_or_path, custom_objects=custom_objects)
        except Exception as e:
            log(f"[ERROR] Failed to load model: {e}")
            return 1.0
    else:
        model = model_or_path

    # Load image — use PIL for format robustness
    try:
        from PIL import Image as PILImage
        PILImage.MAX_IMAGE_PIXELS = None
        full_image = np.array(PILImage.open(image_path).convert('RGB'))
    except Exception:
        full_image = cv2.imread(image_path)
        if full_image is None:
            log(f"[ERROR] Could not load image at {image_path}")
            return 1.0
        full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)

    h, w = full_image.shape[:2]
    log(f"[INFO] Image size: {w} x {h}")

    log(f"[INFO] Sampling {num_samples} patches from interior...")
    patches = _sample_patches(full_image, patch_size, num_samples, border_margin=border_margin)

    # Fallback: Systematic Tiling
    if not patches:
        log("[WARNING] No high-variance patches found. Falling back to systematic tiling...")
        tiles = []
        ph, pw = patch_size
        min_tile_std = 2.0
        for y in range(h // ph):
            for x in range(w // pw):
                tile = full_image[y * ph:(y + 1) * ph, x * pw:(x + 1) * pw]
                if np.std(tile) > min_tile_std:
                    tiles.append(tile)

        max_fallback_tiles = 100
        if len(tiles) > max_fallback_tiles:
            indices = np.linspace(0, len(tiles) - 1, max_fallback_tiles, dtype=int)
            patches = [tiles[i] for i in indices]
        else:
            patches = tiles

        if not patches:
            log("[ERROR] Image appears to be completely empty/solid colour.")
            return 1.0

        log(f"[INFO] Extracted {len(patches)} valid tiles for optimisation.")

    log(f"[INFO] Using {len(patches)} patch(es) for scale optimisation")

    if save_patches:
        os.makedirs(save_patches, exist_ok=True)
        log(f"[INFO] Scale patches will be saved to {save_patches}/")

    # Soft prior over scales — log-Gaussian centred at 1.0. Multiplies the
    # patch score so it acts as a tie-breaker when scoring is comparable;
    # large scoring differences still win. Counteracts the tendency of busy
    # maps to pick scales where ornamentation/cross-hatching happens to look
    # road-like to the model. With sigma_log=0.7:
    #   scale 0.5 → 0.61 weight, scale 0.75 → 0.92, scale 1.0 → 1.0,
    #   scale 1.4 → 0.86, scale 2.0 → 0.61.
    # Pass prior_sigma_log=None (or a large value like 100) to effectively disable.
    if prior_sigma_log is None or prior_sigma_log <= 0:
        def _scale_prior(s: float) -> float: return 1.0
    else:
        sigma_log = float(prior_sigma_log)
        def _scale_prior(s: float) -> float:
            return float(np.exp(-0.5 * (np.log(s) / sigma_log) ** 2))

    def evaluate_scale(scale_val):
        batch_input = np.array([
            _prepare_patch_for_inference(p, scale_val, patch_size)
            for p in patches
        ])
        preds = model.predict(batch_input, verbose=0)
        scores = [score_road_segmentation(np.squeeze(pred)) for pred in preds]
        if not scores:
            return -1.0
        base = float(np.mean(scores))
        # Apply prior only to positive scores; preserve sign of negative means
        # (which represent invalid/clamped scales) so they still get filtered.
        return base * _scale_prior(scale_val) if base > 0 else base

    results = {}

    log(f"[INFO] Coarse sweep (step={coarse_step})...")
    coarse_scales = np.arange(scale_range[0], scale_range[1], coarse_step)
    for scale in tqdm(coarse_scales, desc="Coarse sweep", disable=not verbose):
        scale = round(scale, 2)
        results[scale] = evaluate_scale(scale)

    valid = {k: v for k, v in results.items() if v > 0}
    if not valid:
        log("[WARNING] No valid scales found. Defaulting to 1.0")
        return 1.0

    best_scale = max(valid, key=valid.get)
    best_score = valid[best_scale]
    log(f"[INFO] Best coarse scale: {best_scale:.2f} (score: {best_score:.4f})")

    log(f"[INFO] Fine sweep (step={fine_step})...")
    fine_start = max(scale_range[0], best_scale - 0.15)
    fine_end   = min(scale_range[1], best_scale + 0.15)
    fine_scales = np.arange(fine_start, fine_end, fine_step)

    for scale in tqdm(fine_scales, desc="Fine sweep", disable=not verbose):
        scale = round(scale, 3)
        if scale in results:
            continue
        score = evaluate_scale(scale)
        results[scale] = score
        if score > best_score:
            best_score = score
            best_scale = scale
            log(f"[INFO]   New best: {scale:.3f} (score: {score:.4f})")

    log(f"[INFO] Optimal scale: {best_scale:.3f}")
    
    # Save predictions at key scales if patches were saved
    if save_patches:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        scales_to_save = sorted(set([0.5, 1.0, 1.5, 2.0, best_scale]))
        log(f"[INFO] Saving predictions at scales {scales_to_save}...")
        
        # Summary file
        summary_path = os.path.join(save_patches, f"{base_name}_scale_summary.txt")
        with open(summary_path, 'w') as summary_file:
            summary_file.write(f"Scale Optimization Summary for {base_name}\n")
            summary_file.write(f"{'='*60}\n")
            summary_file.write(f"Chosen optimal scale: {best_scale:.3f}\n")
            summary_file.write(f"Optimal score: {best_score:.4f}\n\n")
            summary_file.write(f"All scale results:\n")
            for scale in sorted(results.keys()):
                marker = " <-- CHOSEN" if scale == best_scale else ""
                summary_file.write(f"  Scale {scale:.2f}: {results[scale]:.4f}{marker}\n")
            summary_file.write(f"\n{'='*60}\n\n")
        
        for scale in scales_to_save:
            scale_dir = os.path.join(save_patches, f"scale_{scale:.2f}")
            os.makedirs(scale_dir, exist_ok=True)
            
            # Prepare all patches for this scale
            batch_input = np.array([
                _prepare_patch_for_inference(p, scale, patch_size)
                for p in patches
            ])
            
            # Get predictions
            preds = model.predict(batch_input, verbose=0)
            
            # Track scores for this scale
            scale_scores = []
            
            # Save each patch and its prediction
            for idx, (patch, pred, scaled_input) in enumerate(zip(patches, preds, batch_input)):
                pred_mask = np.squeeze(pred)
                score, details = score_road_segmentation(pred_mask, detailed=True)
                scale_scores.append((idx, score, details))
                
                # Save original input patch
                patch_path = os.path.join(scale_dir, f"{base_name}_patch_{idx:03d}_original.png")
                cv2.imwrite(patch_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                
                # Save scaled input (what model actually sees)
                scaled_vis = (scaled_input * 255).astype(np.uint8)
                scaled_path = os.path.join(scale_dir, f"{base_name}_patch_{idx:03d}_scaled_input.png")
                cv2.imwrite(scaled_path, cv2.cvtColor(scaled_vis, cv2.COLOR_RGB2BGR))
                
                # Save prediction heatmap
                pred_norm = (pred_mask * 255).astype(np.uint8)
                pred_color = cv2.applyColorMap(pred_norm, cv2.COLORMAP_VIRIDIS)
                pred_path = os.path.join(scale_dir, f"{base_name}_patch_{idx:03d}_pred.png")
                cv2.imwrite(pred_path, pred_color)
                
                # Save binary mask
                binary = (pred_mask > 0.5).astype(np.uint8) * 255
                binary_path = os.path.join(scale_dir, f"{base_name}_patch_{idx:03d}_binary.png")
                cv2.imwrite(binary_path, binary)
            
            # Write per-scale summary
            scale_summary_path = os.path.join(scale_dir, f"_scale_{scale:.2f}_summary.txt")
            with open(scale_summary_path, 'w') as f:
                f.write(f"Scale {scale:.2f} Patch Scores\n")
                f.write(f"{'='*50}\n\n")
                valid_scores = [s for _, s, d in scale_scores if d.get('valid', False)]
                f.write(f"Mean score: {np.mean(valid_scores):.4f}\n" if valid_scores else "Mean score: N/A\n")
                f.write(f"Valid patches: {len(valid_scores)}/{len(scale_scores)}\n\n")
                for idx, score, details in scale_scores:
                    if details.get('valid', False):
                        f.write(f"Patch {idx:03d}: score={score:.2f}, conn={details['connectivity']:.2f}, "
                               f"cov={details['coverage_pct']:.1f}%, comp={details['num_components']}\n")
                    else:
                        f.write(f"Patch {idx:03d}: INVALID - {details.get('reason', 'unknown')}\n")
            
            log(f"[INFO]   Saved scale {scale:.2f} predictions")
    
    return best_scale



# ── VISUALISATION ──
