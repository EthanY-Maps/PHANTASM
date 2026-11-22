"""U-Net inference with TTA, spline blending, and automatic scale optimization."""
import os
import gc
import numpy as np
import cv2
import tensorflow as tf
from scipy.signal.windows import hann
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras import backend as K
from tqdm import tqdm

from utils import save_json

try:
    from modules.Road_Extraction.util import get_optimal_scale
except ModuleNotFoundError:
    from util import get_optimal_scale


# `register_keras_serializable` so load_model() can reconstruct these from a saved .keras file.
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


def predict_with_tta(model, batch_images, weights=None):
    """Predict on original + 3 rotations + 2 flips, averaged into one weighted prediction.

    All 6 augmentations are concatenated into a single mega-batch to amortise
    the per-call Python overhead. `weights` order: [0°, 90°, 180°, 270°, H-flip, V-flip].
    """
    if weights is None:
        weights = [0.3, 0.15, 0.15, 0.15, 0.125, 0.125]
    if len(weights) != 6:
        raise ValueError(
            f"tta_weights must have exactly 6 entries (one per augmentation: "
            f"[0°, 90°, 180°, 270°, H-flip, V-flip]); got {len(weights)}."
        )
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError(f"tta_weights must sum to > 0; got {weight_sum}.")
    weights = [w / weight_sum for w in weights]

    # `.copy()` after np.flip — flip returns a non-contiguous view, but model.predict
    # silently re-allocates the input anyway; .copy() makes it a deliberate one allocation.
    imgs_90  = np.rot90(batch_images, k=1, axes=(1, 2))
    imgs_180 = np.rot90(batch_images, k=2, axes=(1, 2))
    imgs_270 = np.rot90(batch_images, k=3, axes=(1, 2))
    imgs_fh  = np.flip(batch_images, axis=2).copy()
    imgs_fv  = np.flip(batch_images, axis=1).copy()

    mega_batch = np.concatenate([batch_images, imgs_90, imgs_180, imgs_270, imgs_fh, imgs_fv], axis=0)
    mega_preds = model.predict(mega_batch, verbose=0)
    preds_0, preds_90, preds_180, preds_270, preds_fh, preds_fv = np.split(mega_preds, 6)

    preds_90  = np.rot90(preds_90,  k=-1, axes=(1, 2))
    preds_180 = np.rot90(preds_180, k=-2, axes=(1, 2))
    preds_270 = np.rot90(preds_270, k=-3, axes=(1, 2))
    preds_fh  = np.flip(preds_fh, axis=2).copy()
    preds_fv  = np.flip(preds_fv, axis=1).copy()

    return (
        weights[0] * preds_0  + weights[1] * preds_90  + weights[2] * preds_180 +
        weights[3] * preds_270 + weights[4] * preds_fh  + weights[5] * preds_fv
    )


def generate_spline_window(window_size, power=1):
    """2D Hann window raised to `power` — used as blending weights at patch overlaps.

    Default `power=1` gives standard Hann, which satisfies partition-of-unity
    with stride=N/2 (overlapping windows sum to 1.0 everywhere). Higher powers
    bias toward patch centres but break that property, leaving the per-pixel
    weight-sum oscillating between 0.5 and 1.0 across the stride — that ratio
    amplifies any U-Net edge-prediction noise into visible tiling stripes.
    Don't raise above 1 unless you're prepared to see seams.
    """
    h_hann = hann(window_size[0])
    w_hann = hann(window_size[1])
    h_hann = h_hann.reshape(window_size[0], 1)
    w_hann = w_hann.reshape(1, window_size[1])
    window = h_hann * w_hann
    window = window ** power
    return window


def _coverage_pad(dim: int, window: int, stride: int) -> int:
    """
    Minimum padding so that windows starting at every multiple of `stride`
    fully cover [0, dim). Avoids over-padding to a multiple of `window`,
    which produces phantom windows composed entirely of reflected padding.
    """
    if dim <= window:
        return window - dim
    extra = (dim - window) % stride
    return 0 if extra == 0 else stride - extra


def pad_image(img, window_size, stride):
    """Pad img so a `stride`-spaced sliding window covers every original pixel.

    Returns (padded_image, pad_height, pad_width).
    """
    h, w, _ = img.shape
    pad_h = _coverage_pad(h, window_size[0], stride)
    pad_w = _coverage_pad(w, window_size[1], stride)
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    return img_padded, pad_h, pad_w


def predict_large_map_tta(
    MODEL_PATH,
    image_path,
    output_folder,
    window_size=(256, 256),
    stride=128,
    batch_size=256,   # matches RoadExtractionParameters.batch_size and CLI default
    use_tta=True,
    auto_scale=True,
    fixed_scale=None,
    tta_weights=None,
    save_patches=False,
    border_margin=0.1,
    skip_empty_patches=True,
    min_patch_std=5.0,
    use_clahe=True,
    clahe_clip_limit=2.0,
    clahe_grid_size=16,
    log=None,
):
    """Run sliding-window inference on a large map image, with optional TTA and auto-scaling.

    Returns (npy_path, output_parameters) on success, (None, None) on model-load failure.
    """
    if log is None:
        log = print

    os.makedirs(os.path.join(output_folder, "npy_raw"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "heatmaps"), exist_ok=True)

    log(f"[INFO] Loading model from {MODEL_PATH}...")
    try:
        model = load_model(
            MODEL_PATH,
            custom_objects={'dice_loss': dice_loss, 'dice_coefficient': dice_coefficient}
        )
    except Exception as e:
        log(f"[ERROR] Failed to load model: {e}")
        return None, None

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    log(f"[INFO] Processing: {base_name} ({'with TTA' if use_tta else 'without TTA'})")

    # PIL handles TIFF/JP2/etc that cv2 silently mishandles on some installs.
    try:
        from PIL import Image as PILImage
        PILImage.MAX_IMAGE_PIXELS = None
        full_image = np.array(PILImage.open(image_path).convert('RGB'))
    except Exception as e:
        log(f"[ERROR] Failed to read image {image_path}: {e}")
        return None, None

    raw_h, raw_w = full_image.shape[:2]
    log(f"[INFO] Original size: {raw_w} x {raw_h}")

    if fixed_scale is not None:
        optimal_scale = fixed_scale
        log(f"[INFO] Using fixed scale: {optimal_scale:.3f}")
    elif auto_scale:
        patches_dir = os.path.join(output_folder, "scale_patches") if save_patches else None
        optimal_scale = get_optimal_scale(
            model, image_path,
            patch_size=window_size, verbose=False,
            save_patches=patches_dir, border_margin=border_margin, log=log,
        )
    else:
        optimal_scale = 1.0
        log("[INFO] Using default scale: 1.0")

    target_h = int(raw_h * optimal_scale)
    target_w = int(raw_w * optimal_scale)

    if optimal_scale != 1.0:
        interp = cv2.INTER_AREA if optimal_scale < 1.0 else cv2.INTER_CUBIC
        full_image = cv2.resize(full_image, (target_w, target_h), interpolation=interp)
        log(f"[INFO] Resized to: {target_w} x {target_h} (scale: {optimal_scale:.3f})")
        resized_bgr = cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_folder, "heatmaps", f"{base_name}_resized.png"), resized_bgr)

    h_orig, w_orig = full_image.shape[:2]

    # CLAHE on the L channel of LAB. Sharpens thin ink on cream paper while
    # leaving color information untouched, which matters when the model was
    # trained on color synthetic data. Applied AFTER auto-scale resize so the
    # statistics match the patch size the model will see.
    if use_clahe:
        lab = cv2.cvtColor(full_image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit,
                                tileGridSize=(clahe_grid_size, clahe_grid_size))
        lab[..., 0] = clahe.apply(lab[..., 0])
        full_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        log(f"[INFO] CLAHE applied (clip={clahe_clip_limit}, grid={clahe_grid_size}).")
        cv2.imwrite(os.path.join(output_folder, "heatmaps", f"{base_name}_clahe.png"),
                    cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))

    img_padded, _, _ = pad_image(full_image, window_size, stride)
    h_pad, w_pad = img_padded.shape[:2]

    prob_map   = np.zeros((h_pad, w_pad), dtype=np.float32)
    weight_map = np.zeros((h_pad, w_pad), dtype=np.float32)
    spline_window = generate_spline_window(window_size).astype(np.float32)

    # Pre-normalise the padded image once instead of per-patch.
    img_padded_f32 = img_padded.astype(np.float32) / 255.0

    all_coords = [
        (y, x)
        for y in range(0, h_pad - window_size[0] + 1, stride)
        for x in range(0, w_pad - window_size[1] + 1, stride)
    ]

    if skip_empty_patches:
        coordinates = [
            (y, x) for (y, x) in all_coords
            if img_padded[y:y + window_size[0], x:x + window_size[1]].std() >= min_patch_std
        ]
        skipped = len(all_coords) - len(coordinates)
        log(f"[INFO] Total patches: {len(all_coords)} | Skipped (empty): {skipped} | Running: {len(coordinates)} | Batch size: {batch_size}")
    else:
        coordinates = all_coords
        log(f"[INFO] Total patches: {len(coordinates)} | Batch size: {batch_size}")

    desc = "Inferencing (TTA)" if use_tta else "Inferencing"
    for i in tqdm(range(0, len(coordinates), batch_size), desc=desc):
        batch_coords = coordinates[i:i + batch_size]
        batch_images = np.array([
            img_padded_f32[y:y + window_size[0], x:x + window_size[1]]
            for (y, x) in batch_coords
        ])

        batch_preds = (
            predict_with_tta(model, batch_images, weights=tta_weights)
            if use_tta
            else model.predict(batch_images, verbose=0)
        )
        batch_preds = batch_preds[..., 0] if batch_preds.ndim == 4 else batch_preds

        for j, pred in enumerate(batch_preds):
            y, x = batch_coords[j]
            prob_map[y:y + window_size[0], x:x + window_size[1]] += pred * spline_window
            weight_map[y:y + window_size[0], x:x + window_size[1]] += spline_window

    weight_map[weight_map == 0] = 1.0
    final_result = (prob_map / weight_map)[:h_orig, :w_orig]

    if optimal_scale != 1.0:
        log(f"[INFO] Rescaling output back to original size: {raw_w} x {raw_h}")
        final_result = cv2.resize(final_result, (raw_w, raw_h), interpolation=cv2.INTER_LINEAR)

    npy_path = os.path.join(output_folder, "npy_raw", f"{base_name}.npy")
    np.save(npy_path, final_result)
    log(f"[INFO] Saved probability map: {npy_path}")

    # Clip before uint8 cast — an unclipped float→uint8 wraps mod 256 if the
    # model lacks a sigmoid head or the bilinear resize introduces over/undershoot.
    heatmap_norm  = (np.clip(final_result, 0.0, 1.0) * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_VIRIDIS)
    heatmap_path  = os.path.join(output_folder, "heatmaps", f"{base_name}_heatmap.png")
    cv2.imwrite(heatmap_path, heatmap_color)
    log(f"[INFO] Saved heatmap: {heatmap_path}")

    metadata = {
        'original_size': [raw_w, raw_h],
        'processed_size': [w_orig, h_orig],
        'scale': float(optimal_scale),
        'window_size': list(window_size),
        'stride': int(stride),
        'use_tta': bool(use_tta),
    }
    metadata_path = os.path.join(output_folder, "npy_raw", f"{base_name}_metadata.json")
    save_json(metadata_path, metadata, indent=2)

    output_parameters = {
        "processed_size": [w_orig, h_orig],
        "scale": float(optimal_scale),
        "npy_path": npy_path,
        "heatmap_path": heatmap_path,
        "metadata_path": metadata_path,
    }

    del model, full_image, prob_map, weight_map
    K.clear_session()
    gc.collect()

    log("[INFO] Prediction complete.")
    return npy_path, output_parameters


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="U-Net road extraction with TTA")
    parser.add_argument("image",  help="Path to input map image")
    parser.add_argument("model",  help="Path to trained .keras model")
    parser.add_argument("output", help="Output directory")
    parser.add_argument("--window-size",   type=int,   default=256)
    parser.add_argument("--stride",        type=int,   default=64)
    parser.add_argument("--batch-size",    type=int,   default=256)
    parser.add_argument("--no-tta",        action="store_true")
    parser.add_argument("--no-autoscale",  action="store_true")
    parser.add_argument("--scale",         type=float, default=None)
    parser.add_argument("--save-patches",  action="store_true")
    parser.add_argument("--border-margin", type=float, default=0.1)
    args = parser.parse_args()

    predict_large_map_tta(
        MODEL_PATH=args.model,
        image_path=args.image,
        output_folder=args.output,
        window_size=(args.window_size, args.window_size),
        stride=args.stride,
        batch_size=args.batch_size,
        use_tta=not args.no_tta,
        auto_scale=not args.no_autoscale,
        fixed_scale=args.scale,
        save_patches=args.save_patches,
        border_margin=args.border_margin,
    )