import os
import cv2
import numpy as np
import tensorflow as tf
from model import build_dbnet_plusplus_model
import pyclipper
from shapely.geometry import Polygon
from tqdm import tqdm
from PIL import Image
from typing import Callable, List, Optional, Tuple

from utils import save_json

try:
    from pdf2image import convert_from_path
except ImportError:
    pass


def load_image_as_numpy(image_path: str) -> np.ndarray:
    ext = os.path.splitext(image_path)[1].lower()
    if ext == '.pdf':
        pages = convert_from_path(image_path, dpi=300)
        return np.array(pages[0])
    Image.MAX_IMAGE_PIXELS = None
    return np.array(Image.open(image_path).convert('RGB'))


def _coverage_pad(dim: int, window: int, stride: int) -> int:
    """Minimum padding so a `stride`-spaced sliding window covers all of [0, dim).

    Padding to a multiple of `window` (the old approach) left the loop —
    which steps by `stride < window` — overshooting the array on the last
    iteration, producing patches that were mostly synthetic reflection.
    """
    if dim <= window:
        return window - dim
    extra = (dim - window) % stride
    return 0 if extra == 0 else stride - extra


def pad_image_symmetrically(image: np.ndarray, patch_size: int, stride: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Symmetric reflection padding sized so a sliding window with `stride` fully covers the image."""
    h, w = image.shape[:2]
    pad_h_total = _coverage_pad(h, patch_size, stride)
    pad_w_total = _coverage_pad(w, patch_size, stride)
    top    = pad_h_total // 2
    bottom = pad_h_total - top
    left   = pad_w_total // 2
    right  = pad_w_total - left
    padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    return padded, (top, bottom, left, right)


# --- POLYGON EXTRACTION ---
def polygons_from_bitmap(
    pred_map: np.ndarray,
    confidence_thresh: float,
    min_area: int = 50,
    unshrink_ratio: float = 1.5,
) -> List[np.ndarray]:
    """
    Extracts and expands polygons from a probability heatmap.

    Args:
        pred_map: Float probability map [0, 1].
        confidence_thresh: Binarisation threshold.
        min_area: Minimum contour area in pixels to keep.
        unshrink_ratio: Pyclipper expansion ratio.
    """
    bitmap = (pred_map > confidence_thresh).astype(np.uint8)
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        poly = contour[:, 0, :]
        shapely_poly = Polygon(poly)

        # Self-intersecting contours produce invalid polygons whose `.area` may
        # be 0 or negative — a negative area would make `offset` negative and
        # pyclipper would shrink the polygon instead of expanding it. Heal via
        # `buffer(0)` (standard shapely idiom) and skip anything still broken.
        if not shapely_poly.is_valid:
            shapely_poly = shapely_poly.buffer(0)
        if shapely_poly.is_empty or shapely_poly.area <= 0 or shapely_poly.length == 0:
            continue

        offset = shapely_poly.area * unshrink_ratio / shapely_poly.length
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(poly.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = pco.Execute(offset)

        if not expanded:
            continue

        boxes.append(np.array(expanded[0], dtype=np.float32))

    return boxes


# --- POLYGON NMS ---
def nms_polygons(polygons: List[np.ndarray], iou_threshold: float = 0.3) -> List[np.ndarray]:
    """
    Two-pass polygon NMS:
      1. Vectorised AABB IoU as a fast pre-filter.
      2. Exact polygon IoU (Shapely) only for candidates that fail the AABB test.

    This catches the rotated/elongated text case where two polygons have
    overlapping bounding boxes but are geometrically distinct, without paying
    Shapely cost for the (typical) majority that don't overlap at all.

    Args:
        polygons: List of polygon arrays, each shape (N, 2) float32.
        iou_threshold: Polygons overlapping the kept polygon above this IoU are suppressed.

    Returns:
        Filtered list of polygon arrays.
    """
    if len(polygons) <= 1:
        return polygons

    # Axis-aligned bounding boxes: [x1, y1, x2, y2]
    boxes = np.array(
        [[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in polygons],
        dtype=np.float32,
    )
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Cache Shapely polygons lazily to amortise the repeated-conversion cost.
    shapely_cache: List[Optional[Polygon]] = [None] * len(polygons)

    def _shapely_for(idx: int) -> Polygon:
        if shapely_cache[idx] is None:
            poly = Polygon(polygons[idx])
            if not poly.is_valid:
                poly = poly.buffer(0)
            shapely_cache[idx] = poly
        return shapely_cache[idx]

    # Process largest-first so the winner of each suppression step is the biggest box
    order = np.argsort(areas)[::-1]

    kept = []
    while len(order) > 0:
        i = order[0]
        kept.append(i)

        if len(order) == 1:
            break

        rest = order[1:]

        # Pass 1: vectorised AABB IoU
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        aabb_iou = inter / np.maximum(union, 1e-6)

        # Pass 2: confirm AABB-flagged suppressions with exact polygon IoU
        suppressed_mask = aabb_iou > iou_threshold
        if suppressed_mask.any():
            poly_i = _shapely_for(i)
            area_i = poly_i.area
            for k, idx_in_rest in enumerate(rest):
                if not suppressed_mask[k]:
                    continue
                poly_j = _shapely_for(int(idx_in_rest))
                inter_area = poly_i.intersection(poly_j).area
                union_area = area_i + poly_j.area - inter_area
                poly_iou = inter_area / max(union_area, 1e-6)
                if poly_iou <= iou_threshold:
                    suppressed_mask[k] = False  # AABB false positive, keep it

        order = rest[~suppressed_mask]

    return [polygons[i] for i in kept]


# --- BATCH INFERENCE ---
def process_batch(
    model,
    batch_imgs: List[np.ndarray],
    batch_locs: List[Tuple[int, int, int, int]],
    prob_map_full: np.ndarray,
    count_map_full: np.ndarray,
) -> None:
    batch_tensor = np.array(batch_imgs, dtype=np.float32) / 255.0
    preds = model.predict(batch_tensor, verbose=0)
    # Channel 0 = probability map (P); channel 1 = threshold map (T); channel 2 = binary map (B̂)
    probs = preds[..., 0]
    for i, prob_tile in enumerate(probs):
        x, y, valid_w, valid_h = batch_locs[i]
        prob_map_full[y:y + valid_h, x:x + valid_w] += prob_tile[:valid_h, :valid_w]
        count_map_full[y:y + valid_h, x:x + valid_w] += 1.0


# --- MAIN INFERENCE PIPELINE ---
def predict_multi_scale(
    model_path: str,
    image_path: str,
    output_dir: str,
    batch_size: int,
    patch_size: int,
    stride: int,
    confidence_thresh: float,
    scales: List[float],
    min_area: int = 50,
    unshrink_ratio: float = 1.5,
    nms_iou_threshold: float = 0.3,
    log: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Runs multi-scale DBNet++ inference on a map image.

    Args:
        model_path: Path to trained model weights.
        image_path: Path to input image (PNG/JPG/TIFF/PDF).
        output_dir: Directory for output files.
        batch_size: Number of patches per GPU batch.
        patch_size: Square patch dimension in pixels.
        stride: Sliding window stride (overlap = patch_size - stride).
        confidence_thresh: Probability threshold for binarisation.
        scales: List of scale factors for multi-scale inference.
        min_area: Minimum contour area (px²) to retain.
        unshrink_ratio: Pyclipper polygon expansion ratio.
        nms_iou_threshold: IoU threshold for duplicate polygon removal.
        log: Optional logging callback. Defaults to print.

    Returns:
        Path to the saved detection_results.json.
    """
    if log is None:
        log = print

    log(f"[INFO] Loading map: {image_path}")
    full_np = load_image_as_numpy(image_path)
    orig_h, orig_w = full_np.shape[:2]
    log(f"[INFO] Original dimensions: {orig_w}x{orig_h}")

    model = build_dbnet_plusplus_model(input_size=patch_size)
    model.load_weights(model_path)
    log(f"[INFO] Model loaded from {model_path}")

    scale_probability_maps = []
    scale_scores = []

    for scale_val in scales:
        target_w = int(orig_w * scale_val)
        target_h = int(orig_h * scale_val)
        log(f"[INFO] Processing scale {scale_val}x ({target_w}x{target_h})")

        scaled_np = (
            cv2.resize(full_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            if scale_val != 1.0
            else full_np.copy()
        )

        padded_np, (top_pad, bot_pad, left_pad, right_pad) = pad_image_symmetrically(scaled_np, patch_size, stride)
        pad_h, pad_w = padded_np.shape[:2]

        prob_map_full  = np.zeros((pad_h, pad_w), dtype=np.float32)
        count_map_full = np.zeros((pad_h, pad_w), dtype=np.float32)

        # Coverage padding above guarantees these ranges cover the whole image
        # without ever producing a patch that overshoots the padded array.
        y_steps = range(0, pad_h - patch_size + 1, stride)
        x_steps = range(0, pad_w - patch_size + 1, stride)
        total_patches = len(y_steps) * len(x_steps)

        pbar = tqdm(total=total_patches, desc=f"Inference {scale_val}x")
        batch_imgs, batch_locs = [], []

        for y in y_steps:
            for x in x_steps:
                crop = padded_np[y:y + patch_size, x:x + patch_size]
                batch_imgs.append(crop)
                batch_locs.append((x, y, patch_size, patch_size))

                if len(batch_imgs) >= batch_size:
                    process_batch(model, batch_imgs, batch_locs, prob_map_full, count_map_full)
                    pbar.update(len(batch_imgs))
                    batch_imgs, batch_locs = [], []

        if batch_imgs:
            process_batch(model, batch_imgs, batch_locs, prob_map_full, count_map_full)
            pbar.update(len(batch_imgs))
        pbar.close()

        # Crop padding, average overlapping patches
        bot_slice = pad_h - bot_pad if bot_pad > 0 else pad_h
        right_slice = pad_w - right_pad if right_pad > 0 else pad_w
        prob_cropped  = prob_map_full[top_pad:bot_slice, left_pad:right_slice]
        count_cropped = count_map_full[top_pad:bot_slice, left_pad:right_slice]
        scale_prob = prob_cropped / (count_cropped + 1e-6)

        if scale_val == 1.0:
            aligned_prob = scale_prob
        else:
            # INTER_AREA for downsampling (scale > 1 produced a larger map than
            # original), INTER_LINEAR for upsampling. Matches Pred.py:258.
            interp = cv2.INTER_AREA if scale_val > 1.0 else cv2.INTER_LINEAR
            aligned_prob = cv2.resize(scale_prob, (orig_w, orig_h), interpolation=interp)
        scale_probability_maps.append(aligned_prob)

        score = float(np.sum(aligned_prob[aligned_prob > confidence_thresh]))
        scale_scores.append(score)
        log(f"[INFO] Scale {scale_val}x confidence score: {score:.2f}")

    # --- Heatmap fusion (confidence-weighted average) ---
    log("[INFO] Fusing heatmaps...")
    total_score = sum(scale_scores)
    weights = (
        [s / total_score for s in scale_scores]
        if total_score > 0
        else [1.0 / len(scales)] * len(scales)
    )
    for scale_val, w in zip(scales, weights):
        log(f"[INFO]   Scale {scale_val}x → weight {w:.3f}")

    master_prob = np.zeros((orig_h, orig_w), dtype=np.float32)
    for w, prob_map in zip(weights, scale_probability_maps):
        master_prob += w * prob_map

    # --- Polygon extraction + NMS ---
    log("[INFO] Extracting polygons...")
    raw_polygons = polygons_from_bitmap(master_prob, confidence_thresh, min_area, unshrink_ratio)
    log(f"[INFO] {len(raw_polygons)} raw polygons before NMS")

    final_polygons = nms_polygons(raw_polygons, iou_threshold=nms_iou_threshold)
    log(f"[INFO] {len(final_polygons)} polygons after NMS (threshold={nms_iou_threshold})")

    # --- Save outputs ---
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    prob_bitmap = (np.clip(master_prob, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_prob_map.png"), prob_bitmap)

    json_path = os.path.join(output_dir, "detection_results.json")
    feature_collection = {"type": "FeatureCollection", "name": base_name, "features": []}
    for i, poly in enumerate(final_polygons):
        points = poly.astype(np.int32).tolist()
        if points[0] != points[-1]:
            points.append(points[0])
        feature_collection["features"].append({
            "type": "Feature",
            "properties": {"id": i},
            "geometry": {"type": "Polygon", "coordinates": [points]},
        })
    save_json(json_path, feature_collection, indent=4)
    log(f"[INFO] Saved {len(final_polygons)} detections to {json_path}")

    matched_img_bgr = cv2.cvtColor(full_np, cv2.COLOR_RGB2BGR)
    draw_polys = [p.astype(np.int32) for p in final_polygons]
    cv2.drawContours(matched_img_bgr, draw_polys, -1, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_overlay.jpg"), matched_img_bgr)

    log("[INFO] Text detection complete")
    return json_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DBNet++ multi-scale text detection")
    parser.add_argument("image",  help="Path to input map image or PDF")
    parser.add_argument("model",  help="Path to model weights (.keras)")
    parser.add_argument("output", help="Output directory")
    parser.add_argument("--patch-size",   type=int,   default=256)
    parser.add_argument("--batch-size",   type=int,   default=128)
    parser.add_argument("--stride",       type=int,   default=176)
    parser.add_argument("--confidence",   type=float, default=0.2)
    parser.add_argument("--scales",       type=float, nargs="+", default=[1.0, 0.8, 0.5])
    parser.add_argument("--min-area",     type=int,   default=50)
    parser.add_argument("--nms-iou",      type=float, default=0.3)
    args = parser.parse_args()

    predict_multi_scale(
        args.model, args.image, args.output,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        stride=args.stride,
        confidence_thresh=args.confidence,
        scales=args.scales,
        min_area=args.min_area,
        nms_iou_threshold=args.nms_iou,
    )