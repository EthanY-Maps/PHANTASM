"""Hysteresis-thresholded skeletonization with spur pruning, fed by U-Net probability npys."""

import os
import cv2
import numpy as np
from skimage.morphology import skeletonize

try:
    from modules.Road_Graphing.Util import (
        load_and_hysteresis_npy, clean_image, remove_small_objects,
        fill_small_holes, bridge_oriented_gaps, remove_small_skeleton_components,
    )
except ModuleNotFoundError:
    from Util import (
        load_and_hysteresis_npy, clean_image, remove_small_objects,
        fill_small_holes, bridge_oriented_gaps, remove_small_skeleton_components,
    )


def collapse_junction_clusters(skel: np.ndarray, radius: int = 3) -> np.ndarray:
    """Merge clusters of intersection pixels (degree ≥ 3) within `radius` into single pixels.

    Lee's skeletonisation on thick road predictions typically yields T- and Y-
    junctions as small clumps of 2–5 adjacent degree-3 pixels instead of one
    clean junction. Each clump spawns false short edges in the downstream
    graph. We collapse each cluster to its medoid by:
      1. Detecting degree ≥ 3 pixels via a 3×3 neighbour count.
      2. Dilating the junction mask by `radius` and labelling connected regions.
      3. Replacing all cluster pixels in the skeleton with a small filled disk
         at the medoid so the surviving skeleton arms still terminate at one
         shared centre.
    """
    skel_u8 = (skel > 0).astype(np.uint8)
    if not skel_u8.any():
        return skel

    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbour_count = cv2.filter2D(skel_u8, -1, kernel, borderType=cv2.BORDER_CONSTANT) * skel_u8
    junctions = (neighbour_count >= 3).astype(np.uint8)
    if not junctions.any():
        return skel

    dilation_ksize = 2 * radius + 1
    dilated = cv2.dilate(junctions, np.ones((dilation_ksize, dilation_ksize), np.uint8))
    n_labels, labels = cv2.connectedComponents(dilated)

    out = skel_u8.copy() * 255
    for i in range(1, n_labels):
        region = (labels == i) & junctions.astype(bool)
        ys, xs = np.where(region)
        if len(ys) <= 1:
            continue   # single junction pixel — already optimal
        cy, cx = float(np.mean(ys)), float(np.mean(xs))
        # medoid: the cluster pixel closest to the centroid
        best = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
        center_y, center_x = ys[best], xs[best]
        # Clear all cluster pixels, then keep just the medoid lit
        out[region] = 0
        out[center_y, center_x] = 255
        # Reconnect: any neighbour of a cleared pixel that's still skeleton
        # needs a path to the medoid. Draw a thin line in the dilated region.
        for yy, xx in zip(ys, xs):
            if yy == center_y and xx == center_x:
                continue
            cv2.line(out, (int(center_x), int(center_y)), (int(xx), int(yy)), 255, 1)
    return out


def prune_skeleton(skeleton_img, iterations=3):
    """Iteratively strip 1-neighbour endpoints — morphological spur removal.

    Convolution kernel has centre weight 10 + 8×1 neighbours. A pixel value of
    exactly 11 in the filter output means "skeleton pixel with exactly 1 skeleton
    neighbour" → endpoint, which we erase. Repeats `iterations` times, so a
    spur ≤ N pixels long gets fully consumed in N passes.
    """
    pruned = skeleton_img.copy()
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)

    print(f"  - Pruning spurs ({iterations} iterations)...")
    for _ in range(iterations):
        binary = (pruned > 0).astype(np.uint8)
        neighbors = cv2.filter2D(binary, -1, kernel)
        endpoints = (neighbors == 11).astype(np.uint8)
        pruned = pruned - (endpoints * 255)
    return pruned


def process_skeleton(npy_path, output_base,
                     low_thr=0.2, high_thr=0.6, prune_iter=3,
                     max_hole_size=50, gap_bridge_length=7, junction_merge_radius=3,
                     min_object_size=100, min_skeleton_component=20):
    fname = os.path.basename(npy_path)
    base_name = os.path.splitext(fname)[0]

    for d in ("binary_hysteresis", "cleaned", "skeleton_raw", "skeleton_pruned", "overlay"):
        os.makedirs(os.path.join(output_base, d), exist_ok=True)

    binary = load_and_hysteresis_npy(npy_path, low_thr, high_thr)
    cv2.imwrite(os.path.join(output_base, "binary_hysteresis", f"{base_name}.png"), binary)

    cleaned = clean_image(binary, kernel_size=3)
    # U-Net hygiene: bridge short directional gaps in linear features and fill
    # small interior holes before skeletonisation. Both produce dramatically
    # cleaner skeletons by preventing false bifurcations and broken segments.
    if gap_bridge_length > 0:
        cleaned = bridge_oriented_gaps(cleaned, kernel_length=gap_bridge_length)
    if max_hole_size > 0:
        cleaned = fill_small_holes(cleaned, max_hole_size=max_hole_size)
    cleaned = remove_small_objects(cleaned, min_size=min_object_size)
    cv2.imwrite(os.path.join(output_base, "cleaned", f"{base_name}.png"), cleaned)

    print("  - Skeletonizing...")
    skel_bool = skeletonize(cleaned > 0, method='lee')
    skel_raw = (skel_bool.astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(output_base, "skeleton_raw", f"{base_name}.png"), skel_raw)

    skel_pruned = prune_skeleton(skel_raw, iterations=prune_iter)
    # Collapse junction clusters before the graph builder sees them — saves the
    # graph-level `_merge_close_intersections` from having to clean up after
    # Lee's typical "T-junction → 3 adjacent degree-3 pixels" pattern.
    if junction_merge_radius > 0:
        skel_pruned = collapse_junction_clusters(skel_pruned, radius=junction_merge_radius)
    # Drop tiny skeleton islands (typically prediction noise patches that
    # survived `remove_small_objects` because they were larger as binary blobs).
    if min_skeleton_component > 1:
        skel_pruned = remove_small_skeleton_components(skel_pruned, min_pixels=min_skeleton_component)
    cv2.imwrite(os.path.join(output_base, "skeleton_pruned", f"{base_name}.png"), skel_pruned)
    np.save(os.path.join(output_base, "skeleton_pruned", f"{base_name}.npy"), skel_pruned)

    overlay = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    overlay[skel_pruned > 0] = [0, 0, 255]
    cv2.imwrite(os.path.join(output_base, "overlay", f"{base_name}.png"), overlay)

    print(f"Done! Saved to {output_base}")
    return skel_pruned, cleaned


if __name__ == "__main__":
    INPUT_NPY = "data/Output/1903-20-01-2026/Predictions/npy_raw/nla.obj-231540665_edited.npy"
    OUTPUT_DIR = "data/Input/1903/Truth"
    process_skeleton(INPUT_NPY, OUTPUT_DIR, low_thr=0.2, high_thr=0.6, prune_iter=3)
