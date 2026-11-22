from PIL import Image
import os
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import multiprocessing as mp
from typing import Tuple, List, Optional, Literal



# -------------------------- Grid & padding ---------------------------

def _positions_for_dim(dim: int, tile: int, stride: int) -> List[int]:
    """
    Strict, duplicate-free start positions:
      0, stride, 2*stride, ..., (dim - tile) (if not already included)
    """
    if dim <= tile:
        return [0]
    pos = list(range(0, dim - tile + 1, stride))
    last = dim - tile
    if not pos or pos[-1] != last:
        pos.append(last)
    return pos

def get_tile_coordinates_with_overlap(
    width: int,
    height: int,
    tile_size: int,
    overlap_px: int,
) -> Tuple[List[Tuple[int, int, int, int, int, int]], int, int]:
    """
    Returns list of (i, j, left, top, right, bottom) & counts.
    No clamping, no dedup required.
    """
    overlap_px = max(0, min(int(overlap_px), tile_size - 1))
    stride = max(1, tile_size - overlap_px)

    xs = _positions_for_dim(width, tile_size, stride)
    ys = _positions_for_dim(height, tile_size, stride)

    coords: List[Tuple[int, int, int, int, int, int]] = []
    for i, left in enumerate(xs):
        right = left + tile_size
        for j, top in enumerate(ys):
            bottom = top + tile_size
            coords.append((i, j, left, top, right, bottom))

    return coords, len(xs), len(ys)

def _pad_tile_numpy(tile_img, target_size: Tuple[int, int], mode: Literal["constant","edge","reflect"]="edge"):
    """
    Pad a PIL tile to target_size using numpy padding:
      - 'edge' (replicate), great to avoid black seams
      - 'reflect' (mirror)
      - 'constant' (zeros)
    """
    import numpy as np
    from PIL import Image

    tw, th = tile_img.size
    W, H = target_size
    if (tw, th) == (W, H):
        return tile_img

    arr = np.array(tile_img)
    pad_w = W - tw
    pad_h = H - th

    # right/bottom padding only (tiles always cropped from left/top)
    pad_width = None
    if arr.ndim == 2:  # L/gray
        pad_width = ((0, pad_h), (0, pad_w))
    else:  # RGB / RGBA
        pad_width = ((0, pad_h), (0, pad_w), (0, 0))

    if mode == "constant":
        arr_p = np.pad(arr, pad_width, mode="constant", constant_values=0)
    else:
        # numpy's 'edge'/'reflect' behave nicely for seams
        arr_p = np.pad(arr, pad_width, mode=mode)

    return Image.fromarray(arr_p)

# ------------------------------ IO tile ------------------------------

def extract_and_save_tile(
    img: Image.Image,
    coords: Tuple[int, int, int, int, int, int],
    output_path: str,
    tile_size: int,
    target_size: Tuple[int, int],
    pad_mode: Literal["constant","edge","reflect"]="edge",
) -> bool:
    try:
        _i, _j, left, top, right, bottom = coords
        tile = img.crop((left, top, right, bottom))
        if tile.size != target_size:
            tile = _pad_tile_numpy(tile, target_size, mode=pad_mode)
        # PNG: optimize+compress_level for smaller files without going wild
        tile.save(output_path, optimize=True, compress_level=6)
        return True
    except Exception as e:
        print(f"Error processing tile {output_path}: {e}")
        return False

# --------------------------- Main tiler ------------------------------

def tile_image_optimized(
    image_path: str,
    output_dir: str,
    tile_size: int,
    max_workers: Optional[int] = None,
    overlap_pct: float = 0.05,
    pad_mode: Literal["constant","edge","reflect"]="edge",
) -> Optional[str]:
    """
    Optimized image tiling with overlap and edge-safe padding.
    """
    try:
        with Image.open(image_path) as img:
            if img.mode == "F":
                img = img.convert("L")

            width, height = img.size

            print(f"Processing image: {Path(image_path).name} ({width}x{height})")

            overlap_px = max(1, int(round(tile_size * float(overlap_pct))))
            print(f"Tile overlap: {overlap_px} px ({overlap_pct*100:.1f}%)")

            coords, n_x, n_y = get_tile_coordinates_with_overlap(width, height, tile_size, overlap_px)
            total_tiles = len(coords)
            print(f"Will create {total_tiles} tiles ({n_x} × {n_y})")

            os.makedirs(output_dir, exist_ok=True)

            # Create a copy AFTER all prints to keep context manager short
            img_copy = img.copy()

        # Threaded crop+save (I/O bound). Pillow reads are fine from a shared copy.
        max_workers = max_workers or min(mp.cpu_count(), 8)
        successes = 0

        def _job(c):
            i, j, *_ = c
            out = os.path.join(output_dir, f"tile_{i}_{j}.png")
            return extract_and_save_tile(img_copy, c, out, tile_size, (tile_size, tile_size), pad_mode=pad_mode)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_job, coords))
        successes = sum(1 for r in results if r)

        print(f"Successfully created {successes}/{total_tiles} tiles")
        return output_dir if successes > 0 else None

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# --------------------------- Batch tiling ----------------------------

def tile_images_in_folder(
    folder_path: str,
    output_dir: str,
    max_workers: Optional[int] = 4,
    overlap_pct: float = 0.05,
    pad_mode: Literal["constant","edge","reflect"]="edge",
    tile_size: int = 256,
) -> List[str]:
    """
    Process multiple images in parallel.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory.")
        return []

    supported = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".webp"}
    images = [f for f in folder.iterdir() if f.suffix.lower() in supported]
    if not images:
        print("No supported image files found.")
        return []

    print(f"Found {len(images)} image files to process")

    max_workers = max_workers or min(mp.cpu_count(), 8)
    outputs: List[str] = []

    def _run(img_path: Path):
        return tile_image_optimized(
            str(img_path),
            output_dir,
            max_workers=None,          # internal control
            overlap_pct=overlap_pct,
            pad_mode=pad_mode,
            tile_size=tile_size,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for res in ex.map(_run, images):
            if res is not None:
                outputs.append(res)

    print(f"Successfully processed {len(outputs)}/{len(images)} images")
    return outputs

# ---------------------- Utilities / plumbing ------------------------

def estimate_memory_usage(image_path: str) -> float:
    """
    Estimate RAM (MB) ~ 3x image size in memory (img + copy + buffers).
    """
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            ch = len(img.getbands())
            return (w * h * ch * 3) / (1024 * 1024)
    except Exception:
        return 0.0

def tile_large_image_chunked(
    image_path: str,
    output_dir: str,
    chunk_size: int = 4096,
    overlap: int = 0,
) -> Optional[str]:
    """
    Placeholder for extreme sizes; not implemented here.
    """
    print("tile_large_image_chunked: Not implemented.")
    return None

# ------------------------------ CLI ---------------------------------

if __name__ == "__main__":
    Image.MAX_IMAGE_PIXELS = None

    folder_path = 'data/Input/1903'
    output_dir = 'data/Output/Paper Stuff/1903_Tiles'

    overlap_pct = 0
    print(f"Available CPU cores: {mp.cpu_count()}")

    # crude RAM estimate to throttle workers if needed
    folder_obj = Path(folder_path)
    max_workers = None
    if folder_obj.is_dir():
        total_mb = 0.0
        for f in folder_obj.iterdir():
            if f.suffix.lower() in {'.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp', '.webp'}:
                total_mb += estimate_memory_usage(str(f))
        print(f"Estimated memory usage: {total_mb:.1f} MB")
        if total_mb > 2000:
            max_workers = 2
            print("Large images detected, using conservative parallelism")

    tile_images_in_folder(folder_path, output_dir, max_workers=max_workers, overlap_pct=overlap_pct, pad_mode="edge", tile_size=1028)