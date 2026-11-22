"""Shared utilities for the historic map pipeline."""

import os
import sys
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from functools import wraps
from contextlib import contextmanager
import json

from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError, PDFInfoNotInstalledError

# Historic maps frequently exceed PIL's default decompression-bomb limit.
try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
except ImportError:
    pass


IMAGE_EXTENSIONS = ('.png', '.tif', '.tiff', '.jpg', '.jpeg', '.jp2', '.bmp', '.gif', '.pdf')


def find_image_file(path: str) -> Optional[str]:
    """Return the path itself if it's an image, or the first image in the directory, else None."""
    path_obj = Path(path)
    if not path_obj.exists():
        return None
    if path_obj.is_file() and path_obj.suffix.lower() in IMAGE_EXTENSIONS:
        return str(path_obj)
    if path_obj.is_dir():
        for file in path_obj.iterdir():
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                return str(file)
    return None


def validate_path(path: str, path_type: str = "folder") -> bool:
    """`path_type` is "folder", "file", or "any" (file-or-folder)."""
    if not path or not path.strip():
        return False
    path_obj = Path(path)
    if path_type == "folder":
        return path_obj.exists() and path_obj.is_dir()
    elif path_type == "any":
        return path_obj.exists() and (
            path_obj.is_dir() or
            (path_obj.is_file() and path_obj.suffix.lower() in IMAGE_EXTENSIONS)
        )
    return path_obj.exists() and path_obj.is_file()


def collect_image_files(input_path: str) -> list:
    """Returns absolute paths of all image files under `input_path` (recursive for dirs)."""
    path = Path(input_path)
    if path.is_file():
        return [str(path)] if path.suffix.lower() in IMAGE_EXTENSIONS else []
    if path.is_dir():
        return sorted(
            str(f) for f in path.rglob('*')
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
    return []


def parse_bounding_coordinates(coord1_str: str, coord2_str: str) -> List[tuple]:
    """Each input is "x,y"; returns the 4 corners of the rectangle they span."""
    coord1_x, coord1_y = map(float, coord1_str.split(','))
    coord2_x, coord2_y = map(float, coord2_str.split(','))
    return [
        (coord1_x, coord1_y),
        (coord1_x, coord2_y),
        (coord2_x, coord2_y),
        (coord2_x, coord1_y),
    ]


@contextmanager
def timed_operation(operation_name: str, log_callback: Optional[Callable[[str], None]] = None):
    start_time = time.time()
    log = log_callback or print
    try:
        yield start_time
    finally:
        elapsed = time.time() - start_time
        log(f"[TIMING] {operation_name} completed in {elapsed:.2f}s")


def timed_function(stage_name: str):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            log_callback = None
            if args and hasattr(args[0], 'log'):
                log_callback = args[0].log
            elif 'log_callback' in kwargs:
                log_callback = kwargs['log_callback']
            else:
                log_callback = print

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                log_callback(f"[TIMING] {stage_name} completed in {time.time() - start_time:.2f}s")
                return result
            except Exception as e:
                log_callback(f"[ERROR] {stage_name} failed after {time.time() - start_time:.2f}s: {e}")
                raise
        return wrapper
    return decorator


def execute_subprocess(
    command: list,
    log_callback: Optional[Callable[[str], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> None:
    log = log_callback or print
    update_status = status_callback or (lambda x: None)

    log(f"[EXEC] Running: {' '.join(command)}")
    start_time = time.time()

    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for line in process.stdout:
            log(f"[SUBPROCESS] {line.strip()}")
            print(f"[SUBPROCESS] {line.strip()}", file=sys.stderr)

        return_code = process.wait()
        elapsed = time.time() - start_time

        if return_code == 0:
            log(f"[SUBPROCESS] Completed successfully in {elapsed:.2f} seconds")
            update_status(f"Subprocess completed in {elapsed:.2f} seconds")
        else:
            raise subprocess.CalledProcessError(return_code, command)

    except subprocess.CalledProcessError as e:
        log(f"[ERROR] Subprocess failed after {time.time() - start_time:.2f} seconds: {e}")
        raise


def setup_output_directories(
    output_dir: str,
    expt_name: str,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Path]:
    """Create the standard per-experiment output folder layout."""
    log = log_callback or print
    base_output = Path(output_dir) / expt_name

    dirs = {
        'base_output':  base_output,
        'predictions':  base_output / 'Road Extraction',
        'graphing':     base_output / 'Vectorisation',
        'text_output':  base_output / 'Text Detection',
        'geoloc':       base_output / 'Geolocation',
        'cutouts':      base_output / 'OCR Cutouts',
        'plots':        base_output / 'Plots',
        'ocr_results':  base_output / 'OCR Results',
        'similarity':   base_output / 'Similarity',
    }
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    log(f"[INFO] Created output directories in {base_output}")
    return dirs


def get_map_name_from_input(input_path: str) -> str:
    """Returns the filename stem of `input_path` (or of the first image inside, if a dir)."""
    path_obj = Path(input_path)
    if path_obj.is_file() and path_obj.suffix.lower() in IMAGE_EXTENSIONS:
        return path_obj.stem
    image_path = find_image_file(input_path)
    if image_path:
        return Path(image_path).stem
    return "unknown_map"


def load_json(path) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path, data: Any, indent: int = 2) -> None:
    """`ensure_ascii=False` so non-Latin characters round-trip readably."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def deep_merge_missing(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively add keys from `source` to `target` when missing. Mutates and returns `target`."""
    for key, value in source.items():
        if key not in target:
            target[key] = value
        elif isinstance(value, dict) and isinstance(target[key], dict):
            deep_merge_missing(target[key], value)
    return target


def read_local_config_output(local_config_path: str, section: str, key: str = None) -> Any:
    """Read from a module's `Output Parameter` block in `Local_config.json`."""
    config = load_json(local_config_path)
    output = config[section]['Output Parameter']
    return output[key] if key is not None else output


def write_local_config_output(local_config_path: str, section: str, updates: Dict[str, Any]) -> None:
    """Merge updates into a module's `Output Parameter` block in `Local_config.json`."""
    config = load_json(local_config_path)
    config[section]['Output Parameter'].update(updates)
    save_json(local_config_path, config, indent=4)


def convert_pdf_to_png(pdf_path, output_path=None, dpi=300, page_num=0):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        # pdf2image uses 1-based page indexing; preserve caller's 0-based API.
        images = convert_from_path(
            pdf_path, dpi=dpi, first_page=page_num + 1, last_page=page_num + 1,
        )
        if not images:
            raise ValueError(f"Page number {page_num} could not be rendered.")
        page_image = images[0]
    except PDFPageCountError:
        raise ValueError(f"Page number {page_num} is out of range for this document.")
    except PDFInfoNotInstalledError:
        raise EnvironmentError(
            "Poppler is not installed or not in PATH. It is required for pdf2image."
        )

    if output_path is None:
        temp_fd, output_path = tempfile.mkstemp(suffix='.png', prefix='pdf_converted_')
        os.close(temp_fd)

    page_image.save(output_path, format="PNG")
    return output_path
