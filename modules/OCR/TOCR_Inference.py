"""
End-to-End Map OCR Pipeline
Extracts regions from historic maps based on GeoJSON polygons, processes them 
dynamically (PCA/MinArea orientation, buffering, masking, padding), and runs 
conditional dual-inference batch OCR to inject predictions back into the JSON efficiently.
"""
import os
import cv2
import math
import gc
import logging
import warnings
import re
import numpy as np
from collections import Counter
from typing import List, Dict, Optional
from dataclasses import dataclass
from contextlib import contextmanager

import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image
from pdf2image import convert_from_path

from utils import load_json, save_json
from shapely.geometry import Polygon
from shapely.validation import make_valid
from tqdm import tqdm

# --- CONFIGURATION ---
warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── 1. GEOMETRY & IMAGE HELPER FUNCTIONS ──

def load_source_image(image_path: str, target_dpi: int = 300) -> np.ndarray:
    """Loads an image or efficiently converts the first page of a PDF to an OpenCV BGR image."""
    ext = os.path.splitext(image_path)[1].lower()
    
    if ext == '.pdf':
        logger.info(f"Converting PDF (Page 1 only): {image_path} at {target_dpi} DPI")
        pages = convert_from_path(
            image_path, 
            dpi=target_dpi, 
            first_page=1,      
            last_page=1,       
            thread_count=4,    
            grayscale=False
        )
        return cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
        
    else:
        logger.info(f"Loading Image: {image_path}")
        with Image.open(image_path) as pil_img:
            rgb_img = pil_img.convert('RGB')
            return cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)

def simplify_polygon(poly, max_points: int = 12, tolerance_step: float = 1.0):
    """Progressively simplify polygon to remove jitter/curves until ≤ max_points or 10 passes.

    Accepts Polygon or MultiPolygon. `make_valid()` on a self-intersecting input
    typically returns a MultiPolygon; if convex_hull also degenerates to a
    Point/LineString and the caller falls back to the original, we used to
    crash on `.exterior.coords`. Now we pick the largest piece by area.
    """
    if poly.is_empty: return poly
    if poly.geom_type == 'MultiPolygon':
        poly = max(poly.geoms, key=lambda p: p.area)
    if poly.geom_type != 'Polygon':
        return poly  # Point / LineString — caller skips downstream
    current_poly = poly
    for _ in range(10):
        if len(current_poly.exterior.coords) <= max_points: break
        current_poly = current_poly.simplify(tolerance_step, preserve_topology=True)
    return current_poly

def get_optimal_orientation(poly) -> float:
    """
    Calculates rotation angle using PCA for elongated shapes,
    falling back to Minimum Area Rectangle for squarish/irregular shapes.
    """
    if poly.is_empty: return 0.0
    if poly.geom_type != 'Polygon':
        return 0.0  # MultiPolygon/LineString/Point have no single orientation
    pts = np.array(poly.exterior.coords)[:-1]
    if len(pts) < 3: return 0.0

    mean = np.mean(pts, axis=0)
    centered_data = pts - mean
    cov = np.cov(centered_data.T)
    evals, evecs = np.linalg.eig(cov)

    sort_indices = np.argsort(evals)[::-1]
    evals = evals[sort_indices]
    primary_vec = evecs[:, sort_indices[0]]  

    ratio = evals[0] / (evals[1] + 1e-6)

    if ratio < 3.0:
        rect = cv2.minAreaRect(pts.astype(np.float32))
        box = cv2.boxPoints(rect)
        
        edge1 = np.linalg.norm(box[0] - box[1])
        edge2 = np.linalg.norm(box[1] - box[2])
        
        vec = box[1] - box[0] if edge1 > edge2 else box[2] - box[1]
        angle_rad = np.arctan2(vec[1], vec[0])
        angle_deg = np.degrees(angle_rad)
    else:
        angle_rad = np.arctan2(primary_vec[1], primary_vec[0])
        angle_deg = np.degrees(angle_rad)

    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180

    return angle_deg

def rotate_image_expanded(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotates image without cropping corners."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    rad = math.radians(angle)
    sin = math.sin(rad)
    cos = math.cos(rad)

    n_w = int((h * abs(sin)) + (w * abs(cos)))
    n_h = int((h * abs(cos)) + (w * abs(sin)))

    M = cv2.getRotationMatrix2D(center, angle, 1.0) 
    M[0, 2] += (n_w / 2) - center[0]
    M[1, 2] += (n_h / 2) - center[1]

    return cv2.warpAffine(image, M, (n_w, n_h), 
                         flags=cv2.INTER_CUBIC, 
                         borderMode=cv2.BORDER_CONSTANT, 
                         borderValue=(0, 0, 0))

def autocrop_and_pad(image: np.ndarray, pad_margin: int = 15) -> np.ndarray:
    """Tight-crop the post-rotation black borders, then re-pad with white to preserve ascenders/descenders."""
    if image is None or image.size == 0:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return image

    x, y, w, h = cv2.boundingRect(coords)
    tight_crop = image[y:y+h, x:x+w]
    return cv2.copyMakeBorder(
        tight_crop,
        top=pad_margin, bottom=pad_margin, left=pad_margin, right=pad_margin,
        borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255],
    )

# ── 2. OCR CONFIG & PROCESSOR ──

@dataclass
class OCRConfig:
    model_path: str
    device: Optional[str] = None
    batch_size: int = 32
    max_length: int = 32          # caps generation so TrOCR can't run away on noisy crops
    use_mixed_precision: bool = True

class EnhancedBatchOCRProcessor:
    def __init__(self, config: OCRConfig):
        self.config = config
        self.device = self._select_device()
        logger.info(f"Using device: {self.device}")
        
        logger.info(f"Loading model from: {config.model_path}")
        self.processor = TrOCRProcessor.from_pretrained(config.model_path, use_fast=False)
        self.model = VisionEncoderDecoderModel.from_pretrained(config.model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        if hasattr(torch, 'compile') and self.device == 'cuda':
            try:
                self.model = torch.compile(self.model)
                logger.info("Model compilation enabled")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
                
        self.batch_size = config.batch_size

    def _select_device(self) -> str:
        if self.config.device: return self.config.device
        if torch.cuda.is_available(): return "cuda"
        if torch.backends.mps.is_available(): return "mps"
        return "cpu"

    @contextmanager
    def managed_memory(self):
        try:
            yield
        finally:
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            elif self.device == 'mps':
                try: torch.mps.empty_cache()
                except AttributeError: pass
            gc.collect()

    def process_image_batch(self, images: List[Image.Image]) -> List[str]:
        if not images: return []

        try:
            encoding = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                if self.config.use_mixed_precision and self.device == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        generated_ids = self.model.generate(
                            **encoding,
                            max_length=self.config.max_length,
                            num_beams=1,
                            do_sample=False
                        )
                else:
                    generated_ids = self.model.generate(
                        **encoding,
                        max_length=self.config.max_length,
                        num_beams=1,
                        do_sample=False
                    )
            return self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        except RuntimeError as e:
            # Narrow fallback to OOM: anything else (malformed inputs, model
            # config mismatches) deserves to surface, not be silently masked
            # by retrying single-image inference.
            if 'out of memory' not in str(e).lower():
                raise
            logger.warning(
                f"Batch OOM with {len(images)} images on {self.device}; "
                f"falling back to per-image inference"
            )
            return self.process_images_individually(images)

    def process_images_individually(self, images: List[Image.Image]) -> List[str]:
        predictions = []
        for image in images:
            try:
                with self.managed_memory():
                    encoding = self.processor(images=image, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        if self.config.use_mixed_precision and self.device == 'cuda':
                            with torch.autocast(device_type='cuda', dtype=torch.float16):
                                generated_ids = self.model.generate(**encoding, max_length=self.config.max_length)
                        else:
                            generated_ids = self.model.generate(**encoding, max_length=self.config.max_length)
                        prediction = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                    predictions.append(prediction)
            except RuntimeError as e:
                # Per-image OOM is essentially terminal; log clearly and emit
                # an empty so indexing into batch_records stays aligned.
                if 'out of memory' in str(e).lower():
                    logger.error(f"OOM on single image (no fallback left); emitting empty prediction")
                else:
                    logger.error(f"Inference error on single image: {e}")
                predictions.append("")
        return predictions

    def score_ocr_prediction(self, text: str) -> float:
        """Scores text based on map-specific vocabularies to identify upside-down garbage."""
        if not text or len(text.strip()) == 0: 
            return 0.0
            
        text = text.strip()
        score = 10.0  
        
        alnum_count = sum(c.isalnum() for c in text)
        alnum_ratio = alnum_count / len(text)
        score += (alnum_ratio * 10)  
        
        if re.fullmatch(r'\d+[a-zA-Z]?', text):
            score += 15
        elif re.search(r'\d+\s*(ft|ac|acres|ch|m|km|mi)\b', text, re.IGNORECASE):
            score += 15
        elif re.search(r'\b(st|rd|ave|ln|riv|creek|mt|pt)\b', text, re.IGNORECASE):
            score += 10
            
        is_mostly_alpha = sum(c.isalpha() for c in text) > (len(text) / 2)
        if is_mostly_alpha:
            if sum(1 for c in text if c in set("aeiouAEIOU")) > 0: score += 5
            if len(text.split()) > 1: score += 5
                
        counts = Counter(text)
        if len(text) > 5:
            max_char_freq = counts.most_common(1)[0][1] / len(text)
            if max_char_freq > 0.4: score *= 0.3  
                
        score += min(len(text), 20) * 0.5
        return score

# ── 3. ORCHESTRATION PIPELINE ──

def run_map_ocr_pipeline(
    map_path: str, 
    json_path: str, 
    output_json_path: str, 
    model_path: str, 
    buffer_px: int = 10, 
    batch_size: int = 32,
    save_snippets: bool = True,
    snippets_dir: Optional[str] = None,
    confidence_threshold: float = 18.0
):
    """Orchestrates geometry cropping, batch OCR extraction, and JSON updating efficiently."""
    
    if save_snippets:
        if snippets_dir is None:
            snippets_dir = os.path.join(os.path.dirname(output_json_path), "snippets")
        os.makedirs(snippets_dir, exist_ok=True)
        logger.info(f"📁 Saving verification snippets to: {snippets_dir}")

    image = load_source_image(map_path)
    h_img, w_img = image.shape[:2]
    
    data = load_json(json_path)


    config = OCRConfig(
        model_path=model_path,
        batch_size=batch_size,
        max_length=32,
        use_mixed_precision=False,  # disabled in production; see note in OCRConfig
    )
    processor = EnhancedBatchOCRProcessor(config)

    features = data.get('features', [])
    batch_records = []
    
    # Tracking counters
    total_crops = 0
    retry_count = 0
    
    logger.info(f"🚀 Processing {len(features)} map features in batches of {batch_size}...")

    for idx, entry in enumerate(tqdm(features, desc="Extraction & OCR")):
        try:
            coords = entry['geometry']['coordinates'][0]
            raw_poly = Polygon(coords)
            if not raw_poly.is_valid: raw_poly = make_valid(raw_poly)
        except Exception:
            raw_poly = Polygon()
            
        if not raw_poly.is_empty:
            work_poly = raw_poly.convex_hull
            if work_poly.geom_type != 'Polygon': 
                work_poly = raw_poly
                
            simple_poly = simplify_polygon(work_poly, max_points=10)
            buffered_poly = simple_poly.buffer(buffer_px)
            
            minx, miny, maxx, maxy = buffered_poly.bounds
            minx, miny = max(int(minx), 0), max(int(miny), 0)
            maxx, maxy = min(int(maxx), w_img), min(int(maxy), h_img)
            
            if minx < maxx and miny < maxy:
                crop = image[miny:maxy, minx:maxx].copy()
                
                if crop.size > 0:
                    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
                    parts = [buffered_poly] if buffered_poly.geom_type == 'Polygon' else list(buffered_poly.geoms)
                    for part in parts:
                        if part.is_empty: continue
                        local_coords = np.array(part.exterior.coords) - [minx, miny]
                        cv2.fillPoly(mask, [local_coords.astype(np.int32)], 255)
                        
                    masked_crop = cv2.bitwise_and(crop, crop, mask=mask)
                    
                    rot_angle = get_optimal_orientation(simple_poly)
                    rotated_img = rotate_image_expanded(masked_crop, rot_angle)
                    final_img = autocrop_and_pad(rotated_img, pad_margin=15)
                    
                    if final_img is not None and final_img.size > 0:
                        final_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(final_img_rgb)
                        
                        batch_records.append({
                            "feature_idx": idx, 
                            "image": pil_img,
                            "cv2_image": final_img 
                        })
                        total_crops += 1

        # Trigger conditional batch OCR
        if len(batch_records) >= batch_size or idx == len(features) - 1:
            if not batch_records:
                continue
                
            images = [rec["image"] for rec in batch_records]
            
            # Pass 1: Normal Inference
            with processor.managed_memory():
                normal_preds = processor.process_image_batch(images)
            
            retry_indices = []
            retry_images = []
            final_preds = []
            final_scores = []
            
            for i, p_norm in enumerate(normal_preds):
                clean_norm = "".join(c for c in p_norm if c.isalnum() or c in (' ', '-', '_')).strip()
                score_norm = processor.score_ocr_prediction(clean_norm)
                
                final_preds.append(p_norm)
                final_scores.append(score_norm)
                
                # Queue for 180-degree retry if below confidence threshold
                if score_norm < confidence_threshold:
                    retry_indices.append(i)
                    retry_images.append(images[i].transpose(Image.ROTATE_180))
            
            # Pass 2: Conditional 180-Degree Retry
            if retry_images:
                retry_count += len(retry_images)
                with processor.managed_memory():
                    rotated_preds = processor.process_image_batch(retry_images)
                    
                for retry_idx, p_rot in zip(retry_indices, rotated_preds):
                    clean_rot = "".join(c for c in p_rot if c.isalnum() or c in (' ', '-', '_')).strip()
                    score_rot = processor.score_ocr_prediction(clean_rot)
                    
                    if score_rot > final_scores[retry_idx]:
                        final_preds[retry_idx] = p_rot
                        final_scores[retry_idx] = score_rot

            # Inject and Cleanup
            for rec, best_pred, best_score in zip(batch_records, final_preds, final_scores):
                f_idx = rec["feature_idx"]
                
                if "properties" not in data["features"][f_idx]:
                    data["features"][f_idx]["properties"] = {}
                data["features"][f_idx]["properties"]["predicted_text"] = best_pred

                if save_snippets:
                    # Strip for safe filename generation only
                    safe_pred = "".join(c for c in best_pred if c.isalnum() or c in (' ', '-', '_')).strip()
                    safe_pred = safe_pred[:50] if safe_pred else "EMPTY"
                    snippet_filename = os.path.join(snippets_dir, f"idx_{f_idx:05d}_{safe_pred}.png")
                    cv2.imwrite(snippet_filename, rec["cv2_image"])
            
            batch_records.clear()
            gc.collect()

    save_json(output_json_path, data, indent=2)


    # Final Reporting
    if total_crops > 0:
        retry_percentage = (retry_count / total_crops) * 100
        logger.info(f"📊 Stats: Processed {total_crops} text crops.")
        logger.info(f"🔄 Conditional OCR triggered on {retry_count} crops ({retry_percentage:.1f}% required 180-degree checks).")
        
    logger.info(f"✅ Pipeline complete. Updated JSON saved to: {output_json_path}")
    return output_json_path

# ── 4. EXECUTION ──
if __name__ == "__main__":
    INPUT_MAP = "data/Output/Canterbury_24-02-26/1083247_1193-10-Canterbury.png"
    INPUT_JSON = "data/Output/Canterbury_24-02-26/Text Detection/detection_results.json"
    MODEL_PATH = "models/trocr_street_names"
    OUTPUT_JSON = "data/Output/Canterbury_24-02-26/OCR/1083247_1193-10-Canterbury_with_text.json"
    
    run_map_ocr_pipeline(
        map_path=INPUT_MAP,
        json_path=INPUT_JSON,
        output_json_path=OUTPUT_JSON,
        model_path=MODEL_PATH,
        buffer_px=10,
        batch_size=32,
        save_snippets=True
    )