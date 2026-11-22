import os
import cv2
import numpy as np
import torch
import re
import shutil
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from rapidfuzz import process, fuzz

# --- 1. CONFIGURATION ---
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"

MODEL_PATH = 'data/Training Sets/OCR/trocr_tuning_output/final_model'
GAZETTEER_PATH = 'data/Training Sets/OCR/street_names.txt'

# Directory containing the crops you want to test
TEST_IMAGE_DIR = 'data/Training Sets/OCR/Test_Images'
# Directory where the renamed copies will be saved
RENAMED_OUTPUT_DIR = 'data/Training Sets/OCR/Test_Images_Renamed'

CONFIDENCE_THRESHOLD = 75.0

os.makedirs(RENAMED_OUTPUT_DIR, exist_ok=True)

# --- 2. HISTORICAL LEXICON PROCESSOR ---
class HistoricalLexiconProcessor:
    def __init__(self, gazetteer_path, threshold=80.0):
        self.lexicon = []
        self.threshold = threshold
        self.load_gazetteer(gazetteer_path)
        self.normalized_lexicon = {self._normalize(s): s for s in self.lexicon}

    def load_gazetteer(self, path):
        try:
            with open(path, mode='r', encoding='utf-8') as file:
                for line in file:
                    cleaned_line = re.sub(r'\\s*', '', line).strip()
                    if cleaned_line:
                        self.lexicon.append(cleaned_line)
            print(f"Loaded {len(self.lexicon)} street names from gazetteer.")
        except Exception as e:
            print(f"Warning: Could not load gazetteer from {path}. Error: {e}")

    def _normalize(self, text):
        text = text.lower()
        text = text.replace("st.", "street").replace(" st", " street")
        text = text.replace("rd.", "road").replace(" rd", " road")
        text = text.replace("ave.", "avenue").replace(" ave", " avenue")
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    def correct_text(self, ocr_prediction):
        if not self.lexicon: 
            return ocr_prediction, 0.0
            
        norm_prediction = self._normalize(ocr_prediction)
        
        result = process.extractOne(
            norm_prediction, 
            list(self.normalized_lexicon.keys()), 
            scorer=fuzz.token_sort_ratio
        )
        
        if not result: 
            return ocr_prediction, 0.0
            
        best_match_norm, score, _ = result

        if score >= self.threshold:
            return self.normalized_lexicon[best_match_norm], score
        else:
            return ocr_prediction, score

# --- 3. IMAGE PRE-PROCESSING ---
def apply_clahe_to_pil(image):
    cv_img = np.array(image)
    if len(cv_img.shape) == 2:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
    else:
        cv_img = cv_img[:, :, ::-1].copy()

    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return Image.fromarray(enhanced_bgr[:, :, ::-1])

def pad_to_square(image, bg_color=(240, 240, 235)):
    width, height = image.size
    max_dim = max(width, height)
    if width == height: 
        return image
    new_image = Image.new("RGB", (max_dim, max_dim), bg_color)
    new_image.paste(image, ((max_dim - width) // 2, (max_dim - height) // 2))
    return new_image

# --- 4. INFERENCE & FILE HANDLING PIPELINE ---
def load_ocr_model():
    print("Loading TrOCR model and processor...")
    processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    return processor, model, device

def predict_image(image_path, processor, model, device):
    try:
        raw_image = Image.open(image_path).convert("RGB")
        processed_image = pad_to_square(apply_clahe_to_pil(raw_image))
    except Exception as e:
        return f"Error_{e}"

    pixel_values = processor(processed_image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    raw_prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return raw_prediction

def sanitize_filename(text):
    """Removes characters that are invalid in file paths."""
    # Replace invalid path characters with an underscore
    safe_text = re.sub(r'[\\/*?:"<>|]', "_", text)
    # Prevent completely empty filenames if the OCR failed entirely
    return safe_text.strip() if safe_text.strip() else "BLANK_PREDICTION"

def get_unique_filepath(directory, base_name, ext):
    """Ensures no files are overwritten by appending _1, _2 to duplicates."""
    filename = f"{base_name}{ext}"
    filepath = os.path.join(directory, filename)
    counter = 1
    
    while os.path.exists(filepath):
        filename = f"{base_name}_{counter}{ext}"
        filepath = os.path.join(directory, filename)
        counter += 1
        
    return filepath

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    processor, model, device = load_ocr_model()
    corrector = HistoricalLexiconProcessor(GAZETTEER_PATH, CONFIDENCE_THRESHOLD)
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    files_to_process = [f for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith(valid_extensions)]
    
    print(f"\nFound {len(files_to_process)} images to process in '{TEST_IMAGE_DIR}'.")
    print(f"Saving renamed copies to '{RENAMED_OUTPUT_DIR}'...\n")
    
    for filename in files_to_process:
        original_path = os.path.join(TEST_IMAGE_DIR, filename)
        ext = os.path.splitext(filename)[1]
        
        # 1. Run TrOCR
        raw_text = predict_image(original_path, processor, model, device)
        
        # 2. Run Gazetteer validation
        validated_text, match_score = corrector.correct_text(raw_text)
        
        # 3. Determine the final filename
        if match_score >= CONFIDENCE_THRESHOLD:
            final_name = validated_text
        else:
            # If confidence is low, prefix it so it's easy to find and review manually
            final_name = f"REVIEW_{raw_text}"
            
        safe_name = sanitize_filename(raw_text)
        
        # 4. Get a unique path to avoid overwriting identical predictions
        new_filepath = get_unique_filepath(RENAMED_OUTPUT_DIR, safe_name, ext)
        
        # 5. Copy the file to the new directory with the new name
        shutil.copy2(original_path, new_filepath)
        
        print(f"[{match_score:5.1f}%] {filename} -> {os.path.basename(new_filepath)}")

    print(f"\nFinished processing. Check the '{RENAMED_OUTPUT_DIR}' folder.")