import os
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import json
from torch.utils.data import Dataset
from PIL import Image
from pdf2image import convert_from_path
from sklearn.model_selection import train_test_split
from torchvision import transforms
import evaluate

from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    default_data_collator,
    EarlyStoppingCallback
)


# --- 1. NEW CONFIGURATION ---
MAP_JSON_PAIRS = [
    {"map": "data/Input/1877/nla.obj-231444908.tif", "json": "data/Input/1877/1877_Text_Labels.json"},
    {"map": 'data/Input/1886 Atlas of Sydney/Drummoyne/1083255_1193-13-Drummoyne.pdf', "json": 'data/Input/1886 Atlas of Sydney/Drummoyne/1083255_1193-13-Drummoyne_Text.json'},
    {"map": 'data/Input/1886 Atlas of Sydney/Endfield/1083257_1193-15-Enfield.tif', "json": "data/Input/1886 Atlas of Sydney/Endfield/1083257_1193-15-Enfield.json"}
]

PROCESSED_IMAGE_DIR = 'data/Training Sets/OCR/processed_training_crops'
OUTPUT_DIR = 'data/Training Sets/OCR/trocr_tuning_output'

BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 5e-5
MAX_LENGTH = 32
CROP_BUFFER = 4  # Slightly larger buffer for the rotated crop

os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- 2. MATH & PRE-PROCESSING FUNCTIONS ---
def get_rotation_and_bounds(points):
    """Calculates primary axis via PCA to level the text, and returns bounding boxes."""
    pts = np.array(points)
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid

    # Covariance & Eigen decomposition
    cov = np.cov(centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)

    # Sort for primary eigenvector
    sort_indices = np.argsort(evals)[::-1] 
    v1 = evecs[:, sort_indices[0]] 
    v2 = evecs[:, sort_indices[1]] 

    # Prevent upside-down text
    if v1[0] < 0:
        v1 = -v1

    angle_rad = np.arctan2(v1[1], v1[0])
    angle_deg = np.degrees(angle_rad)

    # Project to get tight rotated bounds
    proj_x = centered @ v1
    proj_y = centered @ v2
    rot_bounds = (np.min(proj_x), np.max(proj_x), np.min(proj_y), np.max(proj_y))
    
    # Original bounds for safe cropping area
    orig_bounds = (np.min(pts[:, 0]), np.max(pts[:, 0]), np.min(pts[:, 1]), np.max(pts[:, 1]))

    return angle_deg, centroid, rot_bounds, orig_bounds

def apply_clahe_to_pil(image):
    """Applies Adaptive Histogram Equalization to fix faded historical ink."""
    cv_img = np.array(image)
    if len(cv_img.shape) == 2:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
    else:
        cv_img = cv_img[:, :, ::-1].copy() # RGB to BGR

    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    merged = cv2.merge((cl, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return Image.fromarray(enhanced_bgr[:, :, ::-1])

def pad_to_square(image, bg_color=(240, 240, 235)):
    """Pads image to square to prevent aspect ratio distortion during TrOCR resizing."""
    width, height = image.size
    max_dim = max(width, height)
    if width == height:
        return image
    new_image = Image.new("RGB", (max_dim, max_dim), bg_color)
    new_image.paste(image, ((max_dim - width) // 2, (max_dim - height) // 2))
    return new_image

def prepare_dataset(pairs, output_dir, buffer_size=CROP_BUFFER):
    """Extracts, rotates, binarizes, and saves map text crops using JSON polygons."""
    data_records = []
    crop_idx = 0
    print("--- Starting Pre-processing with PCA Rotation & CLAHE ---")
    
    for pair in pairs:
        map_path = pair["map"]
        json_path = pair["json"]
        print(f"Processing map: {os.path.basename(map_path)}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        shapes = data.get('annotations', data) if isinstance(data, dict) else data

        if map_path.lower().endswith(".pdf"):
            pages = convert_from_path(map_path, dpi=300)
        else:
            pages = [Image.open(map_path).convert("RGB")]

        for shape in shapes:
            truth_text = shape.get("truth_text", "").strip()
            points = shape.get("points")
            if not truth_text or not points or len(points) < 3:
                continue

            page_idx = shape.get('page_index', 0)
            target_image = pages[min(page_idx, len(pages) - 1)]

            try:
                angle_deg, centroid, rot_bounds, orig_bounds = get_rotation_and_bounds(points)
                cx, cy = centroid
                min_x_rot, max_x_rot, min_y_rot, max_y_rot = rot_bounds
                min_x_orig, max_x_orig, min_y_orig, max_y_orig = orig_bounds

                # 1. Safe square crop for rotation
                diag = int(np.sqrt((max_x_orig - min_x_orig)**2 + (max_y_orig - min_y_orig)**2)) + 20
                left = max(0, int(cx - diag/2))
                top = max(0, int(cy - diag/2))
                right = min(target_image.width, int(cx + diag/2))
                bottom = min(target_image.height, int(cy + diag/2))

                large_crop = target_image.crop((left, top, right, bottom))

                # 2. Rotate to level the text
                local_cx, local_cy = cx - left, cy - top
                rotated_crop = large_crop.rotate(angle_deg, resample=Image.BICUBIC, center=(local_cx, local_cy))

                # 3. Final tight crop around the leveled text
                tight_left = max(0, int(local_cx + min_x_rot) - buffer_size)
                tight_top = max(0, int(local_cy + min_y_rot) - buffer_size)
                tight_right = min(rotated_crop.width, int(local_cx + max_x_rot) + buffer_size)
                tight_bottom = min(rotated_crop.height, int(local_cy + max_y_rot) + buffer_size)

                final_crop = rotated_crop.crop((tight_left, tight_top, tight_right, tight_bottom))

                # 4. Enhance contrast and save
                final_crop = apply_clahe_to_pil(final_crop)
                
                crop_filename = f"crop_{crop_idx:06d}.jpg"
                crop_filepath = os.path.join(output_dir, crop_filename)
                final_crop.save(crop_filepath)

                data_records.append({"file_name": crop_filename, "text": truth_text})
                crop_idx += 1
                
            except Exception as e:
                print(f"Error processing {truth_text}: {e}")

    print(f"Pre-processing complete. Extracted {len(data_records)} samples.")
    return pd.DataFrame(data_records)


# --- 3. AUGMENTATION & DATASET ---
train_transforms = transforms.Compose([
    transforms.RandomGrayscale(p=0.2), 
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomPerspective(distortion_scale=0.15, p=0.2), 
    transforms.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.95, 1.05), shear=3),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
])

class OCRDataset(Dataset):
    def __init__(self, df, root_dir, processor, max_target_length=128, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.processor = processor
        self.max_target_length = max_target_length
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df.iloc[idx]['file_name']
        text = str(self.df.iloc[idx]['text'])
        image_path = os.path.join(self.root_dir, file_name)
        
        try:
            image = Image.open(image_path).convert("RGB")
            image = pad_to_square(image)
            if self.transform:
                image = self.transform(image)
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            return self.__getitem__((idx + 1) % len(self.df))

        labels = self.processor.tokenizer(
            text, padding="max_length", max_length=self.max_target_length, truncation=True
        ).input_ids
        labels = [l if l != self.processor.tokenizer.pad_token_id else -100 for l in labels]

        return {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}


# --- 4. METRICS & PLOTTING ---
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}

def plot_history(log_history,output_dir):
    train_loss, eval_loss, eval_cer = [], [], []
    for entry in log_history:
        if 'loss' in entry: train_loss.append({'step': entry['step'], 'loss': entry['loss']})
        elif 'eval_loss' in entry: 
            eval_loss.append({'step': entry['step'], 'loss': entry['eval_loss']})
            eval_cer.append({'step': entry['step'], 'cer': entry['eval_cer']})
            
    df_train, df_eval, df_cer = pd.DataFrame(train_loss), pd.DataFrame(eval_loss), pd.DataFrame(eval_cer)
    if df_train.empty or df_eval.empty: return

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df_train['step'], df_train['loss'], label='Training Loss', alpha=0.6)
    plt.plot(df_eval['step'], df_eval['loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Steps'); plt.ylabel('Loss'); plt.title('Training vs Validation Loss'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    if not df_cer.empty:
        plt.plot(df_cer['step'], df_cer['cer'], label='Validation CER', color='orange')
        plt.xlabel('Steps'); plt.ylabel('CER'); plt.title('Character Error Rate'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_graphs.png")


def verify_data_alignment(df, image_dir, num_samples=10, output_file="sanity_check.png"):
    """
    Generates a 2x5 grid of the first 10 crops and their assigned text labels
    so you can visually verify that the bounding boxes match the JSON text.
    """
    print(f"\nGenerating sanity check grid for the first {num_samples} samples...")
    sample_df = df.head(num_samples)
    
    plt.figure(figsize=(15, 6))
    for i, (index, row) in enumerate(sample_df.iterrows()):
        img_path = os.path.join(image_dir, row['file_name'])
        truth_text = row['text']
        
        try:
            img = Image.open(img_path)
            # Create a 2-row, 5-column subplot
            plt.subplot(2, 5, i + 1)
            plt.imshow(img)
            # Make the title stand out
            plt.title(f"Label: '{truth_text}'", fontsize=12, fontweight='bold', color='darkblue')
            plt.axis('off')
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")
            
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Sanity check saved to '{output_file}'. Please open it and verify the labels match the images!")


# --- 5. MAIN EXECUTION ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    df = prepare_dataset(MAP_JSON_PAIRS, PROCESSED_IMAGE_DIR)

    verify_data_alignment(df, PROCESSED_IMAGE_DIR)
    if len(df) == 0: exit("No training data found. Check your JSON formatting.")

    train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)
    
    model_name = "microsoft/trocr-base-handwritten"
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    
    # Config
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = MAX_LENGTH
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    
    # --- ADD THIS LINE ---
    model.config.repetition_penalty = 2.0  # Forces the model to stop looping syllables
    # ---------------------
    
    model.to(device)

    train_dataset = OCRDataset(train_df, PROCESSED_IMAGE_DIR, processor, MAX_LENGTH, transform=train_transforms)
    eval_dataset = OCRDataset(eval_df, PROCESSED_IMAGE_DIR, processor, MAX_LENGTH, transform=None)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        predict_with_generate=True,
        
        # --- NEW: STEP-BASED EVALUATION ---
        eval_strategy="steps",           # Changed from "epoch"
        eval_steps=500,                  # Run validation every 500 steps
        save_strategy="steps",           # Changed to match eval
        save_steps=500,                  # Save a checkpoint every 500 steps
        # ----------------------------------
        
        logging_steps=50,                # Log training loss every 50 steps to keep the console clean
        
        # --- THE ANTI-OVERFITTING TWEAKS ---
        learning_rate=1e-5,              
        weight_decay=0.05,               
        warmup_ratio=0.2,                
        max_grad_norm=1.0,               
        # -----------------------------------
        
        num_train_epochs=EPOCHS,
        lr_scheduler_type="linear",
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        dataloader_pin_memory=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("Starting training with augmentation...")
    trainer.train()

    print("Saving model...")
    model.save_pretrained(f"{OUTPUT_DIR}/final_model")
    processor.save_pretrained(f"{OUTPUT_DIR}/final_model")
    
    print("Generating graphs...")
    plot_history(trainer.state.log_history, OUTPUT_DIR)