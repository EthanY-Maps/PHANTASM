# config.py
import os

# --- PATHS ---
# Update these paths to match your local setup
JSON_PATH = "data/Training Sets/Detection/ResNet_Box/1877_Text_Labels.json"
IMAGE_PATH = "data/Input/1877/nla.obj-231444908.tif"
CHECKPOINT_DIR = "data/Training Sets/Detection/ResNet_polybox/Checkpoints"
LOG_DIR = "data/Training Sets/Detection/ResNet_Box/Logs"

# --- HYPERPARAMETERS ---
PATCH_SIZE = 256        # Size for Stage 1 (Scanning)
INPUT_SHAPE_POLY = 128  # Size for Stage 2 (Regression Crop)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
SPLIT_RATIO = 0.8       # 80% Train, 20% Val

# --- POLYGON SPECIFICS ---
NUM_RAYS = 64           # Number of rays (vectors) to predict per text instance
MAX_RAY_DIST = 128.0    # Normalization factor (must be > sqrt(64^2 + 64^2) ≈ 90.5)