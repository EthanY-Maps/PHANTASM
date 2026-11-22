import os

# --- PATHS ---
# Where the model will save its weights
CHECKPOINT_DIR = "data/Training Sets/Detection/DBNET_256/checkpoints"
LOG_DIR = "data/Training Sets/Detection/DBNET_256/logs"

# The exact folder name created by cache_dataset_smart.py
CACHE_DIR_NAME = "data/Training Sets/Detection/dbnet_cache_256"

# --- MODEL SETTINGS ---
INPUT_SIZE = 256   # Must match what you used in caching
BATCH_SIZE = 16    # Increase to 16 or 32 if you have VRAM, 8 if on Mac
EPOCHS = 50
LEARNING_RATE = 1e-3

# --- LEGACY (Used only if you run old scripts) ---
# You can leave these as dummy paths now that we use the cache
IMAGE_PATH = ""
JSON_PATH = ""