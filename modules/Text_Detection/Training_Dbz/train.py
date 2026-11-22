import os
import glob
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Adam
# UPDATED: Import the DBNet++ model we built
from model import build_dbnet_plusplus_model 
from training_utils import db_loss
from config import *

AUTOTUNE = tf.data.AUTOTUNE

# --- 1. RAM-CACHED DATA PIPELINE ---
def parse_tfrecord_batch_fn(example_batch):
    feature_description = {
        'X': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([], tf.string),
    }
    
    parsed = tf.io.parse_example(example_batch, feature_description)
    
    X = tf.io.decode_raw(parsed['X'], tf.float32)
    y = tf.io.decode_raw(parsed['y'], tf.float32)
    
    # Reshape to static batch size for Metal compatibility
    X = tf.reshape(X, [BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, 3])
    y = tf.reshape(y, [BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, 3])
    
    # --- GPU-ACCELERATED AUGMENTATION ---
    
    # 1. Photometric (Colour/Lighting) - Applied ONLY to the image (X)
    X = tf.image.random_brightness(X, max_delta=0.15)
    X = tf.image.random_contrast(X, lower=0.8, upper=1.2)
    X = tf.image.random_saturation(X, lower=0.8, upper=1.2)
    
    # Ensure pixel values don't exceed the 0.0 - 1.0 range after adjustments
    X = tf.clip_by_value(X, 0.0, 1.0)
    
    # 2. Geometric (Rotation) - Applied IDENTICALLY to both X and y
    # Generates a random integer: 0 (0°), 1 (90°), 2 (180°), or 3 (270°)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    
    X = tf.image.rot90(X, k=k)
    y = tf.image.rot90(y, k=k)
    
    return X, y

def create_tfrecord_dataset(tfrecord_dir, batch_size, is_training=True):
    pattern = os.path.join(tfrecord_dir, "*.tfrec")
    files = tf.data.Dataset.list_files(pattern, shuffle=is_training)
    
    ds = files.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=AUTOTUNE,
        num_parallel_calls=AUTOTUNE
    )
    
    if is_training:
        ds = ds.shuffle(buffer_size=1024) 
    
    # SWAPPED: We batch FIRST, then map the batch function
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(parse_tfrecord_batch_fn, num_parallel_calls=AUTOTUNE)
    
    # Add prefetch_to_device if standard prefetch isn't keeping GPU at 100%
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    ds = ds.repeat() 
    
    return ds

# --- 2. WARMUP + COSINE DECAY CALLBACK (FIXED) ---
class WarmupCosineDecay(tf.keras.callbacks.Callback):
    def __init__(self, warmup_epochs=5, total_epochs=50, target_lr=1e-3, min_lr=1e-6):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.target_lr = target_lr
        self.min_lr = min_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Linear Warmup
            lr = (self.target_lr / self.warmup_epochs) * (epoch + 1)
        else:
            # Cosine Decay
            progress = (epoch - self.warmup_epochs) / max(1, (self.total_epochs - self.warmup_epochs))
            lr = self.min_lr + 0.5 * (self.target_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
            
        self.model.optimizer.learning_rate.assign(lr)
        print(f"\nEpoch {epoch+1}: Learning Rate = {lr:.2e}")

# --- 3. MAIN ---
def run_fast_training():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    TRAIN_DIR = os.path.join(CACHE_DIR_NAME + "_tfrec", "train")
    VAL_DIR = os.path.join(CACHE_DIR_NAME + "_tfrec", "val")
    
    # --- DYNAMIC EPOCH CALCULATION ---
    OLD_TRAIN_DIR = os.path.join(CACHE_DIR_NAME, "train")
    OLD_VAL_DIR = os.path.join(CACHE_DIR_NAME, "val")
    
    n_train = len(glob.glob(os.path.join(OLD_TRAIN_DIR, "*.npz")))
    n_val = len(glob.glob(os.path.join(OLD_VAL_DIR, "*.npz")))
    
    steps_per_epoch = max(1, n_train // BATCH_SIZE)
    validation_steps = max(1, n_val // BATCH_SIZE)
    
    print(f"Building TFRecord Pipelines from {TRAIN_DIR}...")
    print(f"Dataset Size: {n_train} Train | {n_val} Val")
    print(f"Steps per epoch: {steps_per_epoch}")
    
    train_ds = create_tfrecord_dataset(TRAIN_DIR, BATCH_SIZE, is_training=True)
    val_ds = create_tfrecord_dataset(VAL_DIR, BATCH_SIZE, is_training=False)
    
    print(f"Building DBNet++ (Input: {INPUT_SIZE}x{INPUT_SIZE})...")
    # UPDATED: Use the PlusPlus model
    model = build_dbnet_plusplus_model(input_size=INPUT_SIZE)
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE), 
        loss=db_loss,
    )
    
    callbacks = [
        WarmupCosineDecay(warmup_epochs=5, total_epochs=EPOCHS, target_lr=LEARNING_RATE),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(CHECKPOINT_DIR, "stage2_dbnet_best.keras"), 
            save_best_only=True, monitor='val_loss'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=10, monitor='val_loss', restore_best_weights=True
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(LOG_DIR, "stage2_dbnet_log.csv"))
    ]
    
    model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

if __name__ == "__main__":
    run_fast_training()