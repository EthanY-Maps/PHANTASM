# train.py
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model import build_center_model, build_polygon_model
from training_utils import CenterGenerator, PolygonGenerator, hybrid_blob_loss, vector_iou_loss, f1_metric
from config import *

# --- 1. LEARNING RATE WARMUP CALLBACK ---
class WarmupExponentialDecay(tf.keras.callbacks.Callback):
    def __init__(self, warmup_epochs=3, target_lr=1e-4):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = (self.target_lr / self.warmup_epochs) * (epoch + 1)
            # Use a more direct assignment that handles Keras 3 internal objects better
            self.model.optimizer.learning_rate.assign(lr)
            print(f"\nWarmup Stage: Setting Learning Rate to {lr:.2e}")

def run_pipeline():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # ==========================================
    # STAGE 1: CENTER DETECTOR (RESTRUCTURED)
    # ==========================================
    print("\n" + "="*40)
    print("STAGE 1: Training Deep U-Net with Hybrid Dice Loss")
    print("="*40)
    
    train_gen_1 = CenterGenerator(JSON_PATH, IMAGE_PATH, usage='train')
    val_gen_1 = CenterGenerator(JSON_PATH, IMAGE_PATH, usage='val')
    
    model_center = build_center_model()
    
    # We use a slightly higher learning rate for the U-Net to help it break 
    # out of that grid-pattern local minima.
    model_center.compile(
        optimizer=Adam(learning_rate=1e-4), 
        loss=hybrid_blob_loss, # Pass the function directly, not a string
        metrics=[f1_metric]    # Pass the function in a list
    )
    
    cb_1 = [
        # Warmup for the first 3 epochs
        WarmupExponentialDecay(warmup_epochs=3, target_lr=1e-4),
        
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(CHECKPOINT_DIR, "stage1_center_best.keras"), 
            save_best_only=True, monitor='val_f1_metric', mode='max' # Monitor F1 instead of Loss
        ),
        # Early stopping based on F1 performance
        tf.keras.callbacks.EarlyStopping(
            patience=12, monitor='val_f1_metric', mode='max', restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=5, monitor='val_f1_metric', mode='max', verbose=1
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(LOG_DIR, "stage1_log.csv"))
    ]
    
    model_center.fit(train_gen_1, validation_data=val_gen_1, epochs=EPOCHS, callbacks=cb_1)
    
    del model_center, train_gen_1, val_gen_1
    tf.keras.backend.clear_session()

    # ==========================================
    # STAGE 2: POLYGON REGRESSOR
    # ==========================================
    print("\n" + "="*40)
    print("STAGE 2: Training Nuclear Polygon Regressor")
    print("="*40)
    
    train_gen_2 = PolygonGenerator(JSON_PATH, IMAGE_PATH, usage='train')
    val_gen_2 = PolygonGenerator(JSON_PATH, IMAGE_PATH, usage='val')
    
    model_poly = build_polygon_model()
    model_poly.compile(optimizer=Adam(learning_rate=1e-4), loss=vector_iou_loss, metrics=['mae'])
    
    cb_2 = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(CHECKPOINT_DIR, "stage2_poly_best.keras"), 
            save_best_only=True, monitor='val_loss'
        ),
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(os.path.join(LOG_DIR, "stage2_log.csv"))
    ]
    
    model_poly.fit(train_gen_2, validation_data=val_gen_2, epochs=EPOCHS, callbacks=cb_2)

if __name__ == "__main__":
    run_pipeline()