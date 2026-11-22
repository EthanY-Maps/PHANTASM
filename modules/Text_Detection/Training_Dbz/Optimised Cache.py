import os
import glob
import numpy as np
import tensorflow as tf
from config import CACHE_DIR_NAME

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_dir_to_tfrecords(npz_dir, output_dir, shard_size=256):
    os.makedirs(output_dir, exist_ok=True)
    npz_files = glob.glob(os.path.join(npz_dir, "*.npz"))
    
    if not npz_files:
        print(f"No .npz files found in {npz_dir}")
        return

    print(f"Found {len(npz_files)} files in {npz_dir}. Converting to TFRecords...")
    
    total_shards = (len(npz_files) + shard_size - 1) // shard_size
    
    for shard_id in range(total_shards):
        start_idx = shard_id * shard_size
        end_idx = min((shard_id + 1) * shard_size, len(npz_files))
        shard_files = npz_files[start_idx:end_idx]
        
        tfrec_path = os.path.join(output_dir, f"data_shard_{shard_id:04d}.tfrec")
        
        with tf.io.TFRecordWriter(tfrec_path) as writer:
            for file_path in shard_files:
                # Load the numpy array
                with np.load(file_path) as data:
                    # Convert arrays to raw bytes for lightning-fast storage/reading
                    X_bytes = data['X'].astype(np.float32).tobytes()
                    y_bytes = data['y'].astype(np.float32).tobytes()
                
                # Create a TF Feature map
                feature = {
                    'X': _bytes_feature(X_bytes),
                    'y': _bytes_feature(y_bytes)
                }
                
                # Wrap in an Example and write
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
                
        print(f"Saved shard {shard_id + 1}/{total_shards} -> {tfrec_path}")

if __name__ == "__main__":
    # Create the train and val TFRecords
    TRAIN_DIR = os.path.join(CACHE_DIR_NAME, "train")
    VAL_DIR = os.path.join(CACHE_DIR_NAME, "val")
    
    TRAIN_OUT = os.path.join(CACHE_DIR_NAME + "_tfrec", "train")
    VAL_OUT = os.path.join(CACHE_DIR_NAME + "_tfrec", "val")
    
    print("--- Converting Training Data ---")
    convert_dir_to_tfrecords(TRAIN_DIR, TRAIN_OUT)
    
    print("\n--- Converting Validation Data ---")
    convert_dir_to_tfrecords(VAL_DIR, VAL_OUT)
    
    print("\nConversion Complete! You can now use the _tfrec folders for training.")