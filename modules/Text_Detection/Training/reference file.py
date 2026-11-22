import numpy as np
from PIL import Image
import os

# --- SETTINGS ---
BIG_MAP_PATH = "data/Input/1877/nla.obj-231444908.tif"
OUTPUT_PATH = "data/Output/1903_Testing/Text_Detection/style_profile.png"
SAMPLE_SIZE = (256, 256) # 65,536 pixels is plenty for a perfect histogram

def save_exact_profile():
    print(f"Loading massive map: {BIG_MAP_PATH}...")
    img = Image.open(BIG_MAP_PATH).convert('RGB')
    
    # Convert to numpy array
    # Note: If the image is truly massive (1GB+), this might be slow.
    # For massive TIFs, we can just crop a representative chunk first.
    img_arr = np.array(img)
    
    # Flatten the array to a list of pixels (N, 3)
    pixels = img_arr.reshape(-1, 3)
    
    print(f"Sampling {SAMPLE_SIZE[0]*SAMPLE_SIZE[1]} random pixels...")
    # Randomly choose indices without replacement
    rng = np.random.default_rng()
    indices = rng.choice(pixels.shape[0], size=(SAMPLE_SIZE[0] * SAMPLE_SIZE[1]), replace=False)
    
    # Grab the exact pixel values (No interpolation/blending!)
    sampled_pixels = pixels[indices]
    
    # Reshape back into a square image
    profile_arr = sampled_pixels.reshape((SAMPLE_SIZE[0], SAMPLE_SIZE[1], 3))
    
    # Save
    profile_img = Image.fromarray(profile_arr.astype('uint8'))
    profile_img.save(OUTPUT_PATH)
    print(f"Success! Saved exact histogram profile to {OUTPUT_PATH}")

if __name__ == "__main__":
    save_exact_profile()