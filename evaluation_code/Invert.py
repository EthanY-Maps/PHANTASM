import os
from PIL import Image, ImageOps
import warnings

warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None

def invert_png_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the directory
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".png"):
            file_path = os.path.join(input_folder, filename)
            
            try:
                with Image.open(file_path) as img:
                    # Handle Transparency (RGBA)
                    if img.mode == 'RGBA':
                        # Split the channels
                        r, g, b, a = img.split()
                        
                        # Merge RGB channels to invert them specifically
                        rgb_img = Image.merge('RGB', (r, g, b))
                        inverted_rgb = ImageOps.invert(rgb_img)
                        
                        # Split inverted RGB and merge back with original Alpha
                        r_inv, g_inv, b_inv = inverted_rgb.split()
                        inverted_img = Image.merge('RGBA', (r_inv, g_inv, b_inv, a))
                    
                    # Handle Standard Images (RGB, Grayscale)
                    else:
                        inverted_img = ImageOps.invert(img.convert('RGB'))

                    # Save the result
                    output_path = os.path.join(output_folder, filename)
                    inverted_img.save(output_path)
                    print(f"Inverted: {filename}")
            
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

# --- CONFIGURATION ---
input_directory = "data/Output/Brisbane-06-03/Vectorisation/graph_viz" # <--- CHANGE THIS
output_directory = 'data/Output/Brisbane-06-03/Vectorisation/graph_viz_inverted'  # <--- CHANGE THIS

invert_png_folder(input_directory, output_directory)
print("Processing complete.")