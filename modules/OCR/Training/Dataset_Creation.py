import os
import shutil
import csv
import uuid
from pathlib import Path
from tqdm import tqdm  # pip install tqdm

def reorganize_dataset(source_root, dest_root):
    """
    Flattens a folder-of-folders structure into a single directory + CSV.
    
    Args:
        source_root (str): Path to the parent folder containing label folders.
        dest_root (str): Path where the new dataset will be created.
    """
    
    # 1. Setup Destination Paths
    images_dir = os.path.join(dest_root, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    metadata_path = os.path.join(dest_root, "metadata.csv")
    
    # list to store (filename, text) tuples
    metadata_records = []
    
    # Supported extensions
    valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    
    print(f"Scanning {source_root}...")
    
    # 2. Walk through the directory tree
    # os.walk yields (current_path, directories, files)
    for root, dirs, files in os.walk(source_root):
        # The label is the name of the current folder
        folder_name = os.path.basename(root)

        
        # Skip the root folder itself if it contains files directly
        if root == source_root:
            continue
            
        for file in files:
            file_path = Path(os.path.join(root, file))
            
            if file_path.suffix.lower() in valid_exts:
                # 3. Create a unique filename to avoid conflicts
                # We use the original label in the name for debugging, plus a random ID
                # Cleaning label for filename safety (removing spaces/weird chars)
                safe_label = "".join([c for c in folder_name if c.isalnum()])[:10] 
                unique_id = str(uuid.uuid4())[:8]
                new_filename = f"{safe_label}_{unique_id}{file_path.suffix}"
                
                dest_path = os.path.join(images_dir, new_filename)
                
                # 4. Copy the file (Use move() if you want to delete originals)
                shutil.copy2(file_path, dest_path)
                
                # 5. Add to metadata
                # folder_name IS the text truth
                metadata_records.append({
                    "file_name": new_filename,
                    "text": folder_name 
                })
    
    # 6. Write to CSV
    print(f"Found {len(metadata_records)} images. Writing CSV...")
    
    with open(metadata_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file_name', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(metadata_records)
        
    print(f"Done! Dataset ready at: {dest_root}")

# --- CONFIGURATION ---
# Change these paths to your actual folders
SOURCE_FOLDER = 'data/Training Sets/OCR/training_data'  # The folder with all the subfolders
DESTINATION_FOLDER = 'data/Training Sets/OCR/named_dataset' # Where you want the result

if __name__ == "__main__":
    reorganize_dataset(SOURCE_FOLDER, DESTINATION_FOLDER)