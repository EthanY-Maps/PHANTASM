import numpy as np
import cv2
import matplotlib.pyplot as plt
from training_utils import MapDataGenerator
from config import *

def visualize_targets():
    # 1. Initialize the Generator
    # We use 'val' with split_ratio=0.8 to see the exact data used for validation
    gen = MapDataGenerator(JSON_PATH, IMAGE_PATH, usage='val', split_ratio=0.8)
    
    
    # 2. Grab a batch of Truth Data
    images, targets = gen[0]
    
    # Extract the target channels
    # center_heatmap: (Batch, 256, 256, 1)
    # poly_rays: (Batch, 256, 256, 24)
    gt_heatmaps = targets['center_heatmap']
    gt_rays = targets['poly_rays']
    
    batch_size = images.shape[0]
    angles = np.linspace(0, 360, NUM_RAYS, endpoint=False)

    for i in range(min(batch_size, 5)):
        img = (images[i] * 255).astype(np.uint8).copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        heatmap = gt_heatmaps[i, ..., 0]
        ray_map = gt_rays[i]
        
        # Find the centers in the truth heatmap
        y_centers, x_centers = np.where(heatmap > 0.5)
        
        # Draw the reconstructed polygons from the ray targets
        for y, x in zip(y_centers, x_centers):
            distances = ray_map[y, x, :]
            poly_points = []
            for j, dist in enumerate(distances):
                if dist < 1: continue
                angle_rad = np.deg2rad(angles[j])
                px = x + dist * np.cos(angle_rad)
                py = y + dist * np.sin(angle_rad)
                poly_points.append([int(px), int(py)])
            
            if len(poly_points) >= 3:
                cv2.drawContours(img, [np.array(poly_points)], -1, (255, 0, 0), 1) # Blue outline
                cv2.circle(img, (x, y), 1, (0, 0, 255), -1) # Red dot at center

        # Plotting
        plt.figure(figsize=(15, 5))
        
        # Left: Original Patch
        plt.subplot(1, 3, 1)
        plt.title("Original Image Patch")
        plt.imshow(cv2.cvtColor((images[i]*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        plt.axis('off')
        
        # Middle: Reconstructed Ground Truth Polygons
        plt.subplot(1, 3, 2)
        plt.title("Target Polygons (from Rays)")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        # Right: Target Heatmap
        plt.subplot(1, 3, 3)
        plt.title("Target Heatmap (Centers)")
        plt.imshow(heatmap, cmap='magma')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    visualize_targets()