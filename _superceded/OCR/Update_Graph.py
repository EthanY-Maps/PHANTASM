#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 12:09:49 2025

"""

import os
import geopandas as gpd
from tqdm import tqdm

def update_geojson_with_predictions(geojson_path, cutout_folder, output_geojson_path):
    # Step 1: Build mapping from cutout filenames
    id_to_prediction = {}

    for fname in os.listdir(cutout_folder):
        if fname.lower().endswith(".png") and "_" in fname:
            name_part = os.path.splitext(fname)[0]
            # Expected format: snippet_000123_PredictionText
            parts = name_part.split("_")
            
            # Check if it matches expected pattern (snippet_ID_Text)
            if len(parts) >= 3 and parts[0] == "snippet":
                feature_id_str = parts[1]
                # Reconstruct text parts (in case text contained underscores)
                predicted_text = "_".join(parts[2:])
                
                # Normalize ID to match GeoJSON index (string of int)
                try:
                    clean_id = str(int(feature_id_str))
                    id_to_prediction[clean_id] = predicted_text
                except ValueError:
                    continue

    print(f"🗂️ Found {len(id_to_prediction)} predicted cutouts.")

    # Step 2: Load original GeoJSON (preserves ALL fields and geometries)
    gdf = gpd.read_file(geojson_path)

    # Step 3: Add new column WITHOUT modifying old ones
    predicted_texts = []

    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Updating GeoJSON", unit="feat"):
        feature_id = str(idx)
        predicted_text = id_to_prediction.get(feature_id, None)  # None if not found
        predicted_texts.append(predicted_text)

    gdf["predicted_text"] = predicted_texts  # 🛡️ Safe additive

    # Step 4: Save updated GeoJSON
    gdf.to_file(output_geojson_path, driver="GeoJSON")
    print(f"✅ Updated GeoJSON saved to {output_geojson_path}")

if __name__ == "__main__":
    update_geojson_with_predictions(
        geojson_path="data/Output/1903_08_25/stitch/1903/Sydney_1903.geojson",
        cutout_folder="data/Output/1903_08_25/ocr_cutouts",  # predicted cutouts folder
        output_geojson_path= "data/Output/1903_08_25/stitch/1903/Sydney_1903.geojson"
    )