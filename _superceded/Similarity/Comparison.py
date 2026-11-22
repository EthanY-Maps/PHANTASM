import geopandas as gpd
import pandas as pd
import warnings
import os
import numpy as np
from shapely.ops import unary_union
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

warnings.filterwarnings('ignore')

def batched_unary_union(geometries, batch_size=2000, desc="Creating Union"):
    """Merges geometries in batches to avoid memory spikes."""
    if len(geometries) == 0: return None
    
    # Fast path for small data
    if len(geometries) <= batch_size:
        return unary_union(geometries)

    n_batches = int(np.ceil(len(geometries) / batch_size))
    batches = np.array_split(geometries, n_batches)
    
    union_parts = []
    # Disable tqdm for inner loops if running in parallel context, but here it's fine
    for batch in tqdm(batches, desc=desc, unit="batch", leave=False):
        if len(batch) > 0:
            union_parts.append(unary_union(batch))
            
    print(f"  Finalizing {desc} merge...")
    return unary_union(union_parts)

def process_intersection_batch(batch_gdf, mask_geom):
    """Helper function for parallel processing."""
    if batch_gdf.empty:
        return 0.0
    # Intersection returns the parts of lines inside the mask
    return batch_gdf.geometry.intersection(mask_geom).length.sum()


def clean_and_evaluate(truth_path, pred_path, output_folder, tolerance=5):
    print("--- 1. Cleaning Predictions ---")
    pred_gdf = gpd.read_file(pred_path)
    original_len = len(pred_gdf)
    
    # Method 1: Drop exact geometry duplicates
    # We convert to WKB (binary) to ensure robust duplicate checking
    pred_gdf['wkb'] = pred_gdf.geometry.apply(lambda geom: geom.wkb)
    pred_clean = pred_gdf.drop_duplicates(subset=['wkb'])
    
    # Drop the temporary column
    pred_clean = pred_clean.drop(columns=['wkb'])
    
    new_len = len(pred_clean)
    print(f"Original rows: {original_len}")
    print(f"Cleaned rows:  {new_len}")
    print(f"Removed {original_len - new_len} duplicate lines.")

    return pred_clean

def evaluate_network(truth_path, pred_path, output_folder, tolerance_meters=5):
    print(f"--- Starting High-Performance Evaluation ---")
    

    # Check if unique IDs or geometry counts match row counts

    pred_gdf= clean_and_evaluate(truth_path, pred_path, output_folder, tolerance_meters)

    # 0. Setup
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        csv_path = os.path.join(output_folder, "global_evaluation_results.csv")
    
    # 1. Load Data
    print("Loading data...")
    truth_gdf = gpd.read_file(truth_path)


    print(f"Pred Rows: {len(pred_gdf)}")
    print(f"Unique Geometries: {len(pred_gdf.geometry.unique())}")
    
    if 'MAP008' in truth_gdf.columns:
        truth_gdf = truth_gdf[(truth_gdf['MAP008'].astype(str).str.title() == 'Open') & (truth_gdf['MAP008'].astype(str).str.title() == 'Open')]

    # 2. Project
    target_crs = 7856
    print(f"Projecting to EPSG:{target_crs}...")
    truth_gdf = truth_gdf.to_crs(epsg=target_crs)
    pred_gdf = pred_gdf.to_crs(epsg=target_crs)

    total_truth_km = truth_gdf.length.sum() / 1000
    total_pred_km = pred_gdf.length.sum() / 1000

    # Determine CPU cores for parallel processing
    n_cores = multiprocessing.cpu_count()
    print(f"Using {n_cores} CPU cores for processing.")

    # ================= RECALL =================
    print("\n--- Computing RECALL (Truth covered by Pred) ---")
    
    # Optimization A: Lower resolution buffer (Huge speedup, negligible accuracy loss)
    # resolution=2 means "octagons" instead of perfect circles.
    print("  Creating Prediction Mask...")
    pred_buffers = pred_gdf.geometry.buffer(tolerance_meters, resolution=2)
    pred_union = batched_unary_union(pred_buffers, desc="Merging Preds")

    # Optimization B: Simplification (Optional but recommended)
    # Removes tiny artifacts from the union. 0.5m is safe for 5m tolerance.
    pred_union = pred_union.simplify(0.5)

    # Optimization C: Pre-filtering
    # Only check Truth lines that actually touch the Prediction bounding box area
    print("  Pre-filtering Truth lines...")
    possible_matches_idx = truth_gdf.sindex.query(pred_union, predicate='intersects')
    truth_candidates = truth_gdf.iloc[possible_matches_idx]
    
    print(f"  Checking {len(truth_candidates)} candidates (filtered from {len(truth_gdf)})...")
    
    # Optimization D: Parallel Intersection
    if not truth_candidates.empty:
        # Split into chunks for cores
        batches = np.array_split(truth_candidates, n_cores * 4) 
        
        results = Parallel(n_jobs=n_cores, backend="loky")(
            delayed(process_intersection_batch)(batch, pred_union) 
            for batch in tqdm(batches, desc="  Parallel Intersection", unit="core")
        )
        matched_truth_km = sum(results) / 1000
    else:
        matched_truth_km = 0

    recall = matched_truth_km / total_truth_km if total_truth_km > 0 else 0


    # ================= PRECISION =================
    print("\n--- Computing PRECISION (Pred covered by Truth) ---")
    
    print("  Creating Truth Mask...")
    truth_buffers = truth_gdf.geometry.buffer(tolerance_meters, resolution=2)
    truth_union = batched_unary_union(truth_buffers, desc="Merging Truths")
    truth_union = truth_union.simplify(0.5)

    print("  Pre-filtering Pred lines...")
    possible_matches_idx = pred_gdf.sindex.query(truth_union, predicate='intersects')
    pred_candidates = pred_gdf.iloc[possible_matches_idx]
    
    print(f"  Checking {len(pred_candidates)} candidates (filtered from {len(pred_gdf)})...")

    if not pred_candidates.empty:
        batches = np.array_split(pred_candidates, n_cores * 4)
        
        results = Parallel(n_jobs=n_cores, backend="loky")(
            delayed(process_intersection_batch)(batch, truth_union) 
            for batch in tqdm(batches, desc="  Parallel Intersection", unit="core")
        )
        matched_pred_km = sum(results) / 1000
    else:
        matched_pred_km = 0

    precision = matched_pred_km / total_pred_km if total_pred_km > 0 else 0

    # ================= RESULTS =================
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    result_data = {
        "Type": "Global_Network",
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "Truth_Total_km": total_truth_km,
        "Pred_Total_km": total_pred_km,
        "Matched_Truth_km": matched_truth_km,
        "Matched_Pred_km": matched_pred_km
    }
    
    results_df = pd.DataFrame([result_data]).round(4)
    results_df = results_df.set_index("Type")
    
    if output_folder:
        results_df.to_csv(csv_path)
        print(f"\n✅ Results saved to: {csv_path}")
        
    return results_df

if __name__ == "__main__":
    # Update paths here
    PRED_FILE = "data/Output/1903-20-01-2026/Evaluation/Session_Optuna_20260128_175305/Trials/Trial_023/combined_high_medium_matches.geojson"
    TRUTH_FILE = 'data/GreaterSydneyStreetsEvolution/DATA/Sydney_roads.shp'
    OUTPUT_DIR = "data/Output/1903-20-01-2026/Similarity/"

    df = evaluate_network(TRUTH_FILE, PRED_FILE, OUTPUT_DIR)
    print("\n--- Final Results ---")
    print(df)