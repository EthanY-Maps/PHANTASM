import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np
import warnings
from tqdm import tqdm
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString

# Suppress warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. Helper: Geometry to Points
# =============================================================================

def geom_to_points(gdf, step_meters=10):
    """
    Converts a GeoDataFrame of LineStrings into a numpy array of XY points.
    interpolating points every 'step_meters' to ensure coverage.
    """
    points = []
    
    for geom in gdf.geometry:
        if geom is None or geom.is_empty: continue
        
        # Handle MultiLineStrings by breaking them down
        if isinstance(geom, MultiLineString):
            parts = list(geom.geoms)
        else:
            parts = [geom]
            
        for part in parts:
            if part.length == 0: continue
            
            # Interpolate points along the line
            # We add points every 'step' meters
            length = part.length
            num_points = int(length // step_meters) + 1
            
            for i in range(num_points):
                pt = part.interpolate(i * step_meters)
                points.append((pt.x, pt.y))
            
            # Always add the end point
            pt_end = part.interpolate(length)
            points.append((pt_end.x, pt_end.y))
            
    if not points:
        return np.empty((0, 2))
        
    return np.array(points)

# =============================================================================
# 2. Core Evaluation (KDTree)
# =============================================================================

def evaluate_run_kdtree(pred_gdf, truth_tree, truth_points, tolerance_meters=15):
    """
    Evaluates Precision/Recall using Nearest Neighbor distance.
    """
    # --- 1. PRECISION ---
    # (Are pred points close to ANY truth point?)
    pred_points_arr = geom_to_points(pred_gdf, step_meters=10)
    
    if len(pred_points_arr) == 0:
        return 0.0, 0.0, 0.0
        
    # Query Truth Tree for every Pred Point
    # k=1 returns the distance to the single nearest neighbor
    dists, _ = truth_tree.query(pred_points_arr, k=1, distance_upper_bound=tolerance_meters)
    
    # Check how many found a neighbor within tolerance
    # (Infinite distances mean no neighbor found within upper_bound)
    valid_preds = np.isfinite(dists) 
    precision = np.sum(valid_preds) / len(pred_points_arr)

    # --- 2. RECALL ---
    # (Are truth points close to ANY pred point?)
    
    # Optimization: Only check Truth points that are roughly near the predictions
    # This avoids checking points on the other side of the city
    minx, miny = np.min(pred_points_arr, axis=0) - tolerance_meters
    maxx, maxy = np.max(pred_points_arr, axis=0) + tolerance_meters
    
    # Boolean mask for truth points inside the bounding box
    mask = (
        (truth_points[:, 0] >= minx) & (truth_points[:, 0] <= maxx) &
        (truth_points[:, 1] >= miny) & (truth_points[:, 1] <= maxy)
    )
    
    local_truth_pts = truth_points[mask]
    
    if len(local_truth_pts) == 0:
        # If there is no truth here, Recall is technically "Undefined" or 0 if we missed it.
        # But if predictions exist where no truth exists, Recall isn't the penalty (Precision is).
        # We return 0.0 to be safe.
        recall = 0.0
    else:
        # Build a temporary tree for the predictions
        pred_tree = cKDTree(pred_points_arr)
        
        dists_r, _ = pred_tree.query(local_truth_pts, k=1, distance_upper_bound=tolerance_meters)
        valid_truth = np.isfinite(dists_r)
        recall = np.sum(valid_truth) / len(local_truth_pts)

    # --- 3. F1 ---
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

# =============================================================================
# 3. Batch Processor
# =============================================================================

def run_comparative_analysis(session_dir, truth_path, tolerance_meters=15, filter_open_status=False, output_folder=None):
    
    print(f"🚀 Starting Comparative Analysis on: {session_dir}")

    if output_folder:
        target_dir = output_folder
        os.makedirs(target_dir, exist_ok=True)
    else:
        target_dir = session_dir

    # 1. Load Truth & Build Tree
    print("Loading Ground Truth...")
    try:
        truth_gdf = gpd.read_file(truth_path)
        if filter_open_status and 'Map008' in truth_gdf.columns:
            truth_gdf = truth_gdf[truth_gdf['Map008'].astype(str).str.title() == 'Open']
        
        # Force Projection
        truth_gdf = truth_gdf.to_crs(epsg=7856)
        
        print("  Building Truth Index (KDTree)...")
        truth_points = geom_to_points(truth_gdf, step_meters=10)
        truth_tree = cKDTree(truth_points)
        print(f"  ✅ Indexed {len(truth_points)} truth points.")

    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # 2. Find Files
    search_pattern = os.path.join(session_dir, "**", "combined_high_medium_matches.geojson")
    match_files = glob.glob(search_pattern, recursive=True)
    print(f"📂 Found {len(match_files)} runs.")

    results = []

    # 3. Loop
    for pred_path in tqdm(match_files, desc="Processing", unit="run"):
        try:
            pred_gdf = gpd.read_file(pred_path)
            if pred_gdf.empty: continue
            
            # Force Projection
            if pred_gdf.crs is None or pred_gdf.crs.to_epsg() != 7856:
                pred_gdf = pred_gdf.to_crs(epsg=7856)

            # Metadata
            run_id = os.path.basename(os.path.dirname(pred_path))
            profile = "Unknown"
            if "Run_" in run_id:
                parts = run_id.split("_")
                if len(parts) >= 3: profile = "_".join(parts[2:])

            # Run Fast Eval
            prec, rec, f1 = evaluate_run_kdtree(pred_gdf, truth_tree, truth_points, tolerance_meters)
            
            results.append({
                "Run_ID": run_id,
                "Profile": profile,
                "Precision": prec,
                "Recall": rec,
                "F1_Score": f1,
                "Total_Pred_Length_km": pred_gdf.length.sum() / 1000.0,
                "Path": pred_path
            })
            
        except Exception as e:
            pass

    # 4. Save & Plot
    df = pd.DataFrame(results)
    if not df.empty:
        out_csv = os.path.join(target_dir, "Comparative_Evaluation_Report.csv")
        df.to_csv(out_csv, index=False)
        print(f"\n✅ Report Saved: {out_csv}")
        generate_comparison_plots(df, target_dir)
    else:
        print("❌ No valid results.")

# =============================================================================
# 4. Visualization
# =============================================================================

def generate_comparison_plots(df, out_dir):
    sns.set(style="whitegrid")
    
    # Plot 1: Scatter
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x="Recall", y="Precision", hue="Profile", style="Profile", s=150, alpha=0.8, palette="deep")
    
    x = np.linspace(0.01, 1, 100)
    for f_val in [0.2, 0.4, 0.6, 0.8]:
        y = (f_val * x) / (2 * x - f_val)
        plt.plot(x[(y >= 0) & (y <= 1.05)], y[(y >= 0) & (y <= 1.05)], 'k--', alpha=0.1)

    plt.title("Precision vs Recall (KDTree Method)", fontsize=16)
    plt.xlim(0, 1.05); plt.ylim(0, 1.05)
    plt.savefig(os.path.join(out_dir, "Compare_Scatter.png"), dpi=300)
    plt.close()

    # Plot 2: Bar
    if "Profile" in df.columns:
        best_per_profile = df.loc[df.groupby("Profile")["F1_Score"].idxmax()]
        plt.figure(figsize=(10, 6))
        sns.barplot(data=best_per_profile, x="Profile", y="F1_Score", palette="viridis")
        plt.ylim(0, 1.0)
        plt.title("Best Performance by Strategy")
        plt.savefig(os.path.join(out_dir, "Compare_Bar.png"), dpi=300)
        plt.close()


if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    # 1. Path to the SESSION folder created by your Tuning.py script
    SESSION_FOLDER = "data/Output/1903-20-01-2026/Evaluation/Session_20260122_172847" 
    
    # 2. Path to Ground Truth
    TRUTH_SHP = 'data/GreaterSydneyStreetsEvolution/DATA/Sydney_roads.shp'

    # 3. Where to save the CSV and Plots (Optional - Set to None to save in Session Folder)
    OUTPUT_REPORT_FOLDER = "data/Output/1903-20-01-2026/Evaluation_Reports/Report_20260123"
    
    # 4. Parameters
    TOLERANCE = 15
    FILTER_OPEN_STATUS = True

    # --- EXECUTE ---
    if os.path.exists(SESSION_FOLDER):
        run_comparative_analysis(
            SESSION_FOLDER, 
            TRUTH_SHP, 
            TOLERANCE, 
            filter_open_status=FILTER_OPEN_STATUS,
            output_folder=OUTPUT_REPORT_FOLDER
        )
    else:
        print(f"Error: Session folder not found: {SESSION_FOLDER}")