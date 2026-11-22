"""NSGA-II matching search, repointed at the semi-automatically georeferenced 1903 graph.

Verbatim from _superceded/Similarity/Tuning.py except for: the input paths (the
originals refer to Output/1903-20-01-2026, deleted months ago), the dashboard
subprocess (disabled for batch runs), and the trial count (N_TRIALS env var,
default 220 to match the paper; the archived script requested 1000 and never
reached it).

The original study is unrecoverable -- optuna_pareto.db and pareto_front_results.json
no longer exist -- so Trial 202's parameters cannot be read back. This regenerates the
frontier.

Note the paper tuned on "Sydney_1903_Geolocated_slim.geojson", a slimmed graph that is
also gone. This runs on the full 18,223-linestring graph, so trials are slower and the
frontier is not expected to reproduce the original trial numbering.

    N_TRIALS=220 python scripts/tune_matching.py
"""
import os
import sys
from pathlib import Path

# The archived script lived beside the matcher; it now lives in scripts/.
_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "modules" / "Similarity", _ROOT / "_superceded" / "Similarity"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import optuna
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime
import gc
import sys
import json
import subprocess
import atexit
import time

# --- IMPORT MODULES ---
# Ensure these files are in the same directory
from HistoricMapMatcher import HistoricMapMatcher, MatchingConfig
from Comparison import evaluate_network
from load_osmnx import load_osmnx

# =============================================================================
# 1. Configuration
# =============================================================================

# Paths - UPDATE THESE IF NEEDED
BASE_OUTPUT = "data/Output_Repro/Tuning"
HIST_PATH = "data/Output_Repro/Bypass/Sydney_1903_geolocated.geojson"
OSM_PATH = "data/Input/Model/Sydney/sydney_drive.graphml"
TRUE_LARGE = 'data/GreaterSydneyStreetsEvolution/DATA/Sydney_roads.shp'

# Create Session
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_DIR = os.path.join(BASE_OUTPUT, f"Session_V5-3_Pareto_{SESSION_ID}")
os.makedirs(SESSION_DIR, exist_ok=True)
DB_PATH = f"sqlite:///{os.path.join(SESSION_DIR, 'optuna_pareto.db')}"

# =============================================================================
# 2. Objective Function (Multi-Objective)
# =============================================================================

def objective(trial):
    
    # --- A. LOG-UNIFORM SAMPLING (Explores Orders of Magnitude) ---
    # We remove 'angle_weight' as it is deprecated in V5.3
    w_shape = trial.suggest_float("raw_w_shape", 0.1, 10.0, log=True)
    w_bear  = trial.suggest_float("raw_w_bear", 0.1, 10.0, log=True)
    w_disp  = trial.suggest_float("raw_w_disp", 0.01, 5.0, log=True) # Reduced max for displacement
    w_sinu  = trial.suggest_float("raw_w_sinu", 0.1, 10.0, log=True) # Critical for zig-zags
    w_topo  = trial.suggest_float("raw_w_topo", 0.1, 10.0, log=True) # Critical for parallel jumps
    
    # --- B. STANDARD RANGES ---
    radius = trial.suggest_int("search_radius", 50, 400, step=25)
    
    # Soft Gates (Variable Slopes)
    min_len_ratio = trial.suggest_float("min_len_ratio", 0.05, 0.25)
    max_len_ratio = trial.suggest_float("max_len_ratio", 4.0, 8.0)
    
    # Confidence Thresholds
    conf_high_t = trial.suggest_float("conf_high", 0.15, 0.30)
    conf_med_t  = trial.suggest_float("conf_med", 0.50, 0.70)

    # Complexity Limits
    max_cand_edges = trial.suggest_int("max_candidates_edges", 15, 30)
    max_paths = trial.suggest_int("max_paths_solve", 5, 10)

    # --- NORMALIZE WEIGHTS ---
    # ERROR FIX: Added w_topo to sum, removed w_angle
    raw_sum = w_shape + w_bear + w_disp + w_sinu + w_topo
    
    config = MatchingConfig(
        # Registration (Fixed)
        enable_registration=True,
        registration_strict_threshold=0.25,
        ransac_residual_threshold=200.0, 
        registration_min_matches=8,

        # Pass 1: Batch Matcher
        initial_search_radius=float(radius),
        max_candidates_edges=max_cand_edges, 
        max_paths_to_solve=max_paths, 
        
        # Pass 3: Smart Relaxation
        relaxation_search_radius=float(radius) * 2.5, 
        relaxation_max_candidates=40,
        relaxation_max_paths=10,
        relaxation_length_min=0.1, 
        relaxation_length_max=max_len_ratio, 
        
        # Gates & Flow
        # renamed since the archived script: same [0,1] arccos(dot)/pi gate,
        # same 0.5 value (hard penalty only beyond 90 deg)
        bearing_hard_penalty_threshold=0.5,
        min_len_ratio=min_len_ratio,
        max_len_ratio=max_len_ratio,
        drift_tolerance_meters=80.0, 
        flow_magnitude_weight=0.3,
        flow_direction_weight=0.7,
        
        # Normalized Weights (V5.3 Update)
        shape_weight=w_shape / raw_sum,
        bearing_weight=w_bear / raw_sum,
        displacement_weight=w_disp / raw_sum,
        sinuosity_weight=w_sinu / raw_sum,
        topology_weight=w_topo / raw_sum,
        
        # Thresholds
        high_confidence_threshold=conf_high_t,
        medium_confidence_threshold=conf_med_t,
        low_confidence_threshold=conf_med_t + 0.15,
        very_low_confidence_threshold=0.9
    )
    
    # --- C. EXECUTION ---
    try:
        matcher = HistoricMapMatcher(G_hist, G_osm, config)
        match_results = matcher.run_pipeline(n_jobs=1)
        
        # Export
        run_dir = os.path.join(SESSION_DIR, f"Trial_{trial.number:03d}")
        matcher.export_results(run_dir, match_results)
        
        # Evaluate on Combined High+Medium
        pred_file = os.path.join(run_dir, "combined_high_medium.geojson")
        
        recall = 0.0
        precision = 0.0
        f1 = 0.0
        
        if os.path.exists(pred_file):
            eval_df = evaluate_network(TRUE_LARGE, pred_file, output_folder=None)
            if not eval_df.empty:
                recall = eval_df.loc["Global_Network"]["Recall"]
                precision = eval_df.loc["Global_Network"]["Precision"]
                f_1 = eval_df.loc["Global_Network"]["F1_Score"]
                
                # Log metrics for dashboard
                trial.set_user_attr("f_1_score", f_1)
        
        del matcher, match_results
        gc.collect()
        
        # RETURN TUPLE FOR MULTI-OBJECTIVE
        return recall, precision

    except Exception as e:
        print(f"    Trial {trial.number} Failed: {e}")
        return 0.0, 0.0

# =============================================================================
# 3. MAIN (PARETO SETUP)
# =============================================================================

if __name__ == "__main__":
    print("Loading Graphs...")
    global G_hist, G_osm
    if not os.path.exists(HIST_PATH): sys.exit(1)
    G_hist, G_osm = load_osmnx(HIST_PATH, OSM_PATH)

    print(f"Starting V5.3 Multi-Objective Tuning (Topological Corrections)...")
    
    # 1. Start Dashboard
    print(f"📊 Launching Optuna Dashboard...")
    try:
        raise FileNotFoundError("dashboard disabled for batch runs")
        dashboard_process = subprocess.Popen(
            ["optuna-dashboard", DB_PATH],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"✅ Dashboard running at: http://127.0.0.1:8080")
    except FileNotFoundError:
        print("⚠️  optuna-dashboard not found. Run `pip install optuna-dashboard` to view UI.")
        dashboard_process = None
    
    def cleanup_dashboard():
        if dashboard_process and dashboard_process.poll() is None:
            print("Shutting down dashboard...")
            dashboard_process.terminate()
            
    atexit.register(cleanup_dashboard)

    # 2. Setup Multi-Objective Study
    study = optuna.create_study(
        study_name="MapMatching_Pareto_V5.3",
        directions=["maximize", "maximize"], 
        storage=DB_PATH,
        load_if_exists=True,
        sampler=optuna.samplers.NSGAIISampler(seed=42, population_size=10) 
    )

    # 3. Run Optimization
    n_trials = int(os.environ.get("N_TRIALS", "220"))
    print(f"Running {n_trials} trials")
    study.optimize(objective, n_trials=n_trials)

    # 4. Report Pareto Front
    print("\n--- PARETO FRONT (Best Trade-offs) ---")
    best_trials_data = []
    
    for i, t in enumerate(study.best_trials):
        rec, prec = t.values
        beta = 0.5
        f_05 = (1 + beta**2) * (prec * rec) / ((beta**2 * prec) + rec) if (prec + rec) > 0 else 0.0
        
        print(f"Solution {i}: Recall={rec:.3f}, Precision={prec:.3f} (F0.5 approx {f_05:.3f})")
        
        best_trials_data.append({
            "solution_id": i,
            "trial_number": t.number,
            "metrics": {"recall": rec, "precision": prec, "f0.5": f_05},
            "params": t.params
        })

    # Save Pareto Front to JSON
    with open(os.path.join(SESSION_DIR, "pareto_front_results.json"), "w") as f:
        json.dump(best_trials_data, f, indent=4)
        
    print(f"\n✅ Results saved to {SESSION_DIR}")