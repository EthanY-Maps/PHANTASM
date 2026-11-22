"""
Graph Evaluation Script
=======================
Evaluates predicted road graphs against ground-truth graphs.
Mirrors the structure of UNET_Evaluation.py.

Coordinate convention
---------------------
GeoJSON stores coordinates as [x_col, -y_row] (y was negated on export).
Internally this script works in pixel space: (x=col, y=row), both positive.

Metrics
-------
1. Geometric precision / recall / F1 at multiple buffer thresholds
   - Dense points sampled along every edge (every `sample_spacing` px)
   - KD-tree nearest-neighbour queries between GT and predicted point clouds
   - Reported at thresholds: 5, 10, 20, 50 px

2. Junction detection precision / recall / F1
   - Junction = node of degree ≥ 3 (inferred from edge endpoints)
   - Matched within a configurable radius

3. Graph topology statistics
   - Edge / node counts, total length, avg degree, components
   - Ratio of predicted vs GT for each stat

4. Visualisation
   - GT edges in green, predicted edges in red
   - Saved as PNG next to the output CSV

Usage
-----
# Single pair
python Graph_Evaluation.py

# Adjust the paths / constants at the bottom of the file for batch mode.
"""

import os
import json
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from scipy.spatial import cKDTree


# ─── Constants ──────────────────────────────────────────────────────────────
SAMPLE_SPACING   = 2        # px between sampled points along edges
BUFFER_THRESHOLDS = [5, 10, 20, 50]   # px thresholds for geometric P/R/F1
JUNCTION_THRESHOLD = 20     # px radius for junction matching


# ─── Loading ────────────────────────────────────────────────────────────────

def load_edges_from_geojson(path: str) -> List[List[Tuple[float, float]]]:
    """
    Load edge coordinate lists from a roads GeoJSON file.

    Returns a list of edges; each edge is a list of (x_col, y_row) tuples
    in positive pixel space.  GeoJSON stores [x, -y], so y is negated back.
    """
    with open(path, 'r') as f:
        gj = json.load(f)

    edges = []
    for feat in gj.get('features', []):
        geom = feat.get('geometry', {})
        if geom.get('type') != 'LineString':
            continue
        coords = geom.get('coordinates', [])
        if len(coords) < 2:
            continue
        # [x_col, -y_row]  →  (x_col, y_row)
        edge = [(float(c[0]), -float(c[1])) for c in coords]
        edges.append(edge)

    return edges


# ─── Point sampling ─────────────────────────────────────────────────────────

def sample_points_along_edge(edge: List[Tuple[float, float]],
                              spacing: float = SAMPLE_SPACING
                              ) -> np.ndarray:
    """Sample points at `spacing`-px intervals along one polyline edge."""
    pts = [edge[0]]
    accumulated = 0.0

    for i in range(1, len(edge)):
        x0, y0 = edge[i - 1]
        x1, y1 = edge[i]
        seg_len = np.hypot(x1 - x0, y1 - y0)
        if seg_len < 1e-9:
            continue
        n_steps = int(seg_len / spacing)
        for k in range(1, n_steps + 1):
            t = k * spacing / seg_len
            if t > 1.0:
                break
            pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
        pts.append((x1, y1))

    return np.array(pts, dtype=np.float32)


def sample_points_from_edges(edges: List[List[Tuple[float, float]]],
                              spacing: float = SAMPLE_SPACING) -> np.ndarray:
    """Dense point cloud sampled from all edges."""
    all_pts = [sample_points_along_edge(e, spacing) for e in edges if len(e) >= 2]
    if not all_pts:
        return np.empty((0, 2), dtype=np.float32)
    return np.vstack(all_pts)


# ─── Geometric metrics ───────────────────────────────────────────────────────

def compute_geometric_metrics(gt_pts: np.ndarray,
                               pred_pts: np.ndarray,
                               thresholds: List[int] = BUFFER_THRESHOLDS
                               ) -> Dict[str, float]:
    """
    Buffer-based geometric precision / recall / F1.

    For each threshold T:
      Precision  = fraction of pred points within T px of any GT point
      Recall     = fraction of GT points within T px of any pred point
      F1         = harmonic mean
    """
    metrics: Dict[str, float] = {}

    if len(gt_pts) == 0 or len(pred_pts) == 0:
        for t in thresholds:
            metrics[f"Geo_Precision@{t}px"] = 0.0
            metrics[f"Geo_Recall@{t}px"]    = 0.0
            metrics[f"Geo_F1@{t}px"]        = 0.0
        return metrics

    gt_tree   = cKDTree(gt_pts)
    pred_tree = cKDTree(pred_pts)

    for t in thresholds:
        # Precision: how many pred points are covered by GT?
        d_pred_to_gt, _ = gt_tree.query(pred_pts, workers=-1)
        precision = float(np.mean(d_pred_to_gt <= t))

        # Recall: how many GT points are covered by pred?
        d_gt_to_pred, _ = pred_tree.query(gt_pts, workers=-1)
        recall = float(np.mean(d_gt_to_pred <= t))

        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        metrics[f"Geo_Precision@{t}px"] = precision
        metrics[f"Geo_Recall@{t}px"]    = recall
        metrics[f"Geo_F1@{t}px"]        = f1

    return metrics


# ─── Junction metrics ────────────────────────────────────────────────────────

def extract_junctions(edges: List[List[Tuple[float, float]]],
                       threshold_px: float = 5.0) -> np.ndarray:
    """
    Infer junction positions from edge endpoints.

    Two endpoints within `threshold_px` of each other are considered the same
    node.  Nodes with degree ≥ 3 are junctions.
    """
    # Collect all endpoints
    all_endpoints = []
    for edge in edges:
        all_endpoints.append(edge[0])
        all_endpoints.append(edge[-1])

    if not all_endpoints:
        return np.empty((0, 2), dtype=np.float32)

    pts = np.array(all_endpoints, dtype=np.float32)
    tree = cKDTree(pts)

    # Union-Find to cluster nearby endpoints into nodes
    parent = list(range(len(pts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = tree.query_pairs(r=threshold_px)
    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    # Count degree of each cluster
    from collections import Counter
    cluster_ids = [find(i) for i in range(len(pts))]
    degree = Counter(cluster_ids)

    # Return centroid of clusters with degree ≥ 3
    junctions = []
    from collections import defaultdict
    clusters: Dict[int, List[int]] = defaultdict(list)
    for i, cid in enumerate(cluster_ids):
        clusters[cid].append(i)

    for cid, members in clusters.items():
        if degree[cid] >= 3:
            centroid = pts[members].mean(axis=0)
            junctions.append(centroid)

    return np.array(junctions, dtype=np.float32) if junctions else np.empty((0, 2), dtype=np.float32)


def compute_junction_metrics(gt_edges: List[List[Tuple[float, float]]],
                              pred_edges: List[List[Tuple[float, float]]],
                              threshold: float = JUNCTION_THRESHOLD
                              ) -> Dict[str, float]:
    """Precision / recall / F1 for detected junction nodes."""
    gt_junctions   = extract_junctions(gt_edges)
    pred_junctions = extract_junctions(pred_edges)

    if len(gt_junctions) == 0 and len(pred_junctions) == 0:
        return {"Junction_Precision": 1.0, "Junction_Recall": 1.0, "Junction_F1": 1.0,
                "GT_Junctions": 0, "Pred_Junctions": 0}

    if len(gt_junctions) == 0:
        return {"Junction_Precision": 0.0, "Junction_Recall": 0.0, "Junction_F1": 0.0,
                "GT_Junctions": 0, "Pred_Junctions": int(len(pred_junctions))}

    if len(pred_junctions) == 0:
        return {"Junction_Precision": 0.0, "Junction_Recall": 0.0, "Junction_F1": 0.0,
                "GT_Junctions": int(len(gt_junctions)), "Pred_Junctions": 0}

    gt_tree   = cKDTree(gt_junctions)
    pred_tree = cKDTree(pred_junctions)

    # Precision: predicted junctions matched to GT
    d_pred, _ = gt_tree.query(pred_junctions)
    precision  = float(np.mean(d_pred <= threshold))

    # Recall: GT junctions matched to prediction
    d_gt, _   = pred_tree.query(gt_junctions)
    recall     = float(np.mean(d_gt <= threshold))

    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "Junction_Precision":  precision,
        "Junction_Recall":     recall,
        "Junction_F1":         f1,
        "GT_Junctions":        int(len(gt_junctions)),
        "Pred_Junctions":      int(len(pred_junctions)),
    }


# ─── Topology stats ──────────────────────────────────────────────────────────

def compute_topology_stats(edges: List[List[Tuple[float, float]]],
                            prefix: str = "") -> Dict[str, float]:
    """Basic graph topology statistics from edge list."""
    total_length   = sum(
        sum(np.hypot(e[i+1][0]-e[i][0], e[i+1][1]-e[i][1]) for i in range(len(e)-1))
        for e in edges
    )
    endpoint_counts: Dict[Tuple, int] = {}
    for edge in edges:
        for pt in [edge[0], edge[-1]]:
            key = (round(pt[0]), round(pt[1]))
            endpoint_counts[key] = endpoint_counts.get(key, 0) + 1

    n_nodes = len(endpoint_counts)
    n_edges = len(edges)
    avg_degree = (2 * n_edges / n_nodes) if n_nodes > 0 else 0.0

    return {
        f"{prefix}Edges":        n_edges,
        f"{prefix}Nodes":        n_nodes,
        f"{prefix}Total_Length": round(total_length, 1),
        f"{prefix}Avg_Degree":   round(avg_degree, 3),
    }


# ─── Visualisation ───────────────────────────────────────────────────────────

def _draw_edges(ax, edges, color, alpha=0.8, lw=1.0, label=None):
    first = True
    for edge in edges:
        xs = [p[0] for p in edge]
        ys = [p[1] for p in edge]
        ax.plot(xs, ys, color=color, alpha=alpha, linewidth=lw,
                label=label if first else None)
        first = False


def visualise(gt_edges, pred_edges, save_path: str,
              title: str = "Graph Evaluation"):
    """Overlay GT (green) and predicted (red) edges on a white canvas."""
    all_pts = []
    for edges in [gt_edges, pred_edges]:
        for edge in edges:
            all_pts.extend(edge)

    if not all_pts:
        return

    all_pts = np.array(all_pts)
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)

    fig_w = max(8, (x_max - x_min) / 100)
    fig_h = max(6, (y_max - y_min) / 100)
    fig_w = min(fig_w, 30)
    fig_h = min(fig_h, 24)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor='black')
    ax.set_facecolor('black')

    _draw_edges(ax, gt_edges,   color='#00FF66', alpha=0.7, lw=0.8, label='Ground Truth')
    _draw_edges(ax, pred_edges, color='#FF4444', alpha=0.7, lw=0.8, label='Predicted')

    ax.set_xlim(x_min - 20, x_max + 20)
    ax.set_ylim(y_min - 20, y_max + 20)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title(title, color='white', fontsize=12)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

    gt_patch   = mpatches.Patch(color='#00FF66', label='Ground Truth')
    pred_patch = mpatches.Patch(color='#FF4444', label='Predicted')
    ax.legend(handles=[gt_patch, pred_patch], loc='upper right',
              facecolor='#222222', edgecolor='white', labelcolor='white',
              fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Visualisation saved → {save_path}")


# ─── Main evaluation ─────────────────────────────────────────────────────────

def evaluate_graph_pair(gt_path: str,
                         pred_path: str,
                         output_csv: Optional[str] = None,
                         output_vis: Optional[str] = None,
                         sample_spacing: float = SAMPLE_SPACING,
                         buffer_thresholds: List[int] = BUFFER_THRESHOLDS,
                         junction_threshold: float = JUNCTION_THRESHOLD
                         ) -> pd.DataFrame:
    """
    Full evaluation of one (GT, prediction) pair.

    Args:
        gt_path:           Path to ground-truth GeoJSON.
        pred_path:         Path to predicted GeoJSON.
        output_csv:        If given, save result row to this CSV.
        output_vis:        If given, save overlay visualisation PNG here.
        sample_spacing:    Pixels between sampled points (default 2).
        buffer_thresholds: Buffer distances in pixels for geometric metrics.
        junction_threshold: Radius in pixels for junction matching.

    Returns:
        Single-row DataFrame with all metrics.
    """
    name = Path(pred_path).stem
    print(f"\n{'='*60}")
    print(f"  Evaluating: {name}")
    print(f"  GT   → {gt_path}")
    print(f"  Pred → {pred_path}")

    gt_edges   = load_edges_from_geojson(gt_path)
    pred_edges = load_edges_from_geojson(pred_path)

    print(f"  GT edges: {len(gt_edges)}   Pred edges: {len(pred_edges)}")

    # 1. Topology stats
    gt_stats   = compute_topology_stats(gt_edges,   prefix="GT_")
    pred_stats = compute_topology_stats(pred_edges, prefix="Pred_")
    ratio_stats = {
        "Ratio_Edges":        round(pred_stats["Pred_Edges"]        / max(gt_stats["GT_Edges"], 1),        3),
        "Ratio_Total_Length": round(pred_stats["Pred_Total_Length"] / max(gt_stats["GT_Total_Length"], 1), 3),
    }

    # 2. Geometric metrics (buffer-based)
    print(f"  Sampling points (spacing={sample_spacing}px)…")
    gt_pts   = sample_points_from_edges(gt_edges,   sample_spacing)
    pred_pts = sample_points_from_edges(pred_edges, sample_spacing)
    print(f"  GT pts: {len(gt_pts):,}   Pred pts: {len(pred_pts):,}")

    geo_metrics = compute_geometric_metrics(gt_pts, pred_pts, buffer_thresholds)

    # 3. Junction metrics
    junc_metrics = compute_junction_metrics(gt_edges, pred_edges, junction_threshold)

    # Combine
    all_metrics = {"Map": name, **gt_stats, **pred_stats, **ratio_stats,
                   **geo_metrics, **junc_metrics}

    df = pd.DataFrame([all_metrics])

    # Print summary
    print(f"\n  --- Topology ---")
    print(f"  GT   : {gt_stats['GT_Edges']} edges, {gt_stats['GT_Nodes']} nodes, "
          f"{gt_stats['GT_Total_Length']:.0f}px total length")
    print(f"  Pred : {pred_stats['Pred_Edges']} edges, {pred_stats['Pred_Nodes']} nodes, "
          f"{pred_stats['Pred_Total_Length']:.0f}px total length")
    print(f"  Length ratio (pred/gt): {ratio_stats['Ratio_Total_Length']:.3f}")

    print(f"\n  --- Geometric (buffer) ---")
    for t in buffer_thresholds:
        p = geo_metrics[f"Geo_Precision@{t}px"]
        r = geo_metrics[f"Geo_Recall@{t}px"]
        f = geo_metrics[f"Geo_F1@{t}px"]
        print(f"  @{t:>3}px  P={p:.3f}  R={r:.3f}  F1={f:.3f}")

    print(f"\n  --- Junctions (radius={junction_threshold}px) ---")
    print(f"  GT={junc_metrics['GT_Junctions']}  Pred={junc_metrics['Pred_Junctions']}  "
          f"P={junc_metrics['Junction_Precision']:.3f}  "
          f"R={junc_metrics['Junction_Recall']:.3f}  "
          f"F1={junc_metrics['Junction_F1']:.3f}")

    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\n  Results saved → {output_csv}")

    if output_vis:
        os.makedirs(os.path.dirname(output_vis) or '.', exist_ok=True)
        visualise(gt_edges, pred_edges, output_vis, title=name)

    return df


def evaluate_batch(pairs: List[Tuple[str, str]],
                   output_dir: str,
                   **kwargs) -> pd.DataFrame:
    """
    Evaluate multiple (gt_path, pred_path) pairs and save a combined CSV.

    Args:
        pairs:      List of (gt_path, pred_path) tuples.
        output_dir: Directory for output CSV and visualisations.
        **kwargs:   Forwarded to evaluate_graph_pair.

    Returns:
        Combined DataFrame with one row per map.
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for gt_path, pred_path in pairs:
        name = Path(pred_path).stem
        out_csv = os.path.join(output_dir, f"{name}_eval.csv")
        out_vis = os.path.join(output_dir, f"{name}_eval.png")
        try:
            df = evaluate_graph_pair(gt_path, pred_path,
                                     output_csv=out_csv,
                                     output_vis=out_vis,
                                     **kwargs)
            rows.append(df)
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    if not rows:
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True)

    # Append a mean row
    numeric_cols = combined.select_dtypes(include=np.number).columns
    mean_row     = combined[numeric_cols].mean().to_frame().T
    mean_row["Map"] = "MEAN"
    combined = pd.concat([combined, mean_row], ignore_index=True)

    combined_path = os.path.join(output_dir, "graph_eval_combined.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\n{'='*60}")
    print(f"  Combined results saved → {combined_path}")
    return combined


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Single pair ──────────────────────────────────────────────────────────
    GT_PATH   = 'data/Output/Melbourne/Truths/Melbourne/Melbourne_Truth_roads.geojson'
    PRED_PATH = 'data/Output/Melbourne_4-03/Vectorisation/roads.geojson'
    OUT_DIR   = 'data/Output/Melbourne/Truths/Melbourne/Evaluation'

    evaluate_graph_pair(
        gt_path   = GT_PATH,
        pred_path = PRED_PATH,
        output_csv = os.path.join(OUT_DIR, 'mel.csv'),
        output_vis = os.path.join(OUT_DIR, 'mel.png'),
    )

    # ── Batch example (comment out / edit as needed) ─────────────────────────
    # PAIRS = [
    #     ('/path/to/Truths/Annandale/...geojson',  '/path/to/Annandale.../roads.geojson'),
    #     ('/path/to/Truths/Enfield/...geojson',    '/path/to/Enfield.../roads.geojson'),
    #     ('/path/to/Truths/Macdonaldtown/...geojson', '/path/to/Macdonaldtown.../roads.geojson'),
    # ]
    # evaluate_batch(PAIRS, output_dir='/path/to/Evaluation')
