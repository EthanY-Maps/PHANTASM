import math
import os
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
from shapely.strtree import STRtree
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from shapely.geometry import shape, mapping
from tqdm import tqdm
from collections import Counter

from utils import load_json, save_json

# ── 1. GEOMETRY HELPERS ──

def calculate_bearing(p1, p2):
    """Calculates degrees (0-360) between two points."""
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) % 360

def get_polygon_orientation(poly_geom):
    """Finds the primary axis of a polygon using the minimum rotated rectangle."""
    rect = poly_geom.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    side_lengths = [math.dist(coords[i], coords[i+1]) for i in range(len(coords)-1)]
    max_idx = side_lengths.index(max(side_lengths))
    p1, p2 = coords[max_idx], coords[max_idx+1]
    return calculate_bearing(p1, p2) % 180

def get_vector_away_from_node(edge_geom, node_coords):
    """Bearing of the edge moving away from `node_coords`. Samples 10% along the line for stability."""
    coords = list(edge_geom.coords)
    if len(coords) < 2: return 0.0

    dist_start = math.dist(coords[0], node_coords)
    dist_end = math.dist(coords[-1], node_coords)
    if dist_start < dist_end:
        p1 = coords[0]
        p2_geom = edge_geom.interpolate(0.1, normalized=True)
    else:
        p1 = coords[-1]
        p2_geom = edge_geom.interpolate(0.9, normalized=True)
    return calculate_bearing(p1, (p2_geom.x, p2_geom.y))

# ── 2. AUTO-CROPPED DEBUGGER ──

def plot_graph_state(graph, poly_data, output_path, title):
    plt.figure(figsize=(25, 25))
    ax = plt.gca()
    
    lines, colors, widths = [], [], []
    all_x, all_y = [], []
    
    for u, v, d in graph.edges(data=True):
        geom_coords = list(d['geometry'].coords)
        lines.append(geom_coords)
        xs, ys = zip(*geom_coords)
        all_x.extend(xs); all_y.extend(ys)
        
        if "name" not in d:
            colors.append((0.2, 0.2, 0.2, 0.1))
            widths.append(0.5)
        else:
            conf = d.get("confidence", 0)
            colors.append((1 - conf, conf, 0, 0.8))
            widths.append(2.5 + (conf * 4.0))
            m = d['geometry'].interpolate(0.5, normalized=True)
            plt.text(m.x, m.y, d['name'], fontsize=5, ha='center', weight='bold', alpha=0.7)

    ax.add_collection(LineCollection(lines, colors=colors, linewidths=widths))
    
    if all_x and all_y:
        min_x, max_x = np.percentile(all_x, [0.5, 99.5])
        min_y, max_y = np.percentile(all_y, [0.5, 99.5])
        buffer = (max_y - min_y) * 0.02
        ax.set_xlim(min_x - buffer, max_x + buffer)
        ax.set_ylim(min_y - buffer, max_y + buffer)
        ax.set_aspect('equal', adjustable='datalim')

    plt.axis('off')
    plt.title(title, fontsize=24)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

# ── 3. STATISTICS REPORTER ──

def print_graph_statistics(graph, initial_edge_count=None):
    """Calculates and prints comprehensive stats about the graph state."""
    
    current_edges = graph.number_of_edges()
    total_nodes = graph.number_of_nodes()
    
    if current_edges == 0:
        print("Graph is empty. No stats to report.")
        return

    # --- Edge Stats ---
    named_edges = [d for u, v, d in graph.edges(data=True) if "name" in d]
    unnamed_edges_count = current_edges - len(named_edges)
    
    # Calculate Length Coverage (Physical distance is more important than edge count)
    total_length = sum(d['geometry'].length for u, v, d in graph.edges(data=True))
    named_length = sum(d['geometry'].length for d in named_edges)
    length_coverage_pct = (named_length / total_length * 100) if total_length > 0 else 0
    
    # --- Confidence Stats ---
    confidences = [d.get("confidence", 0) for d in named_edges]
    avg_confidence = np.mean(confidences) if confidences else 0.0
    median_confidence = np.median(confidences) if confidences else 0.0
    
    # --- Name Distribution ---
    all_names = [d["name"] for d in named_edges]
    unique_names = set(all_names)
    name_counts = Counter(all_names)
    top_names = name_counts.most_common(5)
    
    # --- Node Coverage ---
    covered_nodes = 0
    for node in graph.nodes():
        if any("name" in d for _, _, d in graph.edges(node, data=True)):
            covered_nodes += 1
    node_coverage_pct = (covered_nodes / total_nodes * 100) if total_nodes > 0 else 0

    print("\n" + "="*60)
    print("                FINAL NETWORK STATISTICS")
    print("="*60)
    
    print(f"\n--- 1. NETWORK TOTALS ---")
    if initial_edge_count is not None:
        print(f"Starting Edges    : {initial_edge_count:,}")
        print(f"Filtered Edges    : {initial_edge_count - current_edges:,} (removed)")
    print(f"Final Edges       : {current_edges:,}")
    print(f"Final Nodes       : {total_nodes:,}")

    print(f"\n--- 2. NAMING COVERAGE ---")
    print(f"Named Edges       : {len(named_edges):,} ({len(named_edges)/current_edges*100:.1f}%)")
    print(f"Unnamed Edges     : {unnamed_edges_count:,} ({unnamed_edges_count/current_edges*100:.1f}%)")
    print(f"Named Nodes       : {covered_nodes:,} ({node_coverage_pct:.1f}%)")
    print(f"Length Coverage   : {length_coverage_pct:.1f}% of total network distance")
    print(f"Named Length      : {named_length:.1f} meters")
    print(f"total length      : {total_length:.1f} meters")

    print(f"\n--- 3. DATA QUALITY ---")
    print(f"Unique Names Found: {len(unique_names):,}")
    print(f"Avg Confidence    : {avg_confidence:.4f}")
    print(f"Median Confidence : {median_confidence:.4f}")

    print(f"\n--- 4. TOP 5 DETECTED ROADS ---")
    if top_names:
        for name, count in top_names:
            print(f"  - {name:<20} : {count} segments")
    else:
        print("  (No names assigned)")
    
    print("="*60 + "\n")

# ── 4. CORE LOGIC ──

def remove_small_islands(graph, min_size=5):
    components = list(nx.connected_components(graph))
    initial_nodes = len(graph.nodes)
    for component in components:
        if len(component) < min_size:
            graph.remove_nodes_from(component)
    print(f"Cleanup: Removed {initial_nodes - len(graph.nodes)} stray nodes.")

def assign_initial_names(graph, poly_data, threshold=100):
    print(f"Step 1: Anchoring & Initial Candidate Logging...")
    excluded_terms = ("no_high_confidence_match", "suffix_found")
    
    valid_features = [
        f for f in poly_data["features"] 
        if f.get("geometry") and not any(
            term in str(f.get("properties", {}).get("match_status", "")).lower() 
            for term in excluded_terms
        )
    ]
    
    edge_list = list(graph.edges(data=True))
    geometries = [d['geometry'] for u, v, d in edge_list]
    tree = STRtree(geometries)
    
    for u, v in graph.edges():
        graph.edges[u, v]["candidates"] = {}

    for feature in tqdm(valid_features, desc="Initial Voting"):
        poly_geom = shape(feature["geometry"])
        props = feature.get("properties", {})
        label = str(props.get("corrected_predicted_text") or props.get("text") or "").strip().upper()
        if not label or len(label) < 2: continue
        
        centroid = poly_geom.centroid
        poly_orient = get_polygon_orientation(poly_geom)
        indices = tree.query(centroid.buffer(threshold))
        
        for idx in indices:
            edge_geom = geometries[idx]
            r_pts = list(edge_geom.coords)
            road_orient = calculate_bearing(r_pts[0], r_pts[-1]) % 180
            angle_diff = min(abs(road_orient - poly_orient), 180 - abs(road_orient - poly_orient))
            
            if angle_diff > 40: continue # Limit initial angle mismatch
            
            dist = centroid.distance(edge_geom)
            score = math.exp(-(dist**2) / (2 * (threshold/2)**2)) * math.cos(math.radians(angle_diff))
            if poly_geom.intersects(edge_geom.buffer(2)): score *= 40.0 
            
            u, v, _ = edge_list[idx]
            graph.edges[u, v]["candidates"][label] = max(graph.edges[u, v]["candidates"].get(label, 0), score)

    for u, v, d in graph.edges(data=True):
        if d["candidates"]:
            best_name, best_score = max(d["candidates"].items(), key=lambda x: x[1])
            d["name"] = best_name
            d["confidence"] = min(1.0, best_score / (best_score + 1.0))

def propagate_labels(graph, poly_data, plot_dir, iterations=20, angle_tol=40, overtake_multiplier=1.3):
    print("Step 2: Long-Range Propagation...")
    plot_graph_state(graph, poly_data, os.path.join(plot_dir, "debug_initial.png"), "Initial State")

    for i in range(iterations):
        changes = 0
        snapshot = []

        for node in graph.nodes():
            incident = list(graph.edges(node, data=True))
            sources = [e for e in incident if "name" in e[2] and e[2]["confidence"] > 0.01]
            
            for u_s, v_s, d_s in sources:
                s_name = d_s["name"]
                s_conf = d_s["confidence"]
                
                neighbor_node = u_s if v_s == node else v_s
                in_bear = get_vector_away_from_node(d_s['geometry'], graph.nodes[neighbor_node]['coords'])
                
                for u_t, v_t, d_t in incident:
                    if (u_t == u_s and v_t == v_s): continue
                    
                    out_bear = get_vector_away_from_node(d_t['geometry'], graph.nodes[node]["coords"])
                    dev = abs((out_bear - (in_bear + 180)) % 360 - 180)
                    
                    if dev <= angle_tol:
                        momentum = math.cos(math.radians(dev))
                        new_conf = s_conf * 0.985 * momentum
                        
                        d_t["candidates"][s_name] = max(d_t["candidates"].get(s_name, 0), new_conf)
                        
                        current_name = d_t.get("name")
                        current_conf = d_t.get("confidence", 0)
                        
                        if current_name is None or (s_name != current_name and new_conf > current_conf * overtake_multiplier):
                            snapshot.append(((u_t, v_t), s_name, new_conf))
                            changes += 1

        for (u, v), name, conf in snapshot:
            graph.edges[u, v]["name"] = name
            graph.edges[u, v]["confidence"] = conf
        
        if i % 4 == 0 or changes == 0:
            plot_graph_state(graph, poly_data, os.path.join(plot_dir, f"debug_iter_{i}.png"), f"Iteration {i}")
        
        print(f" > Iteration {i}: {changes} updates applied.")
        if changes == 0: break

def run_analysis(roads_path, labels_path, out_path,
                 min_component_size: int = 5,
                 initial_radius: float = 100.0,
                 propagation_iterations: int = 20,
                 propagation_angle_tolerance: float = 40.0,
                 propagation_overtake_multiplier: float = 1.3):
    os.makedirs(out_path, exist_ok=True)
    plot_dir = os.path.join(out_path, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    r_data = load_json(roads_path)
    p_data = load_json(labels_path)

    G = nx.Graph()
    other_features = []

    for feat in r_data["features"]:
        geom_type = feat["geometry"]["type"]
        geom = shape(feat["geometry"])

        if "LineString" in geom_type:
            u, v = str(geom.coords[0]), str(geom.coords[-1])
            G.add_edge(u, v, geometry=geom, **feat["properties"])
            G.nodes[u]["coords"], G.nodes[v]["coords"] = geom.coords[0], geom.coords[-1]
        else:
            other_features.append(feat)

    initial_edge_count = G.number_of_edges()

    remove_small_islands(G, min_size=min_component_size)
    assign_initial_names(G, p_data, threshold=initial_radius)
    propagate_labels(G, p_data, plot_dir,
                     iterations=propagation_iterations,
                     angle_tol=propagation_angle_tolerance,
                     overtake_multiplier=propagation_overtake_multiplier)

    # --- PRINT STATS HERE ---
    print_graph_statistics(G, initial_edge_count=initial_edge_count)
    # ------------------------

    # Convert back to features
    output_features = []
    for u, v, d in G.edges(data=True):
        feat = {
            "type": "Feature",
            "geometry": mapping(d["geometry"]),
            "properties": {k: v for k, v in d.items() if k != "geometry"}
        }
        output_features.append(feat)

    output_features.extend(other_features)

    final_path = os.path.join(out_path, "named_roads.geojson")
    save_json(final_path, {"type": "FeatureCollection", "features": output_features})
    
    print(f"Processing Complete. Included {len(other_features)} point/other features.")
    print(f"File saved to: {final_path}")

    return final_path

if __name__ == "__main__":
    # Update these paths to your local environment
    run_analysis(
        "data/Output/1890_10-02-26/Graphing/roads.geojson",
        "data/Output/1890_10-02-26/GeoLoc/OCR.geojson",
        "data/Output/1890_10-02-26/GeoLoc/"
    )