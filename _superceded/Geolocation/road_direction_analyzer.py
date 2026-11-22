#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import shape, Point
from tqdm import tqdm
from collections import defaultdict
import os
from datetime import datetime


def load_road_graph(roads_path):
    """
    Load and construct road network graph from GeoJSON.
    
    Args:
        roads_path (str): Path to the GeoJSON file containing road data
        
    Returns:
        nx.Graph: NetworkX graph object representing the road network
    """
    with open(roads_path, "r") as f:
        roads_raw = json.load(f)

    # Extract features by type
    point_features = [f for f in roads_raw["features"] if f["geometry"]["type"] == "Point"]
    line_features = [f for f in roads_raw["features"] if f["geometry"]["type"] == "LineString"]

    # Initialize graph
    G = nx.Graph()
    
    # Add nodes from point features
    for feature in point_features:
        props = feature["properties"]
        if "id" not in props:
            continue
        node_id = props["id"]
        node_type = props.get("type", "link")
        coords = tuple(feature["geometry"]["coordinates"])
        
        # Initialize node with default values
        G.add_node(node_id, 
                  type=node_type, 
                  coords=coords, 
                  name="", 
                  secondary_names = [],
                  starting=False,
                  stable_count=0)

        # Initialize intersection node with default values
        if G.nodes[node_id]["type"] == "intersection":
            G.nodes[node_id]["EW"] = ""
            G.nodes[node_id]["NS"] = ""
            G.nodes[node_id]["secondary_EW"] = []
            G.nodes[node_id]["secondary_NS"] = []

    # Add edges from line features
    for feature in line_features:
        props = feature["properties"]
        if "id" not in props:
            continue
        edge_id = props["id"]
        try:
            src, tgt = edge_id.split("-")
            src, tgt = src.strip(), tgt.strip()
            if src in G.nodes and tgt in G.nodes:
                G.add_edge(src, tgt)
        except ValueError:
            continue
            
    return G


def assign_names_from_polygons(G, corrected_path, threshold=30):
    with open(corrected_path, "r") as f:
        corrected_raw = json.load(f)

    # Extract node positions and create KDTree for efficient nearest neighbor search
    node_positions = {
        nid: data["coords"]
        for nid, data in G.nodes(data=True)
        if "coords" in data and len(data["coords"]) == 2
    }
    
    # Create KDTree for efficient nearest neighbor search
    node_coords = list(node_positions.values())
    node_ids = list(node_positions.keys())
    node_tree = KDTree(node_coords)

    # Extract polygons and their centroids
    polygon_data = []
    for feature in tqdm(corrected_raw["features"]):
        geom = feature.get("geometry")
        props = feature.get("properties", {})
        if geom and geom.get("type") in ["Polygon", "MultiPolygon"]:
            try:
                poly = shape(geom)
                label = props.get("corrected_predicted_text")
                if label:
                    centroid = poly.centroid
                    polygon_data.append({
                        'centroid': (centroid.x, centroid.y),
                        'label': label
                    })
            except Exception as e:
                print(f"Skipping feature: {e}")
                continue

    # Process each polygon
    for poly_data in tqdm(polygon_data):
        centroid = poly_data['centroid']
        label = poly_data['label']
        
        # Find nearest nodes (limit to 10 nearest neighbors for efficiency)
        k = min(10, len(node_coords))  # Limit to 10 nearest neighbors
        distances, indices = node_tree.query(centroid, k=k)
        
        # Filter nodes within threshold
        valid_nodes = []
        for dist, idx in zip(distances, indices):
            if dist <= threshold:
                node_id = node_ids[idx]
                node_coord = node_coords[idx]
                valid_nodes.append({
                    'id': node_id,
                    'coord': node_coord,
                    'distance': dist,
                    'y_diff': node_coord[1] - centroid[1]  # Positive means node is above centroid
                })
        
        if valid_nodes:
            # Sort by distance first, then by y-difference (preferring nodes below centroid)
            valid_nodes.sort(key=lambda x: (x['distance'], -x['y_diff']))
            
            # Assign name to the best matching node
            best_node = valid_nodes[0]
            G.nodes[best_node['id']]["name"] = label
            G.nodes[best_node['id']]["starting"] = True

    print("Names Allocated")


def angle_between(p1, p2):
    """Returns the angle in degrees between two coordinate points."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))

def get_link_segment_nodes(graph, node_id):
    """
    Returns all nodes that are part of the same link segment as the given node.
    A link segment is defined as all nodes between two intersection/end nodes.
    """
    if graph.nodes[node_id]["type"] != "link":
        return {node_id}
        
    segment_nodes = {node_id}
    to_visit = {node_id}
    
    while to_visit:
        current = to_visit.pop()
        for neighbor in graph.neighbors(current):
            if neighbor not in segment_nodes and graph.nodes[neighbor]["type"] == "link":
                segment_nodes.add(neighbor)
                to_visit.add(neighbor)
                
    return segment_nodes

def get_link_boundary_nodes(graph, segment_nodes):
    """Returns the boundary nodes (intersections or ends) at both ends of a link segment."""
    boundary_nodes = []
    for n in segment_nodes:
        for neighbor in graph.neighbors(n):
            if neighbor not in segment_nodes and (graph.nodes[neighbor]["type"] in ["intersection", "end"]):
                boundary_nodes.append(neighbor)
    return boundary_nodes

def get_true_direction_angle(angle):
    """Returns the angle's distance from true NS (90/270) or EW (0/180)."""
    angle = (angle + 360) % 360
    ns_diff = min(abs(angle - 90), abs(angle - 270))
    ew_diff = min(abs(angle - 0), abs(angle - 180))
    return 'NS' if ns_diff < ew_diff else 'EW'

def get_intersection_edge_directions(graph, intersection_node):
    """Returns counts of NS and EW edges connected to an intersection."""
    ns_count = 0
    ew_count = 0
    edge_directions = []
    
    for neighbor in graph.neighbors(intersection_node):
        if 'direction' in graph.edges[intersection_node, neighbor]:
            direction = graph.edges[intersection_node, neighbor]['direction']
            if direction == 'NS':
                ns_count += 1
            else:
                ew_count += 1
            edge_directions.append((neighbor, direction))
    
    return ns_count, ew_count, edge_directions

def get_nodes_within_depth(graph, start_node, max_depth):
    """
    Returns all nodes within max_depth steps from start_node in a link segment.
    Only includes nodes that are part of the same link segment.
    """
    visited = {start_node}
    current_depth_nodes = {start_node}
    all_nodes = {start_node}
    
    for depth in range(max_depth):
        next_depth_nodes = set()
        for node in current_depth_nodes:
            for neighbor in graph.neighbors(node):
                if (neighbor not in visited and 
                    graph.nodes[neighbor]["type"] == "link"):
                    next_depth_nodes.add(neighbor)
                    visited.add(neighbor)
                    all_nodes.add(neighbor)
        current_depth_nodes = next_depth_nodes
        if not current_depth_nodes:
            break
    
    return all_nodes

def get_average_direction_to_intersection(graph, nodes, intersection_node):
    """
    Calculate the average direction from a set of nodes to an intersection node.
    Returns the average angle in degrees.
    """
    if not nodes:
        return 0
    
    intersection_coords = graph.nodes[intersection_node]["coords"]
    angles = []
    
    for node in nodes:
        node_coords = graph.nodes[node]["coords"]
        angle = angle_between(node_coords, intersection_coords)
        angles.append(angle)
    
    # Convert angles to radians for averaging
    angles_rad = [math.radians(angle) for angle in angles]
    
    # Calculate average using circular mean
    avg_x = sum(math.cos(angle) for angle in angles_rad) / len(angles_rad)
    avg_y = sum(math.sin(angle) for angle in angles_rad) / len(angles_rad)
    
    avg_angle = math.degrees(math.atan2(avg_y, avg_x))
    return avg_angle

def get_edge_angle_score(graph, node1, node2, target_direction):
    """Returns a score indicating how well the edge matches the target direction.
    Lower score means better match (closer to true NS or EW)."""
    # Get all nodes in the link segment
    segment_nodes = get_link_segment_nodes(graph, node2)
    
    # Calculate how many nodes deep to look
    depth = max(1, math.ceil(len(segment_nodes) / 2))
    
    # Get nodes within the calculated depth
    nodes_to_consider = get_nodes_within_depth(graph, node2, depth)
    
    # Calculate average direction from these nodes to the intersection
    avg_angle = get_average_direction_to_intersection(graph, nodes_to_consider, node1)
    
    if target_direction == 'NS':
        # Calculate distance from true north (90) or south (270)
        score = min(abs(avg_angle - 90), abs(avg_angle - 270))
    else:  # EW
        # Calculate distance from true east (0) or west (180)
        score = min(abs(avg_angle - 0), abs(avg_angle - 180))
    
    return score

def get_edge_length(graph, edge):
    """Calculate the length of an edge in the graph."""
    node1, node2 = edge
    coords1 = graph.nodes[node1]['coords']
    coords2 = graph.nodes[node2]['coords']
    return math.sqrt((coords2[0] - coords1[0])**2 + (coords2[1] - coords1[1])**2)

def get_link_segment_length(graph, start_node):
    """Calculate the total length of a link segment from a starting node."""
    segment_nodes = get_link_segment_nodes(graph, start_node)
    total_length = 0
    
    # Create a set of edges in the segment
    segment_edges = set()
    for node in segment_nodes:
        for neighbor in graph.neighbors(node):
            if neighbor in segment_nodes:
                edge = tuple(sorted([node, neighbor]))
                segment_edges.add(edge)
    
    # Sum up the lengths of all edges in the segment
    for edge in segment_edges:
        total_length += get_edge_length(graph, edge)
    
    return total_length

def get_direction_score(angle):
    """Calculate how close an angle is to true NS or EW.
    Returns a tuple of (NS_score, EW_score) where higher scores mean closer to that direction."""
    angle = (angle + 360) % 360
    
    # Calculate distance from true NS (90/270)
    ns_diff = min(abs(angle - 90), abs(angle - 270))
    ns_score = abs(1 - (ns_diff / 90))  # 1 when perfectly NS, 0 when perfectly EW
    
    # Calculate distance from true EW (0/180)
    ew_diff = min(abs(angle - 0), abs(angle - 180))
    ew_score = abs(1 - (ew_diff / 90))  # 1 when perfectly EW, 0 when perfectly NS
    
    # Round to 3 decimal places for more precision
    return round(ns_score, 3), round(ew_score, 3)

def balance_intersection_directions(graph, intersection_node, edge_properties):
    """Balances the NS/EW directions at an intersection, ensuring 2 NS and 2 EW for first four edges."""
    # Get all edges at this intersection
    intersection_edges = []
    for neighbor in graph.neighbors(intersection_node):
        edge = (intersection_node, neighbor)
        # Find the edge properties
        for ep in edge_properties:
            if ep['edge'] == edge:
                intersection_edges.append(ep)
                break
    
    if not intersection_edges:
        return
    
    # Sort edges by their direction scores
    ns_edges = []
    ew_edges = []
    
    for ep in intersection_edges:
        if ep['ns_score'] > ep['ew_score']:
            ns_edges.append(ep)
        else:
            ew_edges.append(ep)
    
    # Sort each list by their respective scores
    ns_edges.sort(key=lambda x: x['ns_score'], reverse=True)
    ew_edges.sort(key=lambda x: x['ew_score'], reverse=True)
    
    # Assign directions to first four edges
    assigned_count = 0
    ns_count = 0
    ew_count = 0
    
    # First assign top 2 NS edges
    for ep in ns_edges:
        if assigned_count < 4 and ns_count < 2:
            graph.edges[ep['edge']]['direction'] = 'NS'
            assigned_count += 1
            ns_count += 1
    
    # Then assign top 2 EW edges
    for ep in ew_edges:
        if assigned_count < 4 and ew_count < 2:
            graph.edges[ep['edge']]['direction'] = 'EW'
            assigned_count += 1
            ew_count += 1
    
    # Assign remaining edges based on their scores
    for ep in intersection_edges:
        edge = ep['edge']
        if 'direction' not in graph.edges[edge]:
            direction = 'NS' if ep['ns_score'] > ep['ew_score'] else 'EW'
            graph.edges[edge]['direction'] = direction

def propagate_road_directions(graph):
    """
    Propagates road directions from intersections and ends through link segments.
    1. First assigns directions to edges based on length and direction scores
    2. Then propagates those directions one step at a time through adjacent edges
    """
    print("Starting direction propagation...")
    
    # Phase 1: Assign initial directions based on length and direction scores
    print("Phase 1: Assigning initial directions...")
    visited_edges = set()
    
    # First identify all intersection and end nodes
    boundary_nodes = [node_id for node_id, data in graph.nodes(data=True) 
                     if data["type"] in ["intersection", "end"]]
    print(f"Found {len(boundary_nodes)} boundary nodes")
    
    # Collect all edges adjacent to intersections/ends with their properties
    edge_properties = []
    for node_id in tqdm(boundary_nodes, desc="Analyzing edges"):
        for neighbor in graph.neighbors(node_id):
            edge = (node_id, neighbor)
            if edge not in visited_edges:
                try:
                    # Calculate angle and direction scores
                    angle = angle_between(graph.nodes[node_id]["coords"], 
                                       graph.nodes[neighbor]["coords"])
                    ns_score, ew_score = get_direction_score(angle)
                    
                    # Calculate segment length
                    segment_length = get_link_segment_length(graph, neighbor)
                    
                    edge_properties.append({
                        'edge': edge,
                        'length': segment_length,
                        'ns_score': ns_score,
                        'ew_score': ew_score,
                        'angle': angle
                    })
                    visited_edges.add(edge)
                except Exception as e:
                    print(f"Error processing edge {edge}: {str(e)}")
    
    # Sort edges by length
    edge_properties.sort(key=lambda x: x['length'], reverse=True)
    
    # Assign directions based on scores
    print("Assigning directions based on scores...")
    assigned_edges = set()
    
    # First assign top NS and EW edges
    for i in range(2):  # Do this twice for top 2 in each direction
        # Find best unassigned NS edge
        ns_edges = [ep for ep in edge_properties 
                   if ep['edge'] not in assigned_edges 
                   and ep['ns_score'] > ep['ew_score']]
        if ns_edges:
            best_ns = max(ns_edges, key=lambda x: x['ns_score'])
            graph.edges[best_ns['edge']]['direction'] = 'NS'
            assigned_edges.add(best_ns['edge'])
        
        # Find best unassigned EW edge
        ew_edges = [ep for ep in edge_properties 
                   if ep['edge'] not in assigned_edges 
                   and ep['ew_score'] > ep['ns_score']]
        if ew_edges:
            best_ew = max(ew_edges, key=lambda x: x['ew_score'])
            graph.edges[best_ew['edge']]['direction'] = 'EW'
            assigned_edges.add(best_ew['edge'])
    
    # Balance directions at each intersection
    print("Balancing directions at intersections...")
    for node_id in tqdm(boundary_nodes, desc="Balancing intersections"):
        if graph.nodes[node_id]["type"] == "intersection":
            balance_intersection_directions(graph, node_id, edge_properties)
    
    # Assign remaining edges based on direction scores
    for ep in edge_properties:
        if ep['edge'] not in assigned_edges:
            direction = 'NS' if ep['ns_score'] > ep['ew_score'] else 'EW'
            graph.edges[ep['edge']]['direction'] = direction
            assigned_edges.add(ep['edge'])
    
    print(f"Assigned initial directions to {len(assigned_edges)} edges")
    
    # Phase 2: Step-by-step propagation
    print("\nPhase 2: Propagating directions step by step...")
    
    # Keep track of edges that need to be processed in the next step
    edges_to_process = set(assigned_edges)
    iteration = 0
    max_iterations = 100  # Safety limit
    
    while edges_to_process and iteration < max_iterations:
        iteration += 1
        print(f"\nPropagation step {iteration}")
        next_edges = set()
        
        for edge in tqdm(edges_to_process, desc=f"Step {iteration}"):
            if "direction" not in graph.edges[edge]:
                continue
                
            direction = graph.edges[edge]["direction"]
            node1, node2 = edge
            
            # Try to propagate to adjacent edges of both nodes
            for node in [node1, node2]:
                for neighbor in graph.neighbors(node):
                    # Skip if neighbor is an intersection or end
                    if graph.nodes[neighbor]["type"] in ["intersection", "end"]:
                        continue
                        
                    new_edge = (node, neighbor)
                    
                    # Skip if edge already has a direction
                    if "direction" in graph.edges[new_edge]:
                        continue
                        
                    if new_edge not in visited_edges:
                        graph.edges[new_edge]["direction"] = direction
                        visited_edges.add(new_edge)
                        next_edges.add(new_edge)
        
        edges_to_process = next_edges
        print(f"Step {iteration} complete: {len(next_edges)} new edges processed")
        
        if not next_edges:
            print("No more edges to process")
            break
    
    if iteration >= max_iterations:
        print("Warning: Reached maximum iteration limit")
    
    print(f"\nProcessing complete:")
    print(f"- Completed {iteration} propagation steps")
    print(f"- Total edges with directions: {len(visited_edges)}")


def visualize_road_directions(graph,plots_dir, interactive=False):
    """Visualize the road network with color-coded directions."""
    # Position map
    pos = {n: (d["coords"][0], d["coords"][1]) for n, d in graph.nodes(data=True)}

    # Get intersection nodes
    intersection_nodes = [n for n, d in graph.nodes(data=True) if d["type"] == "intersection"]
    
    # Edge groups by direction
    edges_NS = [(u, v) for u, v in graph.edges if graph.edges[u, v].get('direction') == 'NS']
    edges_EW = [(u, v) for u, v in graph.edges if graph.edges[u, v].get('direction') == 'EW']
    edges_undirected = [(u, v) for u, v in graph.edges if 'direction' not in graph.edges[u, v]]

    # Create figure with interactive features if requested
    if interactive:
        plt.ion()
        fig = plt.figure(figsize=(80, 100))
        ax = fig.add_subplot(111)
        ax.set_title("Road Network Direction Analysis (Interactive - Use zoom tool to explore)")
    else:
        plt.figure(figsize=(80, 100))

    # Draw edges with different colors based on direction
    nx.draw_networkx_edges(graph, pos, edgelist=edges_NS, edge_color='blue', alpha=0.6, width=1.5, label='North-South')
    nx.draw_networkx_edges(graph, pos, edgelist=edges_EW, edge_color='red', alpha=0.6, width=1.5, label='East-West')
    nx.draw_networkx_edges(graph, pos, edgelist=edges_undirected, edge_color='gray', alpha=0.3, width=1.0, label='Undirected')

    # Draw all nodes in light gray
    nx.draw_networkx_nodes(graph, pos, node_color='lightgray', node_size=20, alpha=0.3)
    
    # Draw intersection nodes on top in black
    nx.draw_networkx_nodes(graph, pos, nodelist=intersection_nodes, node_color='black', node_size=50)

    # Add angle and score labels on adjacent link nodes
    for intersection in intersection_nodes:
        for neighbor in graph.neighbors(intersection):
            if graph.nodes[neighbor]["type"] == "link":
                # Calculate angle and scores
                angle = angle_between(graph.nodes[intersection]["coords"], 
                                   graph.nodes[neighbor]["coords"])
                ns_score, ew_score = get_direction_score(angle)
                
                # Get the assigned direction if any
                direction = graph.edges[intersection, neighbor].get('direction', 'undirected')
                
                # Format the text more concisely with absolute scores and 2 decimal places
                info = f"{angle:.0f}°\n{abs(ns_score):.2f}/{abs(ew_score):.2f}\n{direction}"
                
                # Get position of the link node
                node_pos = pos[neighbor]
                
                # Add slight offset to avoid overlapping with node
                offset = 0.00005
                x = node_pos[0] + offset
                y = node_pos[1] + offset
                
                plt.text(x, y, info, 
                        fontsize=7,
                        bbox=dict(facecolor='white', alpha=0.3, edgecolor='none'),
                        ha='left', va='bottom')

    plt.legend()
    plt.axis("off")
    
    # Create plots directory if it doesn't exist
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"road_directions_{timestamp}.png"
    filepath = os.path.join(plots_dir, filename)
    
    # Save the plot
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {filepath}")
    
    if interactive:
        plt.show(block=True)
    else:
        plt.show()
    plt.close()  # Close the figure to free memory

def road_direction(ROADS_PATH,plots_dir):
    # Load and process the road network
    print("Loading road network...")
    G = load_road_graph(ROADS_PATH)
    
    print("Classifying road directions...")
    propagate_road_directions(G)
    
    print("Visualizing results...")
    #visualize_road_directions(G, plots_dir, interactive=True)
    
    return G

if __name__ == "__main__":
    # Update these paths to match your data location
    ROADS_PATH = "data/Output/1877_08_05/graphing/roads.geojson"
    plots_dir = "data/Output/1877_08_05/plots"
    
    road_direction(ROADS_PATH,plots_dir) 