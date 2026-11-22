import json
import numpy as np
import pandas as pd
from geopy.distance import geodesic
import math
import os
from datetime import datetime
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional
import networkx as nx
from collections import defaultdict
from scipy.spatial import KDTree


def load_geojson(file_path: str) -> Dict:
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def node_distance(node1,node2):

    
    coords_1 = (node1['pos'][1], node1['pos'][0])  # (latitude, longitude)
    coords_2 = (node2['pos'][1], node2['pos'][0])  # (latitude, longitude)

    try:
        # Calculate the geodesic distance between the two coordinates
        return geodesic(coords_1, coords_2).meters
    except Exception as e:
        # Return standard distance if geodesic fails
        p1 = np.array(coords_1)
        p2 = np.array(coords_2)
        return np.linalg.norm(p1 - p2)



def node_direction_vector(node1, node2): 
    coords_1 = (node1['pos'][1], node1['pos'][0])  # (latitude, longitude)
    coords_2 = (node2['pos'][1], node2['pos'][0])  # (latitude, longitude)
    # Calculate the direction vector
    direction_vector = np.array(coords_2) - np.array(coords_1)
    # Normalize the direction vector
    norm = np.linalg.norm(direction_vector)
    if norm == 0:
        return np.array([0, 0])
    return direction_vector / norm

def find_shape_paths(G):
    shape_paths = {}

    # Filter intersection and end nodes
    intersection_end_nodes = [n for n, data in G.nodes(data=True)
                              if data.get('type') in {'intersection', 'endpoint'}]

    for start_node in tqdm(intersection_end_nodes, desc="Finding shape paths"):
        for neighbor in G.neighbors(start_node):
            if G.nodes[neighbor].get('type') != 'shape':
                if G.nodes[neighbor].get('type') in {'intersection'}:
                    # Store Path if the neighbor is an intersection or end node
                    path = [start_node, neighbor]
                    shape_paths[(start_node, neighbor)] = {
                        "shape_nodes": [start_node, neighbor],
                        "edges": [(start_node, neighbor)]
                    }
                    continue

            path = [start_node]
            shape_nodes = []
            edges = []

            current_node = neighbor
            previous_node = start_node

            while True:
                shape_nodes.append(current_node)
                edges.append((previous_node, current_node))
                
                neighbors = [n for n in G.neighbors(current_node) if n != previous_node]

                if len(neighbors) != 1:
                    # Either dead end or fork — stop
                    break

                next_node = neighbors[0]
                next_type = G.nodes[next_node].get('type')

                if next_type == 'shape':
                    previous_node = current_node
                    current_node = next_node
                elif next_type in {'intersection', 'endpoint'}:
                    # Found a valid terminating node
                    end_node = next_node
                    edges.append((current_node, end_node))
                    path.append(end_node)
                    shape_paths[(start_node, end_node)] = {
                        "shape_nodes": shape_nodes.copy(),
                        "edges": edges.copy()
                    }
                    break
                else:
                    break

    return shape_paths


def save_geojson(graph: Dict, output_file: str,G):
    features = []


    for node in graph['nodes']:
        if node['type'] in ['intersection', 'end']:
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [node['pos'][0], node['pos'][1]]
                },
                'properties': {
                    'id': node['id'],
                    'type': node['type'],
                    'neighbours': node.get('neighbours', {}),
                    
                }
            }
            features.append(feature)

    for edge in graph['edges']:
        if 'coordinates_true' in edge:
            line_coords = edge['coordinates_true']

        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': line_coords
    
            },
            'properties': {
                'type': 'edge',
                'length_m': edge['length_m'],
                'source': edge['source'],
                'target': edge['target']
            }
        }
        features.append(feature)

    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }

    with open(output_file, 'w') as file:
        json.dump(geojson, file, indent=4)

def merge_close_intersection_nodes(G, intersection_nodes, distance_threshold=25):
    # Merge ultra close intersections using KDTree
    distance_threshold = 15  # meters
    
    if intersection_nodes:
        # Build coordinate array for KDTree
        intersection_coords = []
        intersection_lookup = {}
        
        for i, node_id in enumerate(intersection_nodes):
            pos = G.nodes[node_id]['pos']
            # Convert to approximate meters using simple projection
            # (This is approximate but sufficient for close distances)
            lat, lon = pos[1], pos[0]
            x = lon * 111320 * math.cos(math.radians(lat))  # meters
            y = lat * 110540  # meters
            intersection_coords.append([x, y])
            intersection_lookup[i] = node_id
        
        intersection_coords = np.array(intersection_coords)
        
        # Build KDTree
        tree = KDTree(intersection_coords)
        
        # Find pairs within threshold distance
        intersections_to_merge = []
        processed = set()
        
        for i in range(len(intersection_coords)):
            if i in processed:
                continue
                
            # Find all neighbors within threshold
            neighbors = tree.query_ball_point(intersection_coords[i], distance_threshold)
            
            for j in neighbors:
                if j != i and j not in processed:
                    node1 = intersection_lookup[i]
                    node2 = intersection_lookup[j]
                    
                    # Verify with actual geodesic distance
                    actual_distance = node_distance(G.nodes[node1], G.nodes[node2])
                    if actual_distance < distance_threshold:
                        intersections_to_merge.append((node1, node2))
                        processed.add(j)
            
            processed.add(i)
        
        # Merge close intersections
        for node1, node2 in intersections_to_merge:
            if node1 in G.nodes and node2 in G.nodes:
                # Calculate midpoint position
                pos1 = G.nodes[node1]['pos']
                pos2 = G.nodes[node2]['pos']
                midpoint = ((pos1[0] + pos2[0]) / 2, (pos1[1] + pos2[1]) / 2)
                
                # Get all neighbors of both nodes
                neighbors1 = set(G.neighbors(node1)) - {node2}
                neighbors2 = set(G.neighbors(node2)) - {node1}
                
                # Store edge data for reconstruction
                edges_to_recreate = []
                
                # Collect edge data from node1
                for neighbor in neighbors1:
                    if G.has_edge(node1, neighbor):
                        edge_data = G.edges[node1, neighbor].copy()
                        edges_to_recreate.append((neighbor, edge_data))
                
                # Collect edge data from node2
                for neighbor in neighbors2:
                    if G.has_edge(node2, neighbor):
                        edge_data = G.edges[node2, neighbor].copy()
                        edges_to_recreate.append((neighbor, edge_data))
                
                # Remove the old nodes
                G.remove_node(node1)
                G.remove_node(node2)
                
                # Create new merged node
                merged_id = f"merged_{node1}_{node2}"
                G.add_node(merged_id, pos=midpoint, type='intersection')
                
                # Recreate edges with updated coordinates
                for neighbor, edge_data in edges_to_recreate:
                    if neighbor in G.nodes:  # Make sure neighbor still exists
                        # Update coordinates_true if it exists
                        if 'coordinates_true' in edge_data:
                            coords_true = edge_data['coordinates_true'].copy()
                            # Update the endpoint that was merged
                            if coords_true[0] == pos1 or coords_true[0] == pos2:
                                coords_true[0] = midpoint
                            if coords_true[-1] == pos1 or coords_true[-1] == pos2:
                                coords_true[-1] = midpoint
                            edge_data['coordinates_true'] = coords_true
                        
                        # Update basic coordinates
                        if 'coordinates' in edge_data:
                            coords = edge_data['coordinates'].copy()
                            if coords[0] == pos1 or coords[0] == pos2:
                                coords[0] = midpoint
                            if coords[1] == pos1 or coords[1] == pos2:
                                coords[1] = midpoint
                            edge_data['coordinates'] = coords
                        
                        # Recalculate length
                        if 'coordinates_true' in edge_data:
                            coords_for_length = edge_data['coordinates_true']
                            total_length = 0.0
                            for i in range(len(coords_for_length) - 1):
                                p1, p2 = coords_for_length[i], coords_for_length[i + 1]
                                total_length += geodesic((p1[1], p1[0]), (p2[1], p2[0])).meters
                            edge_data['length_m'] = total_length
                        
                        G.add_edge(merged_id, neighbor, **edge_data)



def simplify_geojson(graph,output_file):
    G = nx.Graph()
    edge_lengths = []
    component_nodes = []

    for feature in tqdm(graph['features'], desc="Processing features"):
        if feature['geometry']['type'] == 'Point':
            coords = feature['geometry']['coordinates']
            node_id = feature['properties']['id']
            properties = feature['properties']
            G.add_node(node_id, pos=(coords[0], coords[1]), **properties)

    for node in tqdm(G.nodes(data=True), desc="Processing nodes"):
        adjacent_nodes = G.nodes[node[0]]['adjacent_nodes'] if 'adjacent_nodes' in node[1] else []
        for neighbor in adjacent_nodes:
            if neighbor != node[0]:
                node_distance_value = node_distance(G.nodes[node[0]], G.nodes[neighbor])
                node_direction = node_direction_vector(G.nodes[node[0]], G.nodes[neighbor])
                if node_distance_value < 200:  # Threshold distance in meters
                    edge_lengths.append(node_distance_value)
                    G.add_edge(node[0], neighbor, length_m=node_distance_value, coordinates=[G.nodes[node[0]]['pos'], G.nodes[neighbor]['pos']], type='temp_edge')

    # Remove nodes with no edges
    nodes_to_remove = [node for node, degree in G.degree() if degree == 0]
    G.remove_nodes_from(nodes_to_remove)

    # Find Connected Components
    components = list(nx.connected_components(G))
    for component in components:
        num_nodes = len(component)
        if num_nodes < 100:
            G.remove_nodes_from(component)
        else:
            component_nodes.append(num_nodes)


    shape_paths = find_shape_paths(G)


    # Define Edges between Intersection/End Nodes:
    for shape_path in shape_paths.items():
        
        boundary_nodes = shape_path[0]
        path = shape_path[1]  # Get the shape nodes and edges

        start_coords = G.nodes[boundary_nodes[0]]['pos']
        end_coords = G.nodes[boundary_nodes[1]]['pos']
        intermediate_coords = [G.nodes[node]['pos'] for node in path['shape_nodes']]
        line_coords = [start_coords] + intermediate_coords + [end_coords]

        G.add_edge(boundary_nodes[0], boundary_nodes[1],
                   length_m=np.sum([G.edges[edge]['length_m'] for edge in path['edges']]),
                   coordinates=[G.nodes[boundary_nodes[0]]['pos'], G.nodes[boundary_nodes[1]]['pos']],
                   type='edge', coordinates_true = line_coords, start=boundary_nodes[0], end=boundary_nodes[1])


    shape_nodes_raw = [node for node, data in G.nodes(data=True) if data.get('type') == 'shape']


    # Delete Edges with type 'edge'
    edges_to_remove = [edge for edge in G.edges(data=True) if edge[2].get('type') == 'temp_edge']
    G.remove_edges_from(edges_to_remove)


    # Remove edges of length less than 10 meters
    edges_to_remove = [edge for edge in G.edges(data=True) if edge[2].get('length_m', float('inf')) < 20]
    G.remove_edges_from(edges_to_remove)

    # Remove nodes with no edges on intersection or end nodes

    nodes_to_remove = [node for node, degree in G.degree() if degree == 0 and G.nodes[node].get('type') in {'intersection', 'end'}]
    G.remove_nodes_from(nodes_to_remove)

    # Remove Selflooping Edges
    self_loops = list(nx.selfloop_edges(G))
    G.remove_edges_from(self_loops)

    end_nodes = [node for node, type in G.nodes(data='type') if type == 'end']
    intersection_nodes = [node for node, type in G.nodes(data='type') if type == 'intersection']
    edge_lengths = [data['length_m'] for u, v, data in G.edges(data=True) if 'length_m' in data]

    for node in G.nodes(data=True):
        neighbour = {}
        for neighbor in G.neighbors(node[0]):
            neighbour_distance = node_distance(node[1], G.nodes[neighbor])
            neighbour_direction = node_direction_vector(node[1], G.nodes[neighbor])
            neighbour[neighbor] = {
                'distance': neighbour_distance,
                'direction': neighbour_direction.tolist()  # Convert numpy array to list for JSON serialization
            }
        G.nodes[node[0]]['neighbours'] = neighbour

    # Merge close intersection nodes
    for i in range(3):  # Run multiple times to ensure all close nodes are merged
        merge_close_intersection_nodes(G, intersection_nodes)
        intersection_nodes = [node for node, type in G.nodes(data='type') if type == 'intersection']

    merge_close_intersection_nodes(G, end_nodes, distance_threshold=100)
    end_nodes = [node for node, type in G.nodes(data='type') if type == 'endpoint']


    print("Node Summary:", {
        'total': G.number_of_nodes() - len(shape_nodes_raw),
        'shape_nodes': len(shape_nodes_raw),
        'end_nodes': len(end_nodes),
        'intersection_nodes': len(intersection_nodes)
    })
    print("Number of edges in the graph:", G.number_of_edges())
    print("Edge Length Summary:", {
        'min': np.min(edge_lengths), 
        'max': np.max(edge_lengths),
        'mean': np.mean(edge_lengths),
        'std': np.std(edge_lengths)})

    print("Number of connected components in the graph:", len(components))
    print("Component Node Count Summary:", {
        'min': np.min(component_nodes),
        'max': np.max(component_nodes),
        'mean': np.mean(component_nodes),
        'std': np.std(component_nodes)})

    save_geojson(nx.node_link_data(G,edges = "edges"), output_file, G)


def simplification_main(input_file: str, output_file: str):
    """
    Main function to load, simplify, and save the GeoJSON graph.
    """
    graph = load_geojson(input_file)
    simplify_geojson(graph, output_file)



if __name__ == "__main__":
    input_file = "data/Output/1903_27_10/1903_Original/graphing/roads.geojson"
    output_file = "data/Output/1903_27_10/1903_Original/graphing/roads_simple.geojson"

    simplification_main(input_file, output_file)


