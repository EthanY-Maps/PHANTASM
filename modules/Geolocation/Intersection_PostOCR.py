import os
import networkx as nx
from shapely.geometry import shape, Point, mapping
from tqdm import tqdm
from itertools import combinations
from rapidfuzz import fuzz, process, utils
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from shapely.strtree import STRtree

from utils import load_json, save_json
# ── 1. OPTIMIZED DATA LOADING ──

def load_canonical_data(file_path):
    """Loads canonical names into a flat list for vectorized matching."""
    print(f"--- Loading Reference Data: {file_path} ---")
    if not os.path.exists(file_path):
        print(f"  ⚠️  WARNING: canonical names file not found at {file_path}. "
              "All junctions will produce zero matches — check the upstream "
              "intersection-name extraction step.")
        return []

    canonical_list = []
    with open(file_path, 'r') as f:
        for line in f:
            full = line.strip().upper()
            if " & " in full:
                # Pre-process for token_sort_ratio
                canonical_list.append(full)

    if not canonical_list:
        print(f"  ⚠️  WARNING: canonical file at {file_path} contained 0 valid intersection entries "
              "(expected lines with ' & ' separators).")
    else:
        print(f"Loaded {len(canonical_list)} canonical intersections.")
    return canonical_list

# ── 2. PARALLEL WORKER LOGIC ──

def match_junction(node_data, canonical_list, fuzzy_weight: float = 0.7):
    """Per-junction worker: try each road-name pair against the canonical list, return the best match."""
    node_id, coords, roads_at_junction = node_data
    graph_weight = 1.0 - fuzzy_weight

    node_results = []
    road_names = list(roads_at_junction.keys())

    for r1, r2 in combinations(road_names, 2):
        query = f"{r1} & {r2}"
        # token_sort_ratio + default_process: "A & B" matches "B & A" canonical name.
        match_result = process.extractOne(
            query, canonical_list,
            scorer=fuzz.token_sort_ratio, processor=utils.default_process,
        )
        if match_result:
            best_db_name, best_fuzz, _ = match_result
            graph_avg = (roads_at_junction[r1] + roads_at_junction[r2]) / 2
            final_conf = (graph_avg * graph_weight) + ((best_fuzz / 100.0) * fuzzy_weight)
            node_results.append({
                "id": node_id, "raw": query, "match": best_db_name,
                "score": final_conf, "fuzz": best_fuzz, "coords": coords,
            })

    if not node_results:
        return None
    return max(node_results, key=lambda x: x['score'])

# ── 3. MAIN PROCESSOR ──

def process_intersections(roads_geojson, canon_txt_path, output_path,
                          fuzzy_weight: float = 0.7,
                          verification_threshold: int = 80):
    canonical_list = load_canonical_data(canon_txt_path)
    
    print("\n--- Building Road Graph ---")
    road_data = load_json(roads_geojson)

    G = nx.Graph()
    # Single pass: nodes first within the same loop so node attributes from
    # explicit Point features survive edge processing (the edge pass would
    # otherwise overwrite `coords` from LineString endpoints).
    edge_features = []
    for feat in road_data.get('features', []):
        geom = feat.get('geometry') or {}
        if geom.get('type') == 'Point':
            props = feat['properties']
            G.add_node(props['id'], coords=geom['coordinates'], **props)
        else:
            edge_features.append(feat)

    for feat in edge_features:
        props = feat['properties']
        u, v = props['id'].split('-')
        G.add_edge(u, v, **props)
        G.edges[u, v]['coords'] = feat['geometry']['coordinates']
        # Only fill in node coords from LineString endpoints when the node
        # didn't get its coords from a Point feature in the first pass.
        if 'coords' not in G.nodes[u]:
            G.nodes[u]['coords'] = feat['geometry']['coordinates'][0]
        if 'coords' not in G.nodes[v]:
            G.nodes[v]['coords'] = feat['geometry']['coordinates'][-1]


    # Prepare data for parallel processing (filter dead-ends and no-names first)
    tasks = []

    for node in G.nodes():
        incident = list(G.edges(node, data=True))
        if len(incident) < 2: continue
        
        roads_at_junction = {}
        for u, v, d in incident:
            name = d.get("name")
            if name:
                roads_at_junction[name] = max(roads_at_junction.get(name, 0), d.get("confidence", 0))
        
        if len(roads_at_junction) >= 2:
            try:
                tasks.append((node, G.nodes[node]['coords'], roads_at_junction))
            except KeyError:
                print(f"Skipping node {node!r}: missing 'coords' attribute")




    # EXECUTION
    found_features = []
    func = partial(match_junction, canonical_list=canonical_list, fuzzy_weight=fuzzy_weight)

    # Avoid spawning many subprocesses (each re-imports rapidfuzz/networkx/shapely)
    # for a handful of junctions — that overhead dominates over the work itself.
    # Scale workers with task count and cap at CPU count.
    cpu_count = os.cpu_count() or 1
    n_workers = min(cpu_count, max(1, len(tasks) // 100))

    if n_workers <= 1 or len(tasks) < 32:
        print(f"Processing {len(tasks)} valid junctions serially "
              f"(too few to benefit from multiprocessing)...")
        results = [func(t) for t in tqdm(tasks)]
    else:
        print(f"Processing {len(tasks)} valid junctions with {n_workers} workers...")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(tqdm(executor.map(func, tasks), total=len(tasks)))

    # Filter None results and format GeoJSON
    for res in results:
        if res:
            found_features.append({
                "type": "Feature",
                "id" : res["id"],
                "geometry": mapping(Point(res['coords'])),
                "properties": {
                    "canonical_name": res["match"],
                    "confidence": round(res["score"], 3),
                    "is_verified": res["fuzz"] > verification_threshold,
                    "debug_raw": res["raw"]
                }
            })


    for feature in found_features:
        if feature["id"] in G.nodes():
            G.nodes[feature["id"]]["canonical_name"] = feature["properties"]["canonical_name"]
            G.nodes[feature["id"]]["confidence"] = feature["properties"]["confidence"]
            G.nodes[feature["id"]]["is_verified"] = feature["properties"]["is_verified"]
            G.nodes[feature["id"]]["debug_raw"] = feature["properties"]["debug_raw"]


# 1. Prepare output path
    combined_output_path = os.path.join(output_path, "graph_network_combined.geojson")
    all_features = []

    # 2. Extract Nodes (Intersections)
    for node_id, data in G.nodes(data=True):
        if 'coords' in data:
            all_features.append({
                "type": "Feature",
                "id": f"{node_id}",
                "geometry": {
                    "type": "Point", 
                    "coordinates": data['coords']
                },
                "properties": {
                    "feature_type": data.get("type"),
                    "canonical_name": data.get("canonical_name"),
                    "confidence": data.get("confidence", 0),
                    "is_verified": data.get("is_verified", False),
                    "debug_raw": data.get("debug_raw", ""),
                    "original_id": node_id
                }
            })

    # 3. Extract Edges (Roads)
    for u, v, data in G.edges(data=True):
        if 'coords' in G.nodes[u] and 'coords' in G.nodes[v]:
            all_features.append({
                "type": "Feature",
                "id": f"{u}-{v}",
                "geometry": {
                    "type": "LineString",
                    "coordinates": G.edges[u, v]['coords']
                },
                "properties": {
                    "feature_type": "road",
                    "start": u,
                    "end": v,
                    **data # Spreads existing edge properties (name, weight, etc.)
                }
            })


    active_nodes = {f["properties"]["start"] for f in all_features if f["properties"]["feature_type"] == "road"} | \
                   {f["properties"]["end"] for f in all_features if f["properties"]["feature_type"] == "road"}

    all_features = [f for f in all_features if f["properties"]["feature_type"] == "road" or f["properties"]["original_id"] in active_nodes]

    # 4. Save to a single file
    output_geojson = {
        "type": "FeatureCollection",
        "features": all_features
    }

    save_json(combined_output_path, output_geojson, indent=2)

    print(f"\n--- Export Complete ---")
    print(f"Saved {len(all_features)} combined features to: {combined_output_path}")



    return combined_output_path





# ── RUNNER ──

if __name__ == "__main__":
    process_intersections(
        "data/Output/1903-20-01-2026/GeoLoc/named_roads_final.geojson",  # Output from previous step
        "data/Output/1903-20-01-2026/GeoLoc/intersection_names.txt", # Your text file
        "data/Output/1903-20-01-2026/GeoLoc/"
    )