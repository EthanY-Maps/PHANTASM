# Run from End_To_End/:
#   streamlit run graph_editor.py

import json
import math
import copy
import numpy as np
import streamlit as st
import networkx as nx
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Road Graph Editor")

# ---------------------------------------------------------------------------
# Internal graph format helpers
# ---------------------------------------------------------------------------
# Graph stored as: {'nodes': {id: {'x': float, 'y': float}}, 'edges': [(u, v, length)]}

def euclidean(ax, ay, bx, by):
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def graph_to_nx(graph_dict):
    G = nx.Graph()
    for nid, attrs in graph_dict["nodes"].items():
        G.add_node(nid, x=attrs["x"], y=attrs["y"])
    for u, v, length in graph_dict["edges"]:
        G.add_edge(u, v, weight=length)
    return G


def nx_to_dict(G):
    nodes = {n: {"x": G.nodes[n]["x"], "y": G.nodes[n]["y"]} for n in G.nodes()}
    edges = [(u, v, G[u][v].get("weight", 1.0)) for u, v in G.edges()]
    return {"nodes": nodes, "edges": edges}


def geojson_from_dict(graph_dict):
    features = []
    for nid, attrs in graph_dict["nodes"].items():
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [attrs["x"], attrs["y"]]},
            "properties": {"id": str(nid)},
        })
    for u, v, length in graph_dict["edges"]:
        n_u = graph_dict["nodes"][u]
        n_v = graph_dict["nodes"][v]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[n_u["x"], n_u["y"]], [n_v["x"], n_v["y"]]],
            },
            "properties": {"id": f"{u}-{v}", "length": length},
        })
    return {"type": "FeatureCollection", "features": features}


def load_geojson_to_dict(path: str):
    with open(path, "r") as f:
        data = json.load(f)

    features = data.get("features", [])
    nodes = {}
    edges_raw = []

    # Collect explicit Point nodes
    for feat in features:
        geom = feat.get("geometry", {})
        props = feat.get("properties", {})
        if geom.get("type") == "Point":
            coords = geom["coordinates"]
            nid = props.get("id", f"{coords[0]:.6f},{coords[1]:.6f}")
            nodes[str(nid)] = {"x": float(coords[0]), "y": float(coords[1])}

    def snap_or_create(coord):
        cx, cy = float(coord[0]), float(coord[1])
        for nid, attrs in nodes.items():
            if abs(attrs["x"] - cx) < 1e-6 and abs(attrs["y"] - cy) < 1e-6:
                return nid
        new_id = f"{cx:.6f},{cy:.6f}"
        nodes[new_id] = {"x": cx, "y": cy}
        return new_id

    for feat in features:
        geom = feat.get("geometry", {})
        props = feat.get("properties", {})
        if geom.get("type") == "LineString":
            coords = geom["coordinates"]
            if len(coords) < 2:
                continue
            u = snap_or_create(coords[0])
            v = snap_or_create(coords[-1])
            length = props.get("length", euclidean(
                float(coords[0][0]), float(coords[0][1]),
                float(coords[-1][0]), float(coords[-1][1]),
            ))
            if u != v:
                edges_raw.append((u, v, float(length)))

    return {"nodes": nodes, "edges": edges_raw}


def graph_stats(graph_dict):
    G = graph_to_nx(graph_dict)
    n = G.number_of_nodes()
    e = G.number_of_edges()
    comps = nx.number_connected_components(G) if n > 0 else 0
    dead_ends = sum(1 for _, d in G.degree() if d == 1)
    return n, e, comps, dead_ends


def next_node_id(graph_dict):
    ids = list(graph_dict["nodes"].keys())
    # Try numeric
    numeric = []
    for nid in ids:
        try:
            numeric.append(int(nid))
        except (ValueError, TypeError):
            pass
    if numeric:
        return str(max(numeric) + 1)
    return str(len(ids))


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def init_state():
    defaults = {
        "graph": None,          # current graph dict
        "original": None,       # original loaded graph dict
        "history": [],          # list of (description, graph_snapshot) — max 10
        "op_log": [],           # list of operation description strings
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def push_history(description: str):
    """Save current graph state before a mutation."""
    snap = copy.deepcopy(st.session_state.graph)
    st.session_state.history.append((description, snap))
    if len(st.session_state.history) > 10:
        st.session_state.history.pop(0)
    st.session_state.op_log.append(description)


# ---------------------------------------------------------------------------
# Plotly graph figure
# ---------------------------------------------------------------------------

def build_figure(graph_dict):
    if graph_dict is None:
        fig = go.Figure()
        fig.update_layout(height=700, title="No graph loaded")
        return fig

    nodes = graph_dict["nodes"]
    edges = graph_dict["edges"]
    G = graph_to_nx(graph_dict)

    # Edge traces
    ex, ey = [], []
    for u, v, _ in edges:
        if u in nodes and v in nodes:
            ex += [nodes[u]["x"], nodes[v]["x"], None]
            ey += [nodes[u]["y"], nodes[v]["y"], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ex, y=ey,
        mode="lines",
        line=dict(color="lightgrey", width=1.5),
        hoverinfo="none",
        showlegend=False,
    ))

    # Colour by degree
    degree = dict(G.degree())
    dead_ends_nodes, junction_nodes, through_nodes = [], [], []
    for nid in nodes:
        d = degree.get(nid, 0)
        if d == 1:
            dead_ends_nodes.append(nid)
        elif d >= 3:
            junction_nodes.append(nid)
        else:
            through_nodes.append(nid)

    def node_scatter(node_list, color, name):
        if not node_list:
            return None
        xs = [nodes[n]["x"] for n in node_list]
        ys = [nodes[n]["y"] for n in node_list]
        sizes = [max(6, 4 + degree.get(n, 0) * 2) for n in node_list]
        texts = [
            f"ID: {n}<br>Degree: {degree.get(n, 0)}<br>x: {nodes[n]['x']:.4f}<br>y: {nodes[n]['y']:.4f}"
            for n in node_list
        ]
        return go.Scatter(
            x=xs, y=ys,
            mode="markers",
            marker=dict(color=color, size=sizes, line=dict(width=0.5, color="white")),
            text=texts,
            hovertemplate="%{text}<extra></extra>",
            name=name,
        )

    for trace in [
        node_scatter(dead_ends_nodes, "crimson", "Dead-ends (deg=1)"),
        node_scatter(junction_nodes, "steelblue", "Junctions (deg≥3)"),
        node_scatter(through_nodes, "grey", "Through-nodes (deg=2)"),
    ]:
        if trace is not None:
            fig.add_trace(trace)

    n_nodes = len(nodes)
    n_edges = len(edges)
    fig.update_layout(
        title=f"Road Graph — {n_nodes} nodes, {n_edges} edges",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, scaleanchor="x"),
        height=700,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="white",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Load Graph")
    file_path_input = st.text_input("GeoJSON file path", value="", placeholder="/path/to/network.geojson", key="load_path")
    load_btn = st.button("Load", type="primary")

    if load_btn:
        path = file_path_input.strip()
        if not path:
            st.error("Enter a file path first.")
        else:
            try:
                gd = load_geojson_to_dict(path)
                st.session_state.graph = gd
                st.session_state.original = copy.deepcopy(gd)
                st.session_state.history = []
                st.session_state.op_log = [f"Loaded from: {path}"]
                n, e, c, de = graph_stats(gd)
                st.success(f"Loaded {n} nodes, {e} edges, {c} components.")
            except FileNotFoundError:
                st.error(f"File not found: {path}")
            except Exception as ex:
                st.error(f"Error loading file: {ex}")

    st.divider()

    # --- Live stats panel ---
    st.header("Graph Stats")
    if st.session_state.graph is not None:
        n, e, c, de = graph_stats(st.session_state.graph)
        st.metric("Nodes", n)
        st.metric("Edges", e)
        st.metric("Components", c)
        st.metric("Dead-ends", de)
    else:
        st.info("No graph loaded.")

    st.divider()

    # --- Edit operations ---
    st.header("Edit Operations")

    # Remove nodes
    st.subheader("Remove Nodes")
    remove_ids_input = st.text_input("Node IDs (comma-separated)", key="remove_ids_input")
    if st.button("Remove Selected"):
        if st.session_state.graph is None:
            st.error("Load a graph first.")
        else:
            ids_to_remove = [x.strip() for x in remove_ids_input.split(",") if x.strip()]
            missing = [x for x in ids_to_remove if x not in st.session_state.graph["nodes"]]
            if missing:
                st.error(f"Nodes not found: {missing}")
            elif ids_to_remove:
                push_history(f"Remove nodes: {ids_to_remove}")
                for nid in ids_to_remove:
                    del st.session_state.graph["nodes"][nid]
                st.session_state.graph["edges"] = [
                    (u, v, w) for u, v, w in st.session_state.graph["edges"]
                    if u not in ids_to_remove and v not in ids_to_remove
                ]
                st.success(f"Removed {len(ids_to_remove)} node(s).")
                st.rerun()

    st.divider()

    # Add node
    st.subheader("Add Node")
    add_x = st.number_input("X coordinate", value=0.0, format="%.4f", key="add_x")
    add_y = st.number_input("Y coordinate", value=0.0, format="%.4f", key="add_y")
    if st.button("Add Node"):
        if st.session_state.graph is None:
            st.error("Load a graph first.")
        else:
            new_id = next_node_id(st.session_state.graph)
            push_history(f"Add node {new_id} at ({add_x:.4f}, {add_y:.4f})")
            st.session_state.graph["nodes"][new_id] = {"x": float(add_x), "y": float(add_y)}
            st.success(f"Added node {new_id}.")
            st.rerun()

    st.divider()

    # Add edge
    st.subheader("Add Edge")
    edge_a = st.text_input("Node ID A", key="edge_a")
    edge_b = st.text_input("Node ID B", key="edge_b")
    if st.button("Add Edge"):
        if st.session_state.graph is None:
            st.error("Load a graph first.")
        elif not edge_a.strip() or not edge_b.strip():
            st.error("Enter both node IDs.")
        elif edge_a.strip() not in st.session_state.graph["nodes"]:
            st.error(f"Node A '{edge_a.strip()}' not found.")
        elif edge_b.strip() not in st.session_state.graph["nodes"]:
            st.error(f"Node B '{edge_b.strip()}' not found.")
        elif edge_a.strip() == edge_b.strip():
            st.error("A and B must be different nodes.")
        else:
            u, v = edge_a.strip(), edge_b.strip()
            # Check duplicate
            existing = {(e[0], e[1]) for e in st.session_state.graph["edges"]}
            if (u, v) in existing or (v, u) in existing:
                st.warning("Edge already exists.")
            else:
                n_u = st.session_state.graph["nodes"][u]
                n_v = st.session_state.graph["nodes"][v]
                length = euclidean(n_u["x"], n_u["y"], n_v["x"], n_v["y"])
                push_history(f"Add edge {u}-{v} (length={length:.4f})")
                st.session_state.graph["edges"].append((u, v, length))
                st.success(f"Added edge {u}-{v}.")
                st.rerun()

    st.divider()

    # Auto-connect endpoints
    st.subheader("Auto-connect Dangling Ends")
    snap_dist = st.slider("Snap distance (px)", min_value=10, max_value=200, value=50, step=5)
    if st.button("Connect Dangling Ends"):
        if st.session_state.graph is None:
            st.error("Load a graph first.")
        else:
            G_tmp = graph_to_nx(st.session_state.graph)
            deg1_nodes = [n for n, d in G_tmp.degree() if d == 1]
            nodes_dict = st.session_state.graph["nodes"]
            added = []
            existing_edges = {(e[0], e[1]) for e in st.session_state.graph["edges"]}
            existing_edges |= {(e[1], e[0]) for e in st.session_state.graph["edges"]}

            for i in range(len(deg1_nodes)):
                for j in range(i + 1, len(deg1_nodes)):
                    u, v = deg1_nodes[i], deg1_nodes[j]
                    if (u, v) in existing_edges:
                        continue
                    nu, nv = nodes_dict[u], nodes_dict[v]
                    d = euclidean(nu["x"], nu["y"], nv["x"], nv["y"])
                    if d <= snap_dist:
                        added.append((u, v, d))
                        existing_edges.add((u, v))
                        existing_edges.add((v, u))

            if added:
                push_history(f"Connect dangling ends (snap={snap_dist}): added {len(added)} edges")
                st.session_state.graph["edges"].extend(added)
                st.success(f"Connected {len(added)} dangling pair(s).")
                st.rerun()
            else:
                st.info("No dangling pairs found within snap distance.")

    st.divider()

    # Remove isolated nodes
    st.subheader("Remove Isolated Nodes")
    if st.button("Remove Isolated Nodes"):
        if st.session_state.graph is None:
            st.error("Load a graph first.")
        else:
            G_tmp = graph_to_nx(st.session_state.graph)
            isolated = [n for n, d in G_tmp.degree() if d == 0]
            if isolated:
                push_history(f"Remove {len(isolated)} isolated nodes")
                for nid in isolated:
                    del st.session_state.graph["nodes"][nid]
                st.success(f"Removed {len(isolated)} isolated node(s).")
                st.rerun()
            else:
                st.info("No isolated nodes found.")

    st.divider()

    # Undo
    st.subheader("Undo")
    if st.button("Undo Last Operation"):
        if not st.session_state.history:
            st.info("Nothing to undo.")
        else:
            desc, snap = st.session_state.history.pop()
            st.session_state.graph = snap
            st.session_state.op_log.append(f"UNDO: {desc}")
            st.success(f"Undone: {desc}")
            st.rerun()

    st.divider()

    # Save
    st.header("Save GeoJSON")
    save_path_input = st.text_input("Output file path", value="", placeholder="/path/to/output.geojson", key="save_path")
    if st.button("Save GeoJSON"):
        if st.session_state.graph is None:
            st.error("No graph to save.")
        elif not save_path_input.strip():
            st.error("Enter an output path.")
        else:
            try:
                geojson_data = geojson_from_dict(st.session_state.graph)
                with open(save_path_input.strip(), "w") as f:
                    json.dump(geojson_data, f, indent=2)
                n, e, c, de = graph_stats(st.session_state.graph)
                st.success(f"Saved {n} nodes, {e} edges to {save_path_input.strip()}")
                st.session_state.op_log.append(f"Saved to: {save_path_input.strip()}")
            except Exception as ex:
                st.error(f"Save failed: {ex}")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Road Network Graph Editor")

# Graph figure
fig = build_figure(st.session_state.graph)
st.plotly_chart(fig, use_container_width=True, key="graph")

# Operation log
st.subheader("Operation Log")
log_text = "\n".join(st.session_state.op_log) if st.session_state.op_log else "(no operations yet)"
st.text_area(
    "History",
    value=log_text,
    height=180,
    disabled=True,
    key="op_log_area",
    label_visibility="collapsed",
)

if st.session_state.graph is None:
    st.info("Load a GeoJSON file from the sidebar to begin editing.")
    st.markdown("""
**Expected GeoJSON format:**
- `Point` features with optional `id` property for explicit nodes
- `LineString` features with optional `length` property for edges

**Node colours:**
- Red = dead-ends (degree 1)
- Blue = junctions (degree >= 3)
- Grey = through-nodes (degree 2)
""")
