import folium
import geopandas as gpd
import os

# --- CONFIGURATION ---
HIST_FILE = "data/Output/1903-20-01-2026/GeoLoc/Sydney_1903_Geolocated_slim.geojson"
TRUTH_FILE = 'data/GreaterSydneyStreetsEvolution/DATA/Sydney_roads.shp'
# Your Best Prediction Path
HIGH_FILE = "data/Output/1903-20-01-2026/Evaluation/Session_V5-3_Pareto_20260130_141417/Trial_202/high_confidence.geojson"
MED_FILE = "data/Output/1903-20-01-2026/Evaluation/Session_V5-3_Pareto_20260130_141417/Trial_202/medium_confidence.geojson"
LOW_FILE = "data/Output/1903-20-01-2026/Evaluation/Session_V5-3_Pareto_20260130_141417/Trial_202/low_confidence.geojson"


OUTPUT_MAP = "validation_map_072_precision.html"

# OPTIMIZATION SETTINGS
SIMPLIFY_TOLERANCE = 0.000001  # ~10 meters.
LIMIT_ROWS = None            

def create_validation_map():
    print("--- Generative High-Performance Map ---")
    
    # 1. Load Data
    print("Loading data...")
    if os.path.exists(HIST_FILE):
        gdf_hist = gpd.read_file(HIST_FILE)
    else:
        print(f"❌ Historic file not found: {HIST_FILE}")
        return

    if os.path.exists(TRUTH_FILE):
        gdf_truth = gpd.read_file(TRUTH_FILE)
    else:
        print(f"❌ Truth file not found: {TRUTH_FILE}")
        return
    
    if os.path.exists(HIGH_FILE):
        gdf_high = gpd.read_file(HIGH_FILE)
    else:
        print(f"⚠️ Prediction file not found: {HIGH_FILE}")
        gdf_high = gpd.GeoDataFrame()
    
    if os.path.exists(MED_FILE):
        gdf_med = gpd.read_file(MED_FILE)
    else:
        print(f"⚠️ Prediction file not found: {MED_FILE}")
        gdf_med = gpd.GeoDataFrame()
    
    if os.path.exists(LOW_FILE):
        gdf_low = gpd.read_file(LOW_FILE)
    else:
        print(f"⚠️ Prediction file not found: {LOW_FILE}")
        gdf_low = gpd.GeoDataFrame()

    # 2. Pre-process: Reproject, Filter, SIMPLIFY
    def process_layer(gdf, name, color, style_opts):
        if gdf.empty: return None
        
        # A. Reproject
        if gdf.crs is not None and gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
            
        # B. Filter Lines Only (Removes Intersections/Points)
        gdf = gdf[gdf.geometry.type.isin(['LineString', 'MultiLineString'])]
        
        # C. Simplify Geometry
        gdf['geometry'] = gdf.geometry.simplify(SIMPLIFY_TOLERANCE, preserve_topology=True)
        
        # D. Limit Rows
        if LIMIT_ROWS and len(gdf) > LIMIT_ROWS:
            print(f"  - {name}: Truncating from {len(gdf)} to {LIMIT_ROWS} rows for speed.")
            gdf = gdf.iloc[:LIMIT_ROWS]

        # --- THE FIX IS HERE ---
        # Dynamically build the tooltip based on what columns ACTUALLY exist
        cols = gdf.columns.tolist()
        tooltip_fields = []
        tooltip_aliases = []

        # Check for 'confidence'
        if 'confidence' in cols:
            tooltip_fields.append('confidence')
            tooltip_aliases.append('Confidence')
        
        # Check for 'score'
        if 'score' in cols:
            tooltip_fields.append('score')
            tooltip_aliases.append('Score')

        # Check for 'source_id' (useful for debugging)
        if 'source_id' in cols:
            tooltip_fields.append('source_id')
            tooltip_aliases.append('Source ID')
            
        # Create tooltip object only if we found matching columns
        if tooltip_fields:
            t_tip = folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases, sticky=True)
        else:
            t_tip = None

        # Create Layer
        layer = folium.GeoJson(
            gdf,
            name=name,
            style_function=lambda x: style_opts,
            tooltip=t_tip
        )
        return layer

    # 3. Initialize Map with Canvas Rendering
    if not gdf_hist.empty:
        center = [gdf_hist.geometry.centroid.y.mean(), gdf_hist.geometry.centroid.x.mean()]
    else:
        center = [-33.8688, 151.2093]
        
    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron", prefer_canvas=True)

    # 4. Add Layers
    print("Processing Ground Truth...")
    l_truth = process_layer(gdf_truth, "Ground Truth", '#2ecc71', {'color': '#2ecc71', 'weight': 3, 'opacity': 0.3})
    if l_truth: 
        fg = folium.FeatureGroup(name="Ground Truth", show=False)
        l_truth.add_to(fg)
        fg.add_to(m)

    print("Processing Historic...")
    l_hist = process_layer(gdf_hist, "Historic (1903)", '#e74c3c', {'color': '#e74c3c', 'weight': 2, 'opacity': 0.6, 'dashArray': '4, 4'})
    if l_hist: l_hist.add_to(m)

    print("Processing Prediction...")
    l_pred = process_layer(gdf_high, "Matches (High)", '#3498db', {'color': '#3498db', 'weight': 2, 'opacity': 0.9})
    if l_pred: l_pred.add_to(m)

    l_pred = process_layer(gdf_med, "Matches (Medium)", '#3498db', {'color': '#3498db', 'weight': 2, 'opacity': 0.9})
    if l_pred: l_pred.add_to(m)

    l_pred = process_layer(gdf_low, "Matches (Low)", '#3498db', {'color': '#3498db', 'weight': 2, 'opacity': 0.9})
    if l_pred: l_pred.add_to(m)

    # 5. Save
    folium.LayerControl().add_to(m)
    m.save(OUTPUT_MAP)
    print(f"✅ Fast map saved to: {os.path.abspath(OUTPUT_MAP)}")

if __name__ == "__main__":
    create_validation_map()