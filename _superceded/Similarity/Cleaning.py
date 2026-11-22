import json
from shapely.geometry import shape, mapping

def truncate_coords(geom, precision=5):
    """Recursively rounds coordinates in a GeoJSON geometry."""
    if 'coordinates' in geom:
        geom['coordinates'] = round_recursive(geom['coordinates'], precision)
    return geom

def round_recursive(coords, precision):
    if isinstance(coords, (int, float)):
        return round(coords, precision)
    return [round_recursive(c, precision) for c in coords]

# --- SETTINGS ---
INPUT_FILE = 'data/Output/1903-20-01-2026/GeoLoc/Sydney_1903_Geolocated.geojson'
OUTPUT_FILE = 'data/Output/1903-20-01-2026/GeoLoc/Sydney_1903_Geolocated_slim.geojson'
SIMPLIFY_TOLERANCE = 0.0001  # Adjust based on your needs
PRECISION = 5                # Number of decimal places

with open(INPUT_FILE, 'r') as f:
    data = json.load(f)

for feature in data['features']:
    # 1. Simplify the geometry (reduce point count)
    geom = shape(feature['geometry'])
    simplified_geom = geom.simplify(SIMPLIFY_TOLERANCE, preserve_topology=True)
    
    # 2. Convert back to dict and round decimals (reduce string length)
    feature['geometry'] = truncate_coords(mapping(simplified_geom), PRECISION)

# 3. Write with separators to save more space (removes extra whitespace)
with open(OUTPUT_FILE, 'w') as f:
    json.dump(data, f, separators=(',', ':'))

print(f"Done! Check {OUTPUT_FILE}")