import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from streamlit_searchbox import st_searchbox  # New library for autocomplete
from geopy.geocoders import Nominatim

# 1. Page Configuration
st.set_page_config(layout="wide", page_title="Map Bounding Box Tool")

st.title("📍 City Bounding Box Extractor")
st.markdown("Search for a city, select the correct suggestion, and draw your box.")

# Initialize the Geocoder once
geolocator = Nominatim(user_agent="geo_box_tool_v2")

# --- AUTOCOMPLETE FUNCTION ---
def search_places(search_term: str):
    """
    This function is called every time you type in the search box.
    It queries OpenStreetMap and returns a list of matching places.
    """
    if not search_term:
        return []
    
    try:
        # Get up to 5 suggestions
        locations = geolocator.geocode(search_term, exactly_one=False, limit=5)
        if locations:
            # Return a list of tuples: (Label to show, Value to return)
            return [(loc.address, loc) for loc in locations]
        return []
    except Exception:
        return []

# 2. Input with Auto-Fill (Searchbox)
# The user types -> search_places runs -> dropdown appears
selected_location = st_searchbox(
    search_places,
    key="city_search",
    label="Search City (Type to see suggestions):",
    placeholder="e.g. Sydney..."
)

# 3. Logic: Only proceed if a location was selected
if selected_location:
    # Use the selected location object directly (no need to geocode again)
    location = selected_location
    
    # 4. Create the Map centered on the selected location
    # We use a slightly tighter zoom (10) to encompass the city better
    m = folium.Map(location=[location.latitude, location.longitude], zoom_start=10)

    # 5. Add the Draw Tool
    draw = Draw(
        draw_options={
            'polyline': False,
            'polygon': False, 
            'circle': False, 
            'marker': False, 
            'circlemarker': False,
            'rectangle': True 
        },
        edit_options={'edit': False}
    )
    draw.add_to(m)

    # 6. Display Map
    output = st_folium(m, width=1000, height=600)

    # 7. Process Drawing
    if output.get("all_drawings"):
        last_drawing = output["all_drawings"][-1]
        geometry = last_drawing['geometry']['coordinates'][0]
        
        longitudes = [point[0] for point in geometry]
        latitudes = [point[1] for point in geometry]

        min_lon, max_lon = min(longitudes), max(longitudes)
        min_lat, max_lat = min(latitudes), max(latitudes)

        # 8. Display Results
        st.divider()
        st.subheader("📦 Bounding Box Coordinates")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("**South-West Corner (Bottom-Left)**")
            st.code(f"{min_lat}, {min_lon}")
        with col2:
            st.info("**North-East Corner (Top-Right)**")
            st.code(f"{max_lat}, {max_lon}")

        st.json({
            "south": min_lat,
            "west": min_lon,
            "north": max_lat,
            "east": max_lon
        })
else:
    st.info("Start typing a city name above to begin.")