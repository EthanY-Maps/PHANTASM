import re

from rapidfuzz import process as rf_process, fuzz
from tqdm import tqdm

from utils import load_json, save_json
from _suffixes import ROAD_SUFFIXES

# --- TOGGLES & CONFIG ---
DEBUG_MODE = False
DEFAULT_FUZZY_THRESHOLD = 75   # default if caller doesn't pass `threshold`

# Strip punctuation that TrOCR sometimes hallucinates around real words. Keep
# alphanumerics, spaces, apostrophes (e.g. "St. John's"), and hyphens (e.g.
# "Smith-Jones Rd"). The OCR JSON preserves the raw model output; this clean
# only happens at the matching boundary.
_NAME_NOISE_RE = re.compile(r"[^a-z0-9' -]+")
_MULTI_SPACE_RE = re.compile(r"\s+")

def clean_text_for_matching(text):
    if not isinstance(text, str): return ""
    lowered = text.strip().lower()
    cleaned = _NAME_NOISE_RE.sub(" ", lowered)
    return _MULTI_SPACE_RE.sub(" ", cleaned).strip()

def flip_y_recursive(coords):
    """Recursively flips the Y coordinate for Points, Lines, and Polygons."""
    if not isinstance(coords, list):
        return coords
    if len(coords) >= 2 and isinstance(coords[0], (int, float)):
        return [coords[0], -coords[1]]
    return [flip_y_recursive(c) for c in coords]

def post_ocr(input_json, input_text_dic, output_path, text_input="predicted_text",
             threshold: int = DEFAULT_FUZZY_THRESHOLD):
    # 1. Load Dictionary
    with open(input_text_dic, "r", encoding="utf-8") as f:
        cleaned_names = [line.strip() for line in f if line.strip()]
    
    cleaned_names_lower = [n.lower() for n in cleaned_names]
    name_lookup = {n.lower(): n for n in cleaned_names}

    # 2. Load GeoJSON
    geojson_data = load_json(input_json)

    match_cache = {}
    stats = {"corrected": 0, "verified": 0, "no_match": 0, "suffix_found": 0}

    def get_best_match(ocr_value):
        if not ocr_value: return "", 0
        
        candidates = ocr_value if isinstance(ocr_value, list) else [ocr_value]
        best_overall_match = ""
        best_overall_score = 0

        for text in candidates:
            search_term = clean_text_for_matching(text)
            
            # CONSISTENCY UPDATE: Suffixes are no longer skipped here.
            # They will now go through the fuzzy matching process.
            if not search_term:
                continue

            if search_term in match_cache:
                match, score = match_cache[search_term]
            else:
                results = rf_process.extract(
                    search_term, 
                    cleaned_names_lower, 
                    scorer=fuzz.token_sort_ratio,
                    limit=3
                )
                
                if results:
                    best_val, best_score, _ = results[0]
                    match = name_lookup[best_val]
                    score = best_score
                else:
                    match, score = "", 0
                
                match_cache[search_term] = (match, score)

            if score > best_overall_score:
                best_overall_score = score
                best_overall_match = match

        return (best_overall_match, best_overall_score) if best_overall_score >= threshold else ("", 0)

    # 3. Process Features
    print(f"Starting processing (DEBUG: {DEBUG_MODE})...")
    features = geojson_data.get("features", [])
    
    for feature in tqdm(features, desc="Processing Features", disable=DEBUG_MODE):
        props = feature.get("properties", {})
        original_text = props.get(text_input, "")
        original_clean = clean_text_for_matching(original_text)
        
        # Get the match (suffixes will now have a score and a match if in dictionary)
        match, score = get_best_match(original_text)
        match_clean = clean_text_for_matching(match)
        
        props[f"corrected_{text_input}"] = match
        props[f"score_{text_input}"] = score

        # --- UPDATED FLAGGING LOGIC ---
        
        # Priority 1: Flag if either the OCR or the Match is in the suffix list
        if original_clean in ROAD_SUFFIXES or match_clean in ROAD_SUFFIXES:
            props["match_status"] = "suffix_found"
            props["match_found"] = (score >= threshold)
            # Even if it's a suffix, we track if the text changed
            props["is_modified"] = (match_clean != original_clean and match != "")
            stats["suffix_found"] += 1
            
        # Priority 2: Standard Matching Logic
        elif score >= threshold:
            props["match_found"] = True
            if match_clean == original_clean:
                props["match_status"] = "verified"
                props["is_modified"] = False
                stats["verified"] += 1
            else:
                props["match_status"] = "corrected"
                props["is_modified"] = True
                stats["corrected"] += 1
        else:
            props["match_found"] = False
            props["match_status"] = "no_high_confidence_match"
            props["is_modified"] = False
            stats["no_match"] += 1
        # --- END FLAGGING LOGIC ---

        if "geometry" in feature and feature["geometry"]:
            old_coords = feature["geometry"]["coordinates"]
            feature["geometry"]["coordinates"] = flip_y_recursive(old_coords)

    # 4. Save Output
    save_json(output_path, geojson_data, indent=2)
    
    # Summary Report
    print("-" * 30)
    print("PROCESSING SUMMARY")
    print("-" * 30)
    print(f"Total Features:  {len(features)}")
    print(f"✅ Verified:     {stats['verified']}")
    print(f"🛠️  Corrected:    {stats['corrected']}")
    print(f"🚧 Suffix Found: {stats['suffix_found']} (Processed & Tagged)")
    print(f"❌ No Match:     {stats['no_match']}")
    print("-" * 30)
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    input_json = "data/Output/1903-20-01-2026/Text_detection/detection_results.json"
    input_text_dic = "data/Output/1903-20-01-2026/GeoLoc/intersection_names.txt"
    output_path = "data/Output/1903-20-01-2026/GeoLoc/OCR.geojson"
    
    post_ocr(input_json, input_text_dic, output_path, "predicted_text")