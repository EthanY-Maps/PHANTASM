from utils import load_json


def extract_intersection_names(input_geojson, output_txt):
    data = load_json(input_geojson)
    
    names = []

    for feature in data.get('features', []):

        name = feature.get('properties', {}).get('intersection_name')
        if name:
            names.append(name)
    

    with open(output_txt, 'w', encoding='utf-8') as f:
        for name in names:
            f.write(name + '\n')

if __name__ == "__main__":

    input_geojson = 'data/Input/Model/Sydney/sydney_intersections.geojson'
    output_txt = "data/Output/1903_08_25/geoloc/intersection_names.txt"
    
    extract_intersection_names(input_geojson, output_txt)