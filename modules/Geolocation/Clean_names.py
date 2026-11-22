import re

from _suffixes import ROAD_SUFFIXES as suffixes

def clean_and_split_name(raw_name):
    names = re.split(r'\s*&\s*', raw_name.strip())
    base_names = set()
    found_suffixes = set()

    for name in names:
        words = name.strip().split()
        non_suffix_words = [w for w in words if w.lower() not in suffixes]
        suffix_words = [w for w in words if w.lower() in suffixes]

        if non_suffix_words:
            base_names.add(' '.join(non_suffix_words))
        else:
            base_names.add(name.strip())  # fallback

        found_suffixes.update(suffix_words)

    return base_names, found_suffixes

def remove_lines(input_path):
    """Removes lines without '&' and overwrites the original file."""
    with open(input_path, 'r') as file:
        lines = file.readlines()

    filtered = [line for line in lines if '&' in line]

    with open(input_path, 'w') as out:
        out.writelines(filtered)

def generate_cleaned_names(input_path, output_path):
    """Processes a file and writes cleaned base names and suffixes to output."""
    all_names = set()
    all_suffixes = set()

    with open(input_path, 'r') as file:
        for line in file:
            names, suffixes_found = clean_and_split_name(line)
            all_names.update(names)
            all_suffixes.update(suffixes_found)

    result = all_names.union(all_suffixes)
    result = {name.strip() for name in result if name.strip()}

    with open(output_path, 'w') as out:
        for name in sorted(result):
            out.write(name + '\n')


if __name__ == '__main__':
    input_file = "data/Output/1903_08_25/geoloc/intersection_names.txt"
    cleaned_output = "data/Output/1903_08_25/geoloc/cleaned_intersection_names.txt"

    # Step 1: Remove lines without '&', saving back to the original file
    remove_lines(input_file)

    # Step 2: Generate the cleaned name set to a separate file
    generate_cleaned_names(input_file, cleaned_output)