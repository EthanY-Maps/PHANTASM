"""Shared road-suffix vocabulary used by both Post_OCR and Clean_names."""

ROAD_SUFFIXES = frozenset({
    # Common forms + abbreviations
    "ave", "avenue",
    "blvd", "boulevard", "boulevarde", "bvd",
    "bypass",
    "circuit", "crt",
    "close",
    "court",
    "cove",
    "cr", "crescent",
    "dr", "drive",
    "exit",
    "grove",
    "highway",
    "lane", "ln",
    "offramp", "onramp",
    "parade", "pde",
    "parkway",
    "place", "pl",
    "point",
    "promenade",
    "rd", "road",
    "st", "street",
    "tce", "terrace",
    "walk", "way",

    # Cardinals (used as road-name suffixes: e.g. "Smith Street North")
    "east", "north", "south", "west",

    # Single-word OCR artifacts that aren't true suffixes but appear standalone
    # frequently enough that Post_OCR wants to flag them
    "bay", "bridge", "the", "town", "tunnel",
})
