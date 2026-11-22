#!/usr/bin/env python3
"""
Tkinter launcher for the manual raster geolocator.

Features:
- browse for raster, pixel graph, OSM, suburb polygons, output folder, and pairs JSON
- leave OSM blank and provide a suburb name to auto-fetch Sydney-area roads
- launch the existing interactive geolocator UI
- optionally export directly from saved control points
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import requests
except Exception:
    requests = None

from shapely.geometry import box, shape


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

WORKSPACE_ROOT = THIS_DIR.parents[2]
DEFAULT_LOCAL_SYDNEY_OSM = WORKSPACE_ROOT / "Input" / "Model" / "Sydney_Australia_drive_combined.geojson"

from manual_raster_geolocator import (
    ManualGeolocator,
    ReverseSuburbLookup,
    default_outdir_for_tif,
    export_outputs,
    load_osm,
    load_pairs_json,
    load_pixel_graph,
    load_raster,
    load_suburb_geojson,
)


NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
ROAD_HIGHWAY_REGEX = "^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|service)$"


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_").lower()
    return slug or "auto_osm"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def nominatim_headers() -> dict[str, str]:
    return {"User-Agent": "historic-map-geolocator-gui/1.0"}


def find_suburb_bbox(suburb_name: str, timeout: float = 12.0) -> dict[str, Any]:
    if requests is None:
        raise RuntimeError("The requests package is not available in this environment.")

    queries = [
        f"{suburb_name}, Sydney, New South Wales, Australia",
        f"{suburb_name}, New South Wales, Australia",
    ]

    for query in queries:
        response = requests.get(
            NOMINATIM_SEARCH_URL,
            params={
                "q": query,
                "format": "jsonv2",
                "limit": 5,
                "addressdetails": 1,
                "countrycodes": "au",
            },
            headers=nominatim_headers(),
            timeout=timeout,
        )
        response.raise_for_status()
        results = response.json()
        if results:
            return results[0]

    raise RuntimeError(f"Could not find a Sydney-area match for suburb '{suburb_name}'.")


def bbox_from_nominatim(result: dict[str, Any]) -> tuple[float, float, float, float]:
    raw_bbox = result.get("boundingbox")
    if not raw_bbox or len(raw_bbox) != 4:
        raise RuntimeError("Nominatim result did not include a usable bounding box.")
    south = float(raw_bbox[0])
    north = float(raw_bbox[1])
    west = float(raw_bbox[2])
    east = float(raw_bbox[3])
    return south, west, north, east


def pad_bbox(
    south: float,
    west: float,
    north: float,
    east: float,
    lat_pad_frac: float = 0.15,
    lon_pad_frac: float = 0.15,
) -> tuple[float, float, float, float]:
    lat_span = max(north - south, 0.003)
    lon_span = max(east - west, 0.003)
    south -= lat_span * lat_pad_frac
    north += lat_span * lat_pad_frac
    west -= lon_span * lon_pad_frac
    east += lon_span * lon_pad_frac
    return south, west, north, east


def overpass_query_for_bbox(south: float, west: float, north: float, east: float) -> str:
    return f"""
[out:json][timeout:90];
(
  way["highway"~"{ROAD_HIGHWAY_REGEX}"]({south},{west},{north},{east});
);
out geom;
""".strip()


def overpass_ways_to_geojson(payload: dict[str, Any]) -> dict[str, Any]:
    features = []
    for element in payload.get("elements", []):
        if element.get("type") != "way":
            continue
        geom = element.get("geometry") or []
        if len(geom) < 2:
            continue
        coordinates = [[float(pt["lon"]), float(pt["lat"])] for pt in geom]
        props = dict(element.get("tags", {}))
        props["id"] = str(element.get("id", ""))
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coordinates},
                "properties": props,
            }
        )
    if not features:
        raise RuntimeError("Overpass returned no road features for the selected suburb area.")
    return {"type": "FeatureCollection", "features": features}


def crop_geojson_to_bbox(
    source_geojson_path: Path,
    output_path: Path,
    south: float,
    west: float,
    north: float,
    east: float,
) -> Path:
    raw = json.loads(source_geojson_path.read_text(encoding="utf-8"))
    clip_box = box(west, south, east, north)
    kept = []

    for feature in raw.get("features", []):
        geometry = feature.get("geometry")
        if not geometry:
            continue
        try:
            geom = shape(geometry)
        except Exception:
            continue
        if geom.is_empty:
            continue
        if geom.intersects(clip_box):
            kept.append(feature)

    if not kept:
        raise RuntimeError(f"No road features from {source_geojson_path.name} intersect the suburb bounding box.")

    ensure_parent(output_path)
    output_path.write_text(json.dumps({"type": "FeatureCollection", "features": kept}, indent=2), encoding="utf-8")
    return output_path


def fetch_osm_geojson_for_suburb(suburb_name: str, output_path: Path, timeout: float = 45.0) -> Path:
    if requests is None:
        raise RuntimeError("The requests package is not available in this environment.")

    suburb_result = find_suburb_bbox(suburb_name)
    south, west, north, east = pad_bbox(*bbox_from_nominatim(suburb_result))
    query = overpass_query_for_bbox(south, west, north, east)

    response = requests.post(
        OVERPASS_URL,
        data=query.encode("utf-8"),
        headers={
            "Content-Type": "text/plain; charset=utf-8",
            "User-Agent": "historic-map-geolocator-gui/1.0",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()

    geojson = overpass_ways_to_geojson(payload)
    ensure_parent(output_path)
    output_path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")
    return output_path


def resolve_auto_osm_for_suburb(suburb_name: str, output_path: Path) -> Path:
    suburb_result = find_suburb_bbox(suburb_name)
    south, west, north, east = pad_bbox(*bbox_from_nominatim(suburb_result))

    if DEFAULT_LOCAL_SYDNEY_OSM.exists():
        return crop_geojson_to_bbox(
            source_geojson_path=DEFAULT_LOCAL_SYDNEY_OSM,
            output_path=output_path,
            south=south,
            west=west,
            north=north,
            east=east,
        )

    return fetch_osm_geojson_for_suburb(suburb_name, output_path)


class GeolocatorLauncher:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Manual Raster Geolocator")
        self.root.geometry("900x560")
        self.root.minsize(820, 520)

        self.raster_var = tk.StringVar()
        self.graph_var = tk.StringVar()
        self.osm_var = tk.StringVar()
        self.suburb_var = tk.StringVar()
        self.suburb_geojson_var = tk.StringVar()
        self.outdir_var = tk.StringVar()
        self.pairs_var = tk.StringVar()
        self.model_var = tk.StringVar(value="auto")
        self.export_only_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Choose files and click Launch.")

        self.run_button: ttk.Button | None = None
        self.build_ui()

    def build_ui(self) -> None:
        pad = {"padx": 10, "pady": 6}
        frame = ttk.Frame(self.root, padding=14)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frame,
            text="Browse for the raster and graph. Leave OSM blank to auto-download roads from the suburb name.",
            wraplength=820,
        ).grid(row=0, column=0, columnspan=3, sticky="w", **pad)

        self.add_path_row(frame, 1, "Raster image", self.raster_var, self.browse_raster)
        self.add_path_row(frame, 2, "Pixel graph GeoJSON", self.graph_var, self.browse_graph)
        self.add_path_row(frame, 3, "OSM file (optional)", self.osm_var, self.browse_osm)

        ttk.Label(frame, text="Suburb name").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=self.suburb_var, width=70).grid(row=4, column=1, sticky="ew", **pad)
        ttk.Label(frame, text="Used when OSM is blank").grid(row=4, column=2, sticky="w", **pad)

        self.add_path_row(frame, 5, "Suburb polygons (optional)", self.suburb_geojson_var, self.browse_suburb_geojson)
        self.add_path_row(frame, 6, "Output folder", self.outdir_var, self.browse_outdir, choose_dir=True)
        self.add_path_row(frame, 7, "Pairs JSON (optional)", self.pairs_var, self.browse_pairs)

        ttk.Label(frame, text="Transform model").grid(row=8, column=0, sticky="w", **pad)
        ttk.Combobox(
            frame,
            textvariable=self.model_var,
            values=["auto", "affine", "projective", "tps"],
            state="readonly",
            width=20,
        ).grid(row=8, column=1, sticky="w", **pad)

        ttk.Checkbutton(
            frame,
            text="Export only from saved control points",
            variable=self.export_only_var,
        ).grid(row=9, column=1, sticky="w", **pad)

        buttons = ttk.Frame(frame)
        buttons.grid(row=10, column=0, columnspan=3, sticky="ew", padx=10, pady=14)
        self.run_button = ttk.Button(buttons, text="Launch", command=self.launch)
        self.run_button.pack(side=tk.LEFT)
        ttk.Button(buttons, text="Quit", command=self.root.destroy).pack(side=tk.LEFT, padx=8)

        ttk.Label(
            frame,
            textvariable=self.status_var,
            foreground="#334155",
            wraplength=820,
        ).grid(row=11, column=0, columnspan=3, sticky="w", **pad)

        frame.columnconfigure(1, weight=1)

    def add_path_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        browse_command: Any,
        *,
        choose_dir: bool = False,
    ) -> None:
        del choose_dir
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=10, pady=6)
        ttk.Entry(parent, textvariable=variable, width=70).grid(row=row, column=1, sticky="ew", padx=10, pady=6)
        ttk.Button(parent, text="Browse", command=browse_command).grid(row=row, column=2, sticky="ew", padx=10, pady=6)

    def browse_raster(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose raster image",
            filetypes=[("Images", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")],
        )
        if not path:
            return
        self.raster_var.set(path)
        if not self.outdir_var.get().strip():
            self.outdir_var.set(str(default_outdir_for_tif(Path(path))))

    def browse_graph(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose pixel graph GeoJSON",
            filetypes=[("GeoJSON", "*.geojson *.json"), ("All files", "*.*")],
        )
        if path:
            self.graph_var.set(path)

    def browse_osm(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose OSM file",
            filetypes=[("Graph and GeoJSON", "*.graphml *.geojson *.json"), ("All files", "*.*")],
        )
        if path:
            self.osm_var.set(path)

    def browse_suburb_geojson(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose suburb polygons GeoJSON",
            filetypes=[("GeoJSON", "*.geojson *.json"), ("All files", "*.*")],
        )
        if path:
            self.suburb_geojson_var.set(path)

    def browse_outdir(self) -> None:
        path = filedialog.askdirectory(title="Choose output folder")
        if path:
            self.outdir_var.set(path)

    def browse_pairs(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose pairs JSON",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.pairs_var.set(path)

    def set_busy(self, busy: bool) -> None:
        self.root.config(cursor="watch" if busy else "")
        if self.run_button is not None:
            self.run_button.state(["disabled"] if busy else ["!disabled"])
        self.root.update_idletasks()

    def resolve_paths(self) -> tuple[Path, Path, Path, Path, Path | None, str]:
        raster_path = Path(self.raster_var.get().strip()).expanduser().resolve()
        graph_path = Path(self.graph_var.get().strip()).expanduser().resolve()
        if not raster_path.exists():
            raise FileNotFoundError("Raster image does not exist.")
        if not graph_path.exists():
            raise FileNotFoundError("Pixel graph GeoJSON does not exist.")

        outdir_text = self.outdir_var.get().strip()
        outdir = Path(outdir_text).expanduser().resolve() if outdir_text else default_outdir_for_tif(raster_path)
        pairs_text = self.pairs_var.get().strip()
        pairs_path = Path(pairs_text).expanduser().resolve() if pairs_text else (outdir / "control_points.json")

        suburb_geojson_text = self.suburb_geojson_var.get().strip()
        suburb_geojson_path = Path(suburb_geojson_text).expanduser().resolve() if suburb_geojson_text else None
        suburb_name = self.suburb_var.get().strip()

        osm_text = self.osm_var.get().strip()
        if osm_text:
            osm_path = Path(osm_text).expanduser().resolve()
            if not osm_path.exists():
                raise FileNotFoundError("OSM input file does not exist.")
        else:
            if not suburb_name:
                raise ValueError("Leave OSM blank only when a suburb name is provided.")
            osm_path = outdir / "auto_osm" / f"{slugify(suburb_name)}_roads.geojson"
            if DEFAULT_LOCAL_SYDNEY_OSM.exists():
                self.status_var.set(
                    f"Resolving suburb '{suburb_name}' and cropping local Sydney OSM..."
                )
            else:
                self.status_var.set(
                    f"Resolving suburb '{suburb_name}' and downloading OSM roads..."
                )
            self.root.update_idletasks()
            if not osm_path.exists():
                resolve_auto_osm_for_suburb(suburb_name, osm_path)

        return raster_path, graph_path, osm_path, outdir, suburb_geojson_path, str(pairs_path)

    def launch(self) -> None:
        self.set_busy(True)
        try:
            raster_path, graph_path, osm_path, outdir, suburb_geojson_path, pairs_path_text = self.resolve_paths()
            pairs_path = Path(pairs_path_text)

            self.status_var.set("Loading raster, graph, and OSM...")
            self.root.update_idletasks()

            raster = load_raster(raster_path, preview_max_side=2200)
            pixel_graph = load_pixel_graph(graph_path, raster=raster)
            osm = load_osm(osm_path)
            suburbs = load_suburb_geojson(suburb_geojson_path) if suburb_geojson_path else None
            reverse_lookup = ReverseSuburbLookup(
                enabled=True,
                url="https://nominatim.openstreetmap.org/reverse",
                email=None,
                timeout=1.5,
            )

            if self.export_only_var.get():
                if not pairs_path.exists():
                    raise FileNotFoundError("Export-only mode requires an existing pairs JSON file.")
                self.status_var.set("Exporting from saved control points...")
                self.root.update_idletasks()
                report = export_outputs(
                    raster=raster,
                    pixel_graph=pixel_graph,
                    osm=osm,
                    pairs=load_pairs_json(pairs_path),
                    requested_model=self.model_var.get().strip() or "auto",
                    output_dir=outdir,
                    meters_per_pixel=None,
                    max_output_side=2500,
                    padding_frac=0.05,
                )
                messagebox.showinfo(
                    "Export complete",
                    f"Exported with {report['transform_model']}\n\n{report['outputs']['warped_tif']}",
                )
                self.status_var.set("Export complete.")
                return

            self.status_var.set("Launching interactive geolocator...")
            self.root.update_idletasks()
            self.root.destroy()
            app = ManualGeolocator(
                raster=raster,
                pixel_graph=pixel_graph,
                osm=osm,
                suburbs=suburbs,
                reverse_suburb_lookup=reverse_lookup,
                pairs_path=pairs_path,
                output_dir=outdir,
                args=type(
                    "GuiArgs",
                    (),
                    {
                        "snap_radius_frac": 0.02,
                        "model": self.model_var.get().strip() or "auto",
                        "meters_per_pixel": None,
                        "max_output_side": 2500,
                        "padding_frac": 0.05,
                    },
                )(),
            )
            app.run()
        except Exception as exc:
            self.status_var.set(str(exc))
            messagebox.showerror("Launch failed", str(exc))
        finally:
            try:
                self.set_busy(False)
            except Exception:
                pass

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    launcher = GeolocatorLauncher()
    launcher.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
