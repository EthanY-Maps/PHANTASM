#!/usr/bin/env python3
"""
Manual TIFF georeferencing tool for historic maps.

This script lets you:
1. load a historic raster (.tif/.tiff),
2. overlay its pixel-space road graph (.geojson),
3. load an OSM street graph (.graphml or .geojson),
4. click matching control points between the raster and OSM,
5. fit an affine, projective, or thin-plate spline transform, and
6. export a warped raster, a geolocated graph, and an OSM overlay preview.

The output raster is rectified into EPSG:3857 (Web Mercator), which makes it
easy to preview on top of OSM tiles and to open in GIS software. Because this
script avoids heavy geo dependencies, georeferencing is written as a world file
(.tfw/.pgw) plus a .prj file rather than embedded GeoTIFF tags.

Example:
    python3 manual_raster_geolocator.py \
        --tif data/Input/1877/nla.obj-231444908.tif \
        --graph data/Output/1877_11-02-26/Graphing/roads.geojson \
        --osm data/Input/OSM_Graphs/city_n33.9040_151.1500_n33.8380_151.2250.graphml \
        --outdir data/Output/1877_manual_geoloc

Interactive controls:
    left click on raster   -> pick raster control point
    left click on OSM      -> complete the pair, or inspect suburb when no raster point is pending
    s                      -> save control points
    d                      -> undo most recent pair
    e                      -> export warped outputs
    escape                 -> clear current selection
    q                      -> save and quit
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

fallback_cache = Path(tempfile.gettempdir()) / "manual_raster_geolocator_cache"
(fallback_cache / "mpl").mkdir(parents=True, exist_ok=True)
(fallback_cache / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(fallback_cache / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(fallback_cache / "xdg"))

if "--export-only" in sys.argv:
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from shapely.geometry import Point, shape
from shapely.wkt import loads as load_wkt
from skimage.transform import (
    AffineTransform,
    ProjectiveTransform,
    ThinPlateSplineTransform,
    warp,
)

try:
    import requests
except Exception:
    requests = None


WEB_MERCATOR_WKT = """PROJCS["WGS_1984_Web_Mercator_Auxiliary_Sphere",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Mercator_Auxiliary_Sphere"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",0.0],PARAMETER["Standard_Parallel_1",0.0],PARAMETER["Auxiliary_Sphere_Type",0.0],UNIT["Meter",1.0]]"""


@dataclass
class RasterBundle:
    path: Path
    image_uint8: np.ndarray
    image_preview: np.ndarray
    width: int
    height: int
    preview_width: int
    preview_height: int
    scale_x: float
    scale_y: float


@dataclass
class PixelGraphBundle:
    path: Path
    raw_geojson: dict[str, Any]
    lines_full: list[np.ndarray]
    lines_preview: list[np.ndarray]
    snap_points_full: np.ndarray
    snap_points_preview: np.ndarray
    flip_y: bool


@dataclass
class OSMBundle:
    path: Path
    lines_mercator: list[np.ndarray]
    snap_points_mercator: np.ndarray
    bounds_mercator: tuple[float, float, float, float]


@dataclass
class TransformBundle:
    name: str
    forward_xy: Callable[[np.ndarray], np.ndarray]
    inverse_xy: Callable[[np.ndarray], np.ndarray]
    residuals_m: np.ndarray


@dataclass
class SuburbFeature:
    name: str
    geometry_lonlat: Any
    outlines_mercator: list[np.ndarray]
    centroid_lonlat: tuple[float, float]
    centroid_mercator: tuple[float, float]


@dataclass
class SuburbBundle:
    path: Path
    features: list[SuburbFeature]


@dataclass
class SuburbLookupResult:
    name: str
    source: str
    lonlat: tuple[float, float]
    mercator_xy: tuple[float, float]
    feature: SuburbFeature | None = None
    display_name: str | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual TIFF georeferencing onto OSM.")
    parser.add_argument("--tif", required=True, help="Historic raster (.tif/.tiff)")
    parser.add_argument("--graph", required=True, help="Pixel-space graph GeoJSON for the raster")
    parser.add_argument("--osm", required=True, help="OSM graph as GraphML or GeoJSON")
    parser.add_argument(
        "--suburb-geojson",
        help="Optional suburb boundary GeoJSON in WGS84 lon/lat for offline suburb lookup/highlighting.",
    )
    parser.add_argument("--outdir", help="Output directory. Defaults to <tif stem>_manual_geoloc")
    parser.add_argument(
        "--pairs",
        help="Optional control-point JSON file. Defaults to <outdir>/control_points.json",
    )
    parser.add_argument(
        "--model",
        choices=("auto", "affine", "projective", "tps"),
        default="auto",
        help="Transform model. 'auto' uses affine for 3 points, projective for 4-5, TPS for 6+.",
    )
    parser.add_argument(
        "--preview-max-side",
        type=int,
        default=2200,
        help="Maximum display size for the left raster preview.",
    )
    parser.add_argument(
        "--max-output-side",
        type=int,
        default=2500,
        help="Clamp the largest exported raster dimension to this many pixels.",
    )
    parser.add_argument(
        "--meters-per-pixel",
        type=float,
        default=None,
        help="Override export resolution in EPSG:3857 meters per pixel.",
    )
    parser.add_argument(
        "--snap-radius-frac",
        type=float,
        default=0.02,
        help="Snap radius as a fraction of the visible axis range.",
    )
    parser.add_argument(
        "--padding-frac",
        type=float,
        default=0.05,
        help="Padding around the warped raster extent, as a fraction of the larger span.",
    )
    parser.add_argument(
        "--disable-reverse-suburb-lookup",
        action="store_true",
        help="Disable online suburb lookup via Nominatim when no local suburb polygons are supplied.",
    )
    parser.add_argument(
        "--nominatim-url",
        default="https://nominatim.openstreetmap.org/reverse",
        help="Reverse geocoding endpoint used for suburb lookup.",
    )
    parser.add_argument(
        "--nominatim-email",
        default=None,
        help="Optional email to include with Nominatim requests.",
    )
    parser.add_argument(
        "--nominatim-timeout",
        type=float,
        default=1.5,
        help="Timeout in seconds for online suburb lookups.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip the interactive UI and export immediately from --pairs.",
    )
    return parser.parse_args(argv)


def lonlat_to_web_mercator(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lat = np.clip(lat, -85.05112878, 85.05112878)
    radius = 6_378_137.0
    x = np.radians(lon) * radius
    y = radius * np.log(np.tan(np.pi / 4.0 + np.radians(lat) / 2.0))
    return x, y


def web_mercator_to_lonlat(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    radius = 6_378_137.0
    lon = np.degrees(x / radius)
    lat = np.degrees(2.0 * np.arctan(np.exp(y / radius)) - np.pi / 2.0)
    return lon, lat


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        scaled = image.astype(np.float32) / max(float(info.max), 1.0)
        return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)
    image = np.asarray(image, dtype=np.float32)
    lo = float(np.nanmin(image))
    hi = float(np.nanmax(image))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image - lo) / (hi - lo)
    return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)


def ensure_rgb_or_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] in (3, 4):
        return image
    raise ValueError(f"Unsupported raster shape: {image.shape}")


def resize_image_uint8(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image.copy()
    scale = max_side / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    pil_image = Image.fromarray(image)
    resized = pil_image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    return np.asarray(resized)


def load_raster(path: Path, preview_max_side: int) -> RasterBundle:
    image = Image.open(path)
    image_array = np.asarray(image)
    image_uint8 = ensure_rgb_or_gray(ensure_uint8(image_array))
    preview = resize_image_uint8(image_uint8, preview_max_side)
    height, width = image_uint8.shape[:2]
    preview_height, preview_width = preview.shape[:2]
    return RasterBundle(
        path=path,
        image_uint8=image_uint8,
        image_preview=preview,
        width=width,
        height=height,
        preview_width=preview_width,
        preview_height=preview_height,
        scale_x=width / float(preview_width),
        scale_y=height / float(preview_height),
    )


def iter_geometry_coords(geometry: dict[str, Any]) -> Iterable[np.ndarray]:
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")
    if coords is None:
        return
    if geom_type == "Point":
        yield np.asarray([coords], dtype=float)
    elif geom_type in {"LineString", "MultiPoint"}:
        yield np.asarray(coords, dtype=float)
    elif geom_type in {"Polygon", "MultiLineString"}:
        for part in coords:
            yield np.asarray(part, dtype=float)
    elif geom_type == "MultiPolygon":
        for polygon in coords:
            for ring in polygon:
                yield np.asarray(ring, dtype=float)


def pixel_graph_flip_needed(raw_geojson: dict[str, Any]) -> bool:
    xs: list[float] = []
    ys: list[float] = []
    for feature in raw_geojson.get("features", []):
        geometry = feature.get("geometry") or {}
        for coord_block in iter_geometry_coords(geometry):
            if coord_block.size == 0:
                continue
            xs.extend(coord_block[:, 0].tolist())
            ys.extend(coord_block[:, 1].tolist())
    if not xs or not ys:
        return False
    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    span_x = float(xs_arr.max() - xs_arr.min())
    span_y = float(ys_arr.max() - ys_arr.min())
    pixel_like = max(np.max(np.abs(xs_arr)), np.max(np.abs(ys_arr))) > 100.0 or max(span_x, span_y) > 100.0
    return bool(pixel_like and np.median(ys_arr) < 0.0)


def normalize_pixel_coords(coords: np.ndarray, flip_y: bool) -> np.ndarray:
    coords = np.asarray(coords, dtype=float).copy()
    if coords.ndim == 1:
        coords = coords.reshape(1, 2)
    if flip_y:
        coords[:, 1] *= -1.0
    return coords


def load_pixel_graph(path: Path, raster: RasterBundle) -> PixelGraphBundle:
    raw_geojson = json.loads(path.read_text(encoding="utf-8"))
    flip_y = pixel_graph_flip_needed(raw_geojson)

    lines_full: list[np.ndarray] = []
    snap_blocks_full: list[np.ndarray] = []

    for feature in raw_geojson.get("features", []):
        geometry = feature.get("geometry") or {}
        geom_type = geometry.get("type")
        if geom_type is None:
            continue

        for coord_block in iter_geometry_coords(geometry):
            if coord_block.size == 0:
                continue
            normalized = normalize_pixel_coords(coord_block, flip_y)
            if normalized.shape[1] < 2:
                continue
            if geom_type in {"LineString", "MultiLineString"} and normalized.shape[0] >= 2:
                lines_full.append(normalized[:, :2])
                snap_blocks_full.append(normalized[:, :2])
            elif geom_type in {"Point", "MultiPoint"}:
                snap_blocks_full.append(normalized[:, :2])

    if not snap_blocks_full:
        raise ValueError(f"No usable coordinates found in pixel graph: {path}")

    snap_points_full = np.vstack(snap_blocks_full)
    snap_points_preview = snap_points_full.copy()
    snap_points_preview[:, 0] /= raster.scale_x
    snap_points_preview[:, 1] /= raster.scale_y

    lines_preview = []
    for line in lines_full:
        preview_line = line.copy()
        preview_line[:, 0] /= raster.scale_x
        preview_line[:, 1] /= raster.scale_y
        lines_preview.append(preview_line)

    return PixelGraphBundle(
        path=path,
        raw_geojson=raw_geojson,
        lines_full=lines_full,
        lines_preview=lines_preview,
        snap_points_full=snap_points_full,
        snap_points_preview=snap_points_preview,
        flip_y=flip_y,
    )


def append_osm_line(lines: list[np.ndarray], snap_blocks: list[np.ndarray], lonlat: np.ndarray) -> None:
    lonlat = np.asarray(lonlat, dtype=float)
    if lonlat.ndim != 2 or lonlat.shape[0] < 2:
        return
    x, y = lonlat_to_web_mercator(lonlat[:, 0], lonlat[:, 1])
    line = np.column_stack([x, y])
    lines.append(line)
    snap_blocks.append(line)


def load_osm_geojson(path: Path) -> OSMBundle:
    raw = json.loads(path.read_text(encoding="utf-8"))
    lines: list[np.ndarray] = []
    snap_blocks: list[np.ndarray] = []

    for feature in raw.get("features", []):
        geometry = feature.get("geometry") or {}
        geom_type = geometry.get("type")
        if geom_type in {"LineString", "MultiLineString", "Point", "MultiPoint"}:
            for coord_block in iter_geometry_coords(geometry):
                if coord_block.size == 0:
                    continue
                if geom_type in {"LineString", "MultiLineString"}:
                    append_osm_line(lines, snap_blocks, coord_block[:, :2])
                elif geom_type in {"Point", "MultiPoint"}:
                    lonlat = np.asarray(coord_block[:, :2], dtype=float)
                    x, y = lonlat_to_web_mercator(lonlat[:, 0], lonlat[:, 1])
                    snap_blocks.append(np.column_stack([x, y]))

    if not snap_blocks:
        raise ValueError(f"No usable OSM geometry found in GeoJSON: {path}")

    snap_points = np.vstack(snap_blocks)
    min_x = float(np.min(snap_points[:, 0]))
    min_y = float(np.min(snap_points[:, 1]))
    max_x = float(np.max(snap_points[:, 0]))
    max_y = float(np.max(snap_points[:, 1]))
    return OSMBundle(
        path=path,
        lines_mercator=lines,
        snap_points_mercator=snap_points,
        bounds_mercator=(min_x, min_y, max_x, max_y),
    )


def edge_lonlat_from_graphml(
    graph: nx.Graph,
    u: Any,
    v: Any,
    data: dict[str, Any],
) -> np.ndarray | None:
    geometry_attr = data.get("geometry")
    if geometry_attr:
        try:
            geom = load_wkt(str(geometry_attr))
            if geom.geom_type == "LineString":
                return np.asarray(geom.coords, dtype=float)
            if geom.geom_type == "MultiLineString":
                longest = max(geom.geoms, key=lambda g: g.length)
                return np.asarray(longest.coords, dtype=float)
        except Exception:
            pass

    node_u = graph.nodes[u]
    node_v = graph.nodes[v]
    try:
        lon_u = float(node_u["x"])
        lat_u = float(node_u["y"])
        lon_v = float(node_v["x"])
        lat_v = float(node_v["y"])
    except Exception:
        return None
    return np.asarray([[lon_u, lat_u], [lon_v, lat_v]], dtype=float)


def load_osm_graphml(path: Path) -> OSMBundle:
    graph = nx.read_graphml(path)
    lines: list[np.ndarray] = []
    snap_blocks: list[np.ndarray] = []

    for _, node_data in graph.nodes(data=True):
        try:
            lon = float(node_data["x"])
            lat = float(node_data["y"])
        except Exception:
            continue
        x, y = lonlat_to_web_mercator(np.asarray([lon]), np.asarray([lat]))
        snap_blocks.append(np.asarray([[float(x[0]), float(y[0])]], dtype=float))

    if graph.is_multigraph():
        edge_iter = graph.edges(keys=False, data=True)
    else:
        edge_iter = graph.edges(data=True)

    for edge in edge_iter:
        if len(edge) == 4:
            u, v, _, data = edge
        else:
            u, v, data = edge
        lonlat = edge_lonlat_from_graphml(graph, u, v, data)
        if lonlat is None:
            continue
        append_osm_line(lines, snap_blocks, lonlat[:, :2])

    if not snap_blocks:
        raise ValueError(f"No usable nodes/edges found in OSM graph: {path}")

    snap_points = np.vstack(snap_blocks)
    min_x = float(np.min(snap_points[:, 0]))
    min_y = float(np.min(snap_points[:, 1]))
    max_x = float(np.max(snap_points[:, 0]))
    max_y = float(np.max(snap_points[:, 1]))
    return OSMBundle(
        path=path,
        lines_mercator=lines,
        snap_points_mercator=snap_points,
        bounds_mercator=(min_x, min_y, max_x, max_y),
    )


def load_osm(path: Path) -> OSMBundle:
    suffix = path.suffix.lower()
    if suffix == ".graphml":
        return load_osm_graphml(path)
    if suffix in {".geojson", ".json"}:
        return load_osm_geojson(path)
    raise ValueError(f"Unsupported OSM input format: {path}")


SUBURB_NAME_KEYS = (
    "suburb",
    "locality",
    "name",
    "official_name",
    "SSC_NAME_2016",
    "SSC_NAME_2021",
    "SSC_NAME21",
    "SSC_NAME16",
    "LOCALITY_NAME",
    "locality_name",
    "neighbourhood",
)


def suburb_name_from_properties(properties: dict[str, Any]) -> str | None:
    for key in SUBURB_NAME_KEYS:
        value = properties.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def polygon_outlines_to_mercator(geom: Any) -> list[np.ndarray]:
    if geom.is_empty:
        return []
    polygons = []
    if geom.geom_type == "Polygon":
        polygons = [geom]
    elif geom.geom_type == "MultiPolygon":
        polygons = list(geom.geoms)
    else:
        return []

    outlines: list[np.ndarray] = []
    for polygon in polygons:
        coords = np.asarray(polygon.exterior.coords, dtype=float)
        if coords.ndim != 2 or coords.shape[0] < 2:
            continue
        x, y = lonlat_to_web_mercator(coords[:, 0], coords[:, 1])
        outlines.append(np.column_stack([x, y]))
    return outlines


def load_suburb_geojson(path: Path) -> SuburbBundle:
    raw = json.loads(path.read_text(encoding="utf-8"))
    features: list[SuburbFeature] = []

    for idx, feature in enumerate(raw.get("features", []), start=1):
        geometry = feature.get("geometry") or {}
        properties = feature.get("properties", {}) or {}
        try:
            geom = shape(geometry)
        except Exception:
            continue
        if geom.is_empty or geom.geom_type not in {"Polygon", "MultiPolygon"}:
            continue

        name = suburb_name_from_properties(properties) or f"Suburb {idx}"
        rep = geom.representative_point()
        lon = float(rep.x)
        lat = float(rep.y)
        mx, my = lonlat_to_web_mercator(np.asarray([lon]), np.asarray([lat]))
        features.append(
            SuburbFeature(
                name=name,
                geometry_lonlat=geom,
                outlines_mercator=polygon_outlines_to_mercator(geom),
                centroid_lonlat=(lon, lat),
                centroid_mercator=(float(mx[0]), float(my[0])),
            )
        )

    if not features:
        raise ValueError(f"No polygon suburb features found in {path}")
    return SuburbBundle(path=path, features=features)


class ReverseSuburbLookup:
    def __init__(
        self,
        *,
        enabled: bool,
        url: str,
        email: str | None,
        timeout: float,
    ) -> None:
        self.enabled = bool(enabled and requests is not None)
        self.url = url
        self.email = email
        self.timeout = float(timeout)
        self.cache: dict[tuple[float, float], SuburbLookupResult | None] = {}

    def lookup(self, lon: float, lat: float) -> SuburbLookupResult | None:
        if not self.enabled:
            return None

        cache_key = (round(float(lon), 5), round(float(lat), 5))
        if cache_key in self.cache:
            return self.cache[cache_key]

        params = {
            "lat": float(lat),
            "lon": float(lon),
            "format": "jsonv2",
            "addressdetails": 1,
            "zoom": 14,
        }
        if self.email:
            params["email"] = self.email

        headers = {"User-Agent": "historic-map-geolocator/1.0"}

        try:
            response = requests.get(self.url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            self.cache[cache_key] = None
            return None

        address = payload.get("address", {}) if isinstance(payload, dict) else {}
        name = None
        for key in ("suburb", "city_district", "neighbourhood", "borough", "quarter", "town", "village", "hamlet", "city", "municipality", "county"):
            value = address.get(key)
            if value:
                name = str(value)
                break
        if not name:
            display_name = str(payload.get("display_name", "")).strip()
            name = display_name.split(",")[0].strip() if display_name else "Unknown suburb"

        mx, my = lonlat_to_web_mercator(np.asarray([lon]), np.asarray([lat]))
        result = SuburbLookupResult(
            name=name,
            source="Nominatim",
            lonlat=(float(lon), float(lat)),
            mercator_xy=(float(mx[0]), float(my[0])),
            feature=None,
            display_name=str(payload.get("display_name", "")).strip() or None,
        )
        self.cache[cache_key] = result
        return result


def estimate_transform_model(
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
    requested: str,
) -> TransformBundle:
    src_xy = np.asarray(src_xy, dtype=float)
    dst_xy = np.asarray(dst_xy, dtype=float)
    if src_xy.shape != dst_xy.shape or src_xy.ndim != 2 or src_xy.shape[1] != 2:
        raise ValueError("Control points must be Nx2 arrays.")

    n_points = src_xy.shape[0]
    if n_points < 3:
        raise ValueError("At least 3 control points are required.")

    if requested == "auto":
        if n_points == 3:
            model_name = "affine"
        elif n_points <= 5:
            model_name = "projective"
        else:
            model_name = "tps"
    else:
        model_name = requested

    if model_name == "affine":
        forward = AffineTransform()
        inverse = AffineTransform()
    elif model_name == "projective":
        forward = ProjectiveTransform()
        inverse = ProjectiveTransform()
    elif model_name == "tps":
        forward = ThinPlateSplineTransform()
        inverse = ThinPlateSplineTransform()
    else:
        raise ValueError(f"Unsupported transform model: {model_name}")

    ok_forward = forward.estimate(src_xy, dst_xy)
    ok_inverse = inverse.estimate(dst_xy, src_xy)
    if not ok_forward or not ok_inverse:
        raise RuntimeError(f"Failed to estimate {model_name} transform.")

    predicted = np.asarray(forward(src_xy), dtype=float)
    residuals = np.linalg.norm(predicted - dst_xy, axis=1)
    return TransformBundle(
        name=model_name,
        forward_xy=lambda pts: np.asarray(forward(np.asarray(pts, dtype=float)), dtype=float),
        inverse_xy=lambda pts: np.asarray(inverse(np.asarray(pts, dtype=float)), dtype=float),
        residuals_m=residuals,
    )


def sample_image_border(width: int, height: int, samples_per_side: int = 128) -> np.ndarray:
    xs = np.linspace(0.0, max(width - 1, 1), samples_per_side)
    ys = np.linspace(0.0, max(height - 1, 1), samples_per_side)
    top = np.column_stack([xs, np.zeros_like(xs)])
    right = np.column_stack([np.full_like(ys, max(width - 1, 0)), ys])
    bottom = np.column_stack([xs[::-1], np.full_like(xs, max(height - 1, 0))])
    left = np.column_stack([np.zeros_like(ys), ys[::-1]])
    return np.vstack([top, right, bottom, left])


def subsample_points(points: np.ndarray, max_points: int = 2000) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points).astype(int)
    return points[idx]


def estimate_output_canvas(
    transform_bundle: TransformBundle,
    raster: RasterBundle,
    pixel_graph: PixelGraphBundle,
    meters_per_pixel: float | None,
    max_output_side: int,
    padding_frac: float,
) -> tuple[float, float, float, float, float, int, int]:
    border_samples = sample_image_border(raster.width, raster.height, samples_per_side=128)
    graph_samples = subsample_points(pixel_graph.snap_points_full, max_points=2000)
    extent_samples = np.vstack([border_samples, graph_samples])

    warped = transform_bundle.forward_xy(extent_samples)
    warped = warped[np.isfinite(warped).all(axis=1)]
    if len(warped) == 0:
        raise RuntimeError("Could not determine warped raster extent.")

    min_x = float(np.min(warped[:, 0]))
    min_y = float(np.min(warped[:, 1]))
    max_x = float(np.max(warped[:, 0]))
    max_y = float(np.max(warped[:, 1]))

    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    pad = padding_frac * max(span_x, span_y)
    min_x -= pad
    min_y -= pad
    max_x += pad
    max_y += pad

    if meters_per_pixel is None:
        meters_per_pixel = max((max_x - min_x) / float(raster.width), (max_y - min_y) / float(raster.height))
    meters_per_pixel = max(float(meters_per_pixel), 0.01)

    width_out = int(math.ceil((max_x - min_x) / meters_per_pixel))
    height_out = int(math.ceil((max_y - min_y) / meters_per_pixel))

    longest_side = max(width_out, height_out, 1)
    if longest_side > max_output_side:
        scale = longest_side / float(max_output_side)
        meters_per_pixel *= scale
        width_out = int(math.ceil((max_x - min_x) / meters_per_pixel))
        height_out = int(math.ceil((max_y - min_y) / meters_per_pixel))

    width_out = max(width_out, 1)
    height_out = max(height_out, 1)
    return min_x, min_y, max_x, max_y, meters_per_pixel, width_out, height_out


def make_inverse_map(
    transform_bundle: TransformBundle,
    min_x: float,
    max_y: float,
    meters_per_pixel: float,
) -> Callable[[np.ndarray], np.ndarray]:
    def inverse_map(coords_rc: np.ndarray) -> np.ndarray:
        coords_rc = np.asarray(coords_rc, dtype=np.float32)
        rows = coords_rc[:, 0]
        cols = coords_rc[:, 1]
        world_x = min_x + cols * meters_per_pixel
        world_y = max_y - rows * meters_per_pixel
        src_xy = transform_bundle.inverse_xy(np.column_stack([world_x, world_y]))
        src_cols = src_xy[:, 0]
        src_rows = src_xy[:, 1]
        return np.column_stack([src_rows, src_cols]).astype(np.float32)

    return inverse_map


def warp_raster_to_canvas(
    raster: RasterBundle,
    transform_bundle: TransformBundle,
    min_x: float,
    max_y: float,
    meters_per_pixel: float,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    inverse_map = make_inverse_map(transform_bundle, min_x=min_x, max_y=max_y, meters_per_pixel=meters_per_pixel)
    image_float = raster.image_uint8.astype(np.float32)
    warped = warp(
        image_float,
        inverse_map=inverse_map,
        output_shape=(output_height, output_width),
        preserve_range=True,
        order=1,
        mode="constant",
        cval=0.0,
    )
    return np.clip(np.round(warped), 0, 255).astype(np.uint8)


def add_alpha_for_background(image_uint8: np.ndarray) -> np.ndarray:
    if image_uint8.ndim == 2:
        alpha = (image_uint8 > 0).astype(np.uint8) * 255
        return np.dstack([image_uint8, image_uint8, image_uint8, alpha])
    if image_uint8.shape[2] == 4:
        return image_uint8
    if image_uint8.shape[2] == 3:
        alpha = (np.any(image_uint8 > 0, axis=2)).astype(np.uint8) * 255
        return np.dstack([image_uint8, alpha])
    raise ValueError(f"Unsupported warped image shape: {image_uint8.shape}")


def write_world_file(path: Path, meters_per_pixel: float, min_x: float, max_y: float) -> None:
    center_x = min_x + meters_per_pixel / 2.0
    center_y = max_y - meters_per_pixel / 2.0
    contents = "\n".join(
        [
            f"{meters_per_pixel:.12f}",
            "0.0",
            "0.0",
            f"{-meters_per_pixel:.12f}",
            f"{center_x:.12f}",
            f"{center_y:.12f}",
        ]
    )
    path.write_text(contents + "\n", encoding="utf-8")


def write_projection_file(path: Path) -> None:
    path.write_text(WEB_MERCATOR_WKT + "\n", encoding="utf-8")


def geometry_transform_coords(
    geometry: dict[str, Any],
    coord_fn: Callable[[Sequence[float]], list[float]],
) -> dict[str, Any]:
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")
    if geom_type == "Point":
        return {"type": geom_type, "coordinates": coord_fn(coords)}
    if geom_type in {"LineString", "MultiPoint"}:
        return {"type": geom_type, "coordinates": [coord_fn(c) for c in coords]}
    if geom_type in {"Polygon", "MultiLineString"}:
        return {"type": geom_type, "coordinates": [[coord_fn(c) for c in part] for part in coords]}
    if geom_type == "MultiPolygon":
        return {
            "type": geom_type,
            "coordinates": [[[coord_fn(c) for c in ring] for ring in polygon] for polygon in coords],
        }
    return geometry


def transform_pixel_graph_to_geojson(
    pixel_graph: PixelGraphBundle,
    transform_bundle: TransformBundle,
) -> dict[str, Any]:
    def coord_fn(coord: Sequence[float]) -> list[float]:
        raw = np.asarray(coord[:2], dtype=float)
        full = normalize_pixel_coords(raw, pixel_graph.flip_y)[0]
        world = transform_bundle.forward_xy(full.reshape(1, 2))[0]
        lon, lat = web_mercator_to_lonlat(np.asarray([world[0]]), np.asarray([world[1]]))
        return [float(lon[0]), float(lat[0])]

    out_features: list[dict[str, Any]] = []
    for feature in pixel_graph.raw_geojson.get("features", []):
        geometry = feature.get("geometry") or {}
        out_geometry = geometry_transform_coords(geometry, coord_fn)
        out_feature = {
            "type": "Feature",
            "properties": feature.get("properties", {}),
            "geometry": out_geometry,
        }
        if "id" in feature:
            out_feature["id"] = feature["id"]
        out_features.append(out_feature)

    return {
        "type": "FeatureCollection",
        "name": f"{pixel_graph.path.stem}_geolocated",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": out_features,
    }


def build_summary_plot(
    raster: RasterBundle,
    pixel_graph: PixelGraphBundle,
    osm: OSMBundle,
    preview_pairs: list[dict[str, Any]],
    warped_rgba: np.ndarray,
    canvas_extent: tuple[float, float, float, float],
    summary_path: Path,
) -> None:
    min_x, min_y, max_x, max_y = canvas_extent
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 9))

    ax_left.imshow(raster.image_preview, origin="upper")
    for line in pixel_graph.lines_preview:
        ax_left.plot(line[:, 0], line[:, 1], color="#e63946", linewidth=0.8, alpha=0.55)
    for pair in preview_pairs:
        x, y = pair["image_xy_preview"]
        ax_left.scatter([x], [y], s=48, c="#ffb703", edgecolors="black", linewidths=0.5, zorder=5)
        ax_left.text(x + 6, y + 6, str(pair["id"]), color="white", fontsize=9, weight="bold")
    ax_left.set_title("Historic raster + pixel graph")
    ax_left.set_axis_off()

    for line in osm.lines_mercator:
        ax_right.plot(line[:, 0], line[:, 1], color="#1d4ed8", linewidth=0.8, alpha=0.45)
    ax_right.imshow(
        warped_rgba,
        extent=[min_x, max_x, min_y, max_y],
        origin="upper",
        alpha=0.78,
        zorder=2,
    )
    for pair in preview_pairs:
        ox, oy = pair["osm_xy_mercator"]
        ax_right.scatter([ox], [oy], s=48, c="#ffb703", edgecolors="black", linewidths=0.5, zorder=5)
        ax_right.text(ox, oy, str(pair["id"]), color="black", fontsize=9, weight="bold")
    ax_right.set_title("Warped raster on OSM graph (EPSG:3857)")
    ax_right.set_aspect("equal")
    ax_right.set_axis_off()

    plt.tight_layout()
    fig.savefig(summary_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_leaflet_html(
    output_dir: Path,
    png_filename: str,
    canvas_extent: tuple[float, float, float, float],
    preview_pairs: list[dict[str, Any]],
) -> None:
    min_x, min_y, max_x, max_y = canvas_extent
    west, south = web_mercator_to_lonlat(np.asarray([min_x]), np.asarray([min_y]))
    east, north = web_mercator_to_lonlat(np.asarray([max_x]), np.asarray([max_y]))
    south = float(south[0])
    north = float(north[0])
    west = float(west[0])
    east = float(east[0])

    markers = []
    for pair in preview_pairs:
        lon, lat = pair["osm_lonlat"]
        markers.append({"id": pair["id"], "lon": lon, "lat": lat})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Historic Map Overlay</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    html, body, #map {{
      height: 100%;
      margin: 0;
    }}
    .legend {{
      position: absolute;
      left: 16px;
      bottom: 16px;
      z-index: 999;
      background: rgba(255, 255, 255, 0.94);
      padding: 10px 12px;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.18);
      font-family: sans-serif;
      font-size: 14px;
      line-height: 1.35;
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="legend">
    <strong>Historic Raster Overlay</strong><br>
    EPSG:3857-rectified raster on top of OSM tiles.<br>
    Click markers to inspect manual control points.
  </div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const south = {south};
    const north = {north};
    const west = {west};
    const east = {east};
    const map = L.map('map');
    const bounds = [[south, west], [north, east]];
    map.fitBounds(bounds, {{padding: [20, 20]}});

    L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 22,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);

    L.imageOverlay({json.dumps(png_filename)}, bounds, {{
      opacity: 0.72,
      interactive: false
    }}).addTo(map);

    const markers = {json.dumps(markers)};
    markers.forEach((marker) => {{
      L.circleMarker([marker.lat, marker.lon], {{
        radius: 5,
        color: '#ffb703',
        fillColor: '#ffb703',
        fillOpacity: 1.0,
        weight: 1
      }}).bindPopup(`Control point ${{marker.id}}`).addTo(map);
    }});
  </script>
</body>
</html>
"""
    (output_dir / "overlay.html").write_text(html, encoding="utf-8")


def default_outdir_for_tif(tif_path: Path) -> Path:
    return tif_path.with_name(f"{tif_path.stem}_manual_geoloc")


def serializable_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for pair in pairs:
        out.append(
            {
                "id": int(pair["id"]),
                "image_xy_full": [float(pair["image_xy_full"][0]), float(pair["image_xy_full"][1])],
                "osm_xy_mercator": [float(pair["osm_xy_mercator"][0]), float(pair["osm_xy_mercator"][1])],
                "osm_lonlat": [float(pair["osm_lonlat"][0]), float(pair["osm_lonlat"][1])],
            }
        )
    return out


def save_pairs_json(
    pairs_path: Path,
    tif_path: Path,
    graph_path: Path,
    osm_path: Path,
    pairs: list[dict[str, Any]],
) -> None:
    payload = {
        "tif": str(tif_path),
        "graph": str(graph_path),
        "osm": str(osm_path),
        "pairs": serializable_pairs(pairs),
    }
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_pairs_json(pairs_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(pairs_path.read_text(encoding="utf-8"))
    raw_pairs = payload.get("pairs", payload) if isinstance(payload, dict) else payload
    pairs: list[dict[str, Any]] = []
    for idx, pair in enumerate(raw_pairs, start=1):
        full_xy = np.asarray(pair["image_xy_full"], dtype=float)
        merc = np.asarray(pair["osm_xy_mercator"], dtype=float)
        lonlat = np.asarray(pair["osm_lonlat"], dtype=float)
        pair_id = int(pair.get("id", idx))
        pairs.append(
            {
                "id": pair_id,
                "image_xy_full": full_xy,
                "osm_xy_mercator": merc,
                "osm_lonlat": lonlat,
            }
        )
    return pairs


def enrich_pairs_for_preview(pairs: list[dict[str, Any]], raster: RasterBundle) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for pair in pairs:
        preview_xy = np.asarray(
            [
                pair["image_xy_full"][0] / raster.scale_x,
                pair["image_xy_full"][1] / raster.scale_y,
            ],
            dtype=float,
        )
        enriched.append(
            {
                **pair,
                "image_xy_preview": preview_xy,
            }
        )
    return enriched


def export_outputs(
    raster: RasterBundle,
    pixel_graph: PixelGraphBundle,
    osm: OSMBundle,
    pairs: list[dict[str, Any]],
    requested_model: str,
    output_dir: Path,
    meters_per_pixel: float | None,
    max_output_side: int,
    padding_frac: float,
) -> dict[str, Any]:
    if len(pairs) < 3:
        raise ValueError("At least 3 control points are required before exporting.")

    output_dir.mkdir(parents=True, exist_ok=True)
    src = np.asarray([pair["image_xy_full"] for pair in pairs], dtype=float)
    dst = np.asarray([pair["osm_xy_mercator"] for pair in pairs], dtype=float)
    transform_bundle = estimate_transform_model(src_xy=src, dst_xy=dst, requested=requested_model)

    min_x, min_y, max_x, max_y, mpp, out_w, out_h = estimate_output_canvas(
        transform_bundle=transform_bundle,
        raster=raster,
        pixel_graph=pixel_graph,
        meters_per_pixel=meters_per_pixel,
        max_output_side=max_output_side,
        padding_frac=padding_frac,
    )

    warped = warp_raster_to_canvas(
        raster=raster,
        transform_bundle=transform_bundle,
        min_x=min_x,
        max_y=max_y,
        meters_per_pixel=mpp,
        output_height=out_h,
        output_width=out_w,
    )
    warped_rgba = add_alpha_for_background(warped)

    stem = raster.path.stem
    tif_path = output_dir / f"{stem}_geolocated.tif"
    png_path = output_dir / f"{stem}_geolocated.png"
    world_tif = output_dir / f"{stem}_geolocated.tfw"
    world_png = output_dir / f"{stem}_geolocated.pgw"
    prj_path = output_dir / f"{stem}_geolocated.prj"
    graph_path = output_dir / f"{stem}_graph_geolocated.geojson"
    summary_path = output_dir / f"{stem}_summary.png"
    report_path = output_dir / f"{stem}_report.json"

    Image.fromarray(warped).save(tif_path)
    Image.fromarray(warped_rgba).save(png_path)
    write_world_file(world_tif, meters_per_pixel=mpp, min_x=min_x, max_y=max_y)
    write_world_file(world_png, meters_per_pixel=mpp, min_x=min_x, max_y=max_y)
    write_projection_file(prj_path)

    geolocated_graph = transform_pixel_graph_to_geojson(pixel_graph, transform_bundle)
    graph_path.write_text(json.dumps(geolocated_graph, indent=2), encoding="utf-8")

    preview_pairs = enrich_pairs_for_preview(pairs, raster)
    build_summary_plot(
        raster=raster,
        pixel_graph=pixel_graph,
        osm=osm,
        preview_pairs=preview_pairs,
        warped_rgba=warped_rgba,
        canvas_extent=(min_x, min_y, max_x, max_y),
        summary_path=summary_path,
    )
    build_leaflet_html(
        output_dir=output_dir,
        png_filename=png_path.name,
        canvas_extent=(min_x, min_y, max_x, max_y),
        preview_pairs=preview_pairs,
    )

    west, south = web_mercator_to_lonlat(np.asarray([min_x]), np.asarray([min_y]))
    east, north = web_mercator_to_lonlat(np.asarray([max_x]), np.asarray([max_y]))
    report = {
        "transform_model": transform_bundle.name,
        "control_points": serializable_pairs(pairs),
        "residuals_m": transform_bundle.residuals_m.astype(float).tolist(),
        "mean_residual_m": float(np.mean(transform_bundle.residuals_m)),
        "max_residual_m": float(np.max(transform_bundle.residuals_m)),
        "epsg": 3857,
        "meters_per_pixel": float(mpp),
        "canvas_bounds_mercator": {
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y,
        },
        "canvas_bounds_lonlat": {
            "west": float(west[0]),
            "south": float(south[0]),
            "east": float(east[0]),
            "north": float(north[0]),
        },
        "outputs": {
            "warped_tif": str(tif_path),
            "warped_png": str(png_path),
            "world_tif": str(world_tif),
            "world_png": str(world_png),
            "projection": str(prj_path),
            "geolocated_graph": str(graph_path),
            "summary_png": str(summary_path),
            "overlay_html": str(output_dir / "overlay.html"),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


class ManualGeolocator:
    def __init__(
        self,
        raster: RasterBundle,
        pixel_graph: PixelGraphBundle,
        osm: OSMBundle,
        suburbs: SuburbBundle | None,
        reverse_suburb_lookup: ReverseSuburbLookup,
        pairs_path: Path,
        output_dir: Path,
        args: argparse.Namespace,
    ) -> None:
        self.raster = raster
        self.pixel_graph = pixel_graph
        self.osm = osm
        self.suburbs = suburbs
        self.reverse_suburb_lookup = reverse_suburb_lookup
        self.pairs_path = pairs_path
        self.output_dir = output_dir
        self.args = args

        self.pairs: list[dict[str, Any]] = []
        if self.pairs_path.exists():
            self.pairs = enrich_pairs_for_preview(load_pairs_json(self.pairs_path), raster=self.raster)

        self.pending_image_xy_preview: np.ndarray | None = None
        self.active_suburb: SuburbLookupResult | None = None
        self.pixel_tree = cKDTree(self.pixel_graph.snap_points_preview) if len(self.pixel_graph.snap_points_preview) else None
        self.osm_tree = cKDTree(self.osm.snap_points_mercator) if len(self.osm.snap_points_mercator) else None

        self.fig, (self.ax_left, self.ax_right) = plt.subplots(1, 2, figsize=(18, 9))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.status_text = self.fig.text(
            0.5,
            0.02,
            "",
            ha="center",
            va="bottom",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
        )
        self.draw()

    def axis_tolerance(self, ax: Any) -> float:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        return self.args.snap_radius_frac * max(abs(x1 - x0), abs(y1 - y0), 1.0)

    def maybe_snap(self, point_xy: np.ndarray, tree: cKDTree | None, candidates: np.ndarray, tolerance: float) -> np.ndarray:
        if tree is None or len(candidates) == 0:
            return point_xy
        distance, idx = tree.query(point_xy, k=1)
        if np.isfinite(distance) and float(distance) <= tolerance:
            return candidates[int(idx)]
        return point_xy

    def toolbar_is_active(self) -> bool:
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        return bool(toolbar and getattr(toolbar, "mode", ""))

    def lookup_local_suburb(self, lon: float, lat: float, mercator_xy: np.ndarray) -> SuburbLookupResult | None:
        if self.suburbs is None:
            return None

        point = Point(float(lon), float(lat))
        matches = []
        for feature in self.suburbs.features:
            try:
                if feature.geometry_lonlat.contains(point) or feature.geometry_lonlat.touches(point):
                    matches.append(feature)
            except Exception:
                continue

        if not matches:
            return None

        best = min(matches, key=lambda feature: feature.geometry_lonlat.area)
        return SuburbLookupResult(
            name=best.name,
            source=f"local polygons ({self.suburbs.path.name})",
            lonlat=(float(lon), float(lat)),
            mercator_xy=(float(mercator_xy[0]), float(mercator_xy[1])),
            feature=best,
            display_name=best.name,
        )

    def inspect_suburb_at_mercator(self, mercator_xy: np.ndarray) -> str:
        lon, lat = web_mercator_to_lonlat(np.asarray([mercator_xy[0]]), np.asarray([mercator_xy[1]]))
        lon_f = float(lon[0])
        lat_f = float(lat[0])

        result = self.lookup_local_suburb(lon_f, lat_f, mercator_xy)
        if result is None:
            result = self.reverse_suburb_lookup.lookup(lon_f, lat_f)
        if result is None:
            result = SuburbLookupResult(
                name="Unknown suburb",
                source="no local polygons / reverse lookup unavailable",
                lonlat=(lon_f, lat_f),
                mercator_xy=(float(mercator_xy[0]), float(mercator_xy[1])),
            )

        self.active_suburb = result
        return (
            f"Suburb at clicked OSM point: {result.name} "
            f"[{result.source}] at lon {result.lonlat[0]:.6f}, lat {result.lonlat[1]:.6f}"
        )

    def on_click(self, event: Any) -> None:
        if event.inaxes is None or event.button != 1 or self.toolbar_is_active():
            return
        if event.xdata is None or event.ydata is None:
            return

        if event.inaxes is self.ax_left:
            clicked = np.asarray([event.xdata, event.ydata], dtype=float)
            snapped = self.maybe_snap(
                point_xy=clicked,
                tree=self.pixel_tree,
                candidates=self.pixel_graph.snap_points_preview,
                tolerance=self.axis_tolerance(self.ax_left),
            )
            self.pending_image_xy_preview = snapped
            self.draw()
            return

        if event.inaxes is self.ax_right:
            clicked = np.asarray([event.xdata, event.ydata], dtype=float)
            snapped = self.maybe_snap(
                point_xy=clicked,
                tree=self.osm_tree,
                candidates=self.osm.snap_points_mercator,
                tolerance=self.axis_tolerance(self.ax_right),
            )
            if self.pending_image_xy_preview is None:
                status = self.inspect_suburb_at_mercator(snapped)
                self.draw(status=status)
                return

            lon, lat = web_mercator_to_lonlat(np.asarray([snapped[0]]), np.asarray([snapped[1]]))
            full_xy = np.asarray(
                [
                    self.pending_image_xy_preview[0] * self.raster.scale_x,
                    self.pending_image_xy_preview[1] * self.raster.scale_y,
                ],
                dtype=float,
            )
            pair_id = max([pair["id"] for pair in self.pairs], default=0) + 1
            self.pairs.append(
                {
                    "id": pair_id,
                    "image_xy_preview": self.pending_image_xy_preview.copy(),
                    "image_xy_full": full_xy,
                    "osm_xy_mercator": snapped.copy(),
                    "osm_lonlat": np.asarray([float(lon[0]), float(lat[0])], dtype=float),
                }
            )
            self.pending_image_xy_preview = None
            self.draw()

    def on_key(self, event: Any) -> None:
        key = (event.key or "").lower()
        if key == "d":
            if self.pairs:
                self.pairs.pop()
            self.pending_image_xy_preview = None
            self.draw()
        elif key == "s":
            self.save_pairs()
            self.draw(status=f"Saved {len(self.pairs)} control points to {self.pairs_path}")
        elif key == "e":
            self.export()
        elif key == "escape":
            self.pending_image_xy_preview = None
            self.draw()
        elif key == "q":
            self.save_pairs()
            plt.close(self.fig)

    def save_pairs(self) -> None:
        full_pairs = []
        for pair in self.pairs:
            full_pairs.append(
                {
                    "id": pair["id"],
                    "image_xy_full": pair["image_xy_full"],
                    "osm_xy_mercator": pair["osm_xy_mercator"],
                    "osm_lonlat": pair["osm_lonlat"],
                }
            )
        save_pairs_json(
            pairs_path=self.pairs_path,
            tif_path=self.raster.path,
            graph_path=self.pixel_graph.path,
            osm_path=self.osm.path,
            pairs=full_pairs,
        )

    def export(self) -> None:
        self.save_pairs()
        full_pairs = load_pairs_json(self.pairs_path)
        report = export_outputs(
            raster=self.raster,
            pixel_graph=self.pixel_graph,
            osm=self.osm,
            pairs=full_pairs,
            requested_model=self.args.model,
            output_dir=self.output_dir,
            meters_per_pixel=self.args.meters_per_pixel,
            max_output_side=self.args.max_output_side,
            padding_frac=self.args.padding_frac,
        )
        status = (
            f"Exported with {report['transform_model']} | "
            f"mean residual {report['mean_residual_m']:.2f} m | "
            f"{report['outputs']['warped_tif']}"
        )
        self.draw(status=status)

    def draw(self, status: str | None = None) -> None:
        self.ax_left.clear()
        self.ax_right.clear()

        self.ax_left.imshow(self.raster.image_preview, origin="upper")
        for line in self.pixel_graph.lines_preview:
            self.ax_left.plot(line[:, 0], line[:, 1], color="#ef4444", linewidth=0.75, alpha=0.50)
        self.ax_left.set_title(f"Historic raster + graph\n{self.raster.path.name}")
        self.ax_left.set_axis_off()

        for line in self.osm.lines_mercator:
            self.ax_right.plot(line[:, 0], line[:, 1], color="#2563eb", linewidth=0.75, alpha=0.45)

        if self.active_suburb is not None and self.active_suburb.feature is not None:
            for outline in self.active_suburb.feature.outlines_mercator:
                self.ax_right.plot(outline[:, 0], outline[:, 1], color="#16a34a", linewidth=1.6, alpha=0.85, zorder=4)

        self.ax_right.set_title(f"OSM graph\n{self.osm.path.name}")
        self.ax_right.set_aspect("equal")
        self.ax_right.set_axis_off()

        for pair in self.pairs:
            x, y = pair["image_xy_preview"]
            ox, oy = pair["osm_xy_mercator"]
            self.ax_left.scatter([x], [y], s=48, c="#ffb703", edgecolors="black", linewidths=0.45, zorder=5)
            self.ax_left.text(x + 6, y + 6, str(pair["id"]), color="white", fontsize=9, weight="bold")
            self.ax_right.scatter([ox], [oy], s=48, c="#ffb703", edgecolors="black", linewidths=0.45, zorder=5)
            self.ax_right.text(ox, oy, str(pair["id"]), color="black", fontsize=9, weight="bold")

        if self.pending_image_xy_preview is not None:
            x, y = self.pending_image_xy_preview
            self.ax_left.scatter([x], [y], s=80, c="#fb8500", edgecolors="black", linewidths=0.55, zorder=6)

        if self.active_suburb is not None:
            sx, sy = self.active_suburb.mercator_xy
            self.ax_right.scatter([sx], [sy], s=80, c="#06b6d4", edgecolors="black", linewidths=0.55, zorder=6)
            self.ax_right.text(
                0.02,
                0.98,
                f"Suburb: {self.active_suburb.name}\nSource: {self.active_suburb.source}",
                transform=self.ax_right.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92},
            )

        if status is None:
            if self.pending_image_xy_preview is None:
                status = (
                    "Click a point on the historic raster, then click the matching point on the OSM graph. "
                    "If no raster point is pending, click the OSM graph to inspect the suburb there. "
                    "Keys: s=save, d=undo, e=export, esc=clear, q=quit."
                )
            else:
                status = "Raster point selected. Click the matching OSM point on the right, or press Esc to cancel."
        self.status_text.set_text(status)

        self.fig.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])
        self.fig.canvas.draw_idle()

    def run(self) -> None:
        plt.show()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    tif_path = Path(args.tif).expanduser().resolve()
    graph_path = Path(args.graph).expanduser().resolve()
    osm_path = Path(args.osm).expanduser().resolve()
    suburb_geojson_path = Path(args.suburb_geojson).expanduser().resolve() if args.suburb_geojson else None
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else default_outdir_for_tif(tif_path)
    pairs_path = Path(args.pairs).expanduser().resolve() if args.pairs else outdir / "control_points.json"

    raster = load_raster(tif_path, preview_max_side=args.preview_max_side)
    pixel_graph = load_pixel_graph(graph_path, raster=raster)
    osm = load_osm(osm_path)
    suburbs = load_suburb_geojson(suburb_geojson_path) if suburb_geojson_path else None
    reverse_suburb_lookup = ReverseSuburbLookup(
        enabled=not args.disable_reverse_suburb_lookup,
        url=args.nominatim_url,
        email=args.nominatim_email,
        timeout=args.nominatim_timeout,
    )

    if args.export_only:
        if not pairs_path.exists():
            raise FileNotFoundError(f"--export-only requires an existing pairs file: {pairs_path}")
        pairs = load_pairs_json(pairs_path)
        report = export_outputs(
            raster=raster,
            pixel_graph=pixel_graph,
            osm=osm,
            pairs=pairs,
            requested_model=args.model,
            output_dir=outdir,
            meters_per_pixel=args.meters_per_pixel,
            max_output_side=args.max_output_side,
            padding_frac=args.padding_frac,
        )
        print(json.dumps(report, indent=2))
        return 0

    app = ManualGeolocator(
        raster=raster,
        pixel_graph=pixel_graph,
        osm=osm,
        suburbs=suburbs,
        reverse_suburb_lookup=reverse_suburb_lookup,
        pairs_path=pairs_path,
        output_dir=outdir,
        args=args,
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
