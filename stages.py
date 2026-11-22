"""Single source of truth for pipeline stages. Consumers: pipeline.py, gui.py, config.py."""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Stage:
    key: str            # config / GUI checkbox key, e.g. 'pred'
    module: str         # importable name under `modules`, e.g. 'road_extraction'
    display: str        # human-readable name, e.g. 'Road Extraction'
    description: str    # one-line tooltip
    default_enabled: bool


STAGES: Tuple[Stage, ...] = (
    Stage('pred',       'road_extraction',     'Road Extraction',
          'U-Net sliding-window inference → probability map', True),
    Stage('graph',      'road_vectorization',  'Road Vectorization',
          'Hysteresis + skeletonisation → GeoJSON graph', True),
    Stage('text',       'detection',           'Text Detection',
          'DBNet++ multi-scale inference → text polygons', False),
    Stage('ocr',        'ocr',                 'OCR',
          'TrOCR batch inference → street name strings', False),
    Stage('geoloc',     'geolocation',         'Geolocation',
          'OSM anchor matching + TPS warp → lat/lon GeoJSON', False),
    Stage('similarity', 'similarity',          'Similarity',
          'Hausdorff + bearing scoring → OSM road matches', False),
)


STAGES_BY_KEY: Dict[str, Stage] = {s.key: s for s in STAGES}
STAGES_BY_MODULE: Dict[str, Stage] = {s.module: s for s in STAGES}


def default_modules_config() -> Dict[str, bool]:
    return {s.key: s.default_enabled for s in STAGES}
