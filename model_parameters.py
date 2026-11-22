"""Per-stage parameter dataclasses, serialized to / from `config/model_parameters.json`."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict, field

@dataclass
class PreprocessingParameters:
    """Parameters for preprocessing/tiling stage."""
    # Tile size parameters
    min_tile_size: int = 256
    max_tile_size: int = 256
    target_tile_size: int = 256
    
    # Overlap and padding
    overlap_pct: float = 0.05
    pad_mode: str = "edge"  # "constant", "edge", "reflect"
    
    # Processing
    max_workers: Optional[int] = 4  # None = auto-detect

    # PDF input handling — DPI used to rasterize PDFs into PNG before any model runs
    pdf_dpi: int = 300


@dataclass
class RoadExtractionParameters:
    """Parameters for road extraction/prediction stage."""
    batch_size: int = 256
    window_size: tuple = (256, 256)
    stride: int = 96              # 62% overlap — ~4 patches per pixel, denser than 170 default without exploding TTA cost
    use_tta: bool = True          # Test-Time Augmentation (6 augmentations averaged)
    auto_scale: bool = True       # Automatically find optimal inference scale
    fixed_scale: Optional[float] = None  # Override auto_scale with an explicit factor.
                                  # Pred.py has always supported this; it was never
                                  # exposed. Atlas sheets need ~0.5 (they out-resolve
                                  # the 1877 training map, so roads read too wide).
    save_patches: bool = False    # Save scale-optimisation sample patches to disk
    border_margin: float = 0.1   # Fraction of image border to exclude from scale sampling
    skip_empty_patches: bool = True  # Skip near-uniform patches (blank margins, water)
    min_patch_std: float = 5.0       # Std-dev threshold (0–255 scale) below which patches are skipped
    use_clahe: bool = True           # Adaptive histogram equalisation before tiling — sharpens thin ink on cream paper
    clahe_clip_limit: float = 2.0    # CLAHE clip threshold (higher = more contrast boost)
    clahe_grid_size: int = 16        # CLAHE tile grid (NxN); smaller = more local



@dataclass
class RoadVectorizationParameters:
    """Parameters for road vectorization stage."""

    cleaning_passes: int = 2

    Low_threshold: float = 0.35
    High_threshold: float = 0.6
    spur_threshold: int = 3

    # Pre-skeleton U-Net hygiene
    max_hole_size: int = 50              # interior speckle holes (≤ N px) filled before skeleton
    gap_bridge_length: int = 7           # oriented-kernel closing for short linear gaps (px); 0 disables
    junction_merge_radius: int = 6       # collapse junction-pixel clusters within N px to one centre
    min_object_size: int = 100           # drop binary connected components smaller than N px
    min_skeleton_component: int = 20     # drop skeleton islands smaller than N px

    # Endpoint reconnection (heals broken roads at the graph level)
    endpoint_reconnect_radius_factor: float = 2.0  # search radius = road_width × this; 0 disables
    endpoint_reconnect_max_bearing_deg: float = 30.0  # max bearing misalignment for a heal

    # (A) Image-evidence pruning — drop edges where the BASE map has no
    # casing-pair ink on either side of the predicted centreline.
    evidence_enabled: bool = True
    evidence_ink_lum_threshold: int = 160         # samples darker than this count as ink
    evidence_min_casing_fraction: float = 0.30    # < this fraction → drop edge
    evidence_casing_offset_factor: float = 0.60   # × road_width/2 = casing distance from centreline
    evidence_casing_extra_search_px: int = 2      # tolerance band around the casing distance

    # (B) Active-contour ("snake") snapping — pull edge interiors onto local ink.
    snake_enabled: bool = True
    snake_iterations: int = 8
    snake_search_radius_px: Optional[int] = None  # None = road_width // 2
    snake_image_weight: float = 0.55              # blend toward darkest-pixel target
    snake_smoothness_weight: float = 0.45         # blend toward neighbour midpoint


@dataclass
class DetectionParameters:
    """Parameters for text detection stage (DBNet++)."""
    patch_size: int = 256
    batch_size: int = 128
    stride: int = 176
    confidence_threshold: float = 0.5
    overlap_threshold: float = 0.5
    # Polygon extraction
    min_area: int = 50             # minimum contour area (px²) to keep
    unshrink_ratio: float = 1.5   # pyclipper expansion ratio
    nms_iou_threshold: float = 0.3 # IoU threshold for duplicate polygon removal



@dataclass
class OCRParameters:
    """Parameters for OCR stage."""
    # Polygon extraction
    buffer_px: int = 10
    batch_size: int = 32

    # Inference parameters (if needed)
    confidence_threshold: float = 0.5


@dataclass
class GeolocationParameters:
    """Parameters for geolocation stage."""
    # Road propagation
    buffer_threshold: float = 50.0  # legacy/unused; kept for back-compat with old configs

    # Post-OCR fuzzy matching against the street-name dictionary
    fuzzy_match_threshold: int = 75   # 0-100; matches below this score are dropped

    # Road-name propagation across the road graph
    propagation_initial_radius: float = 100.0    # px; spatial buffer for first-pass label assignment
    propagation_iterations: int = 20             # max passes of label diffusion
    propagation_angle_tolerance: float = 40.0    # degrees; only propagate along edges within this bearing diff
    propagation_overtake_multiplier: float = 1.3 # incumbent name is replaced when challenger conf > current * this
    min_component_size: int = 5                  # drop connected components with fewer nodes

    # Anchor matching (OSM intersection ↔ historic intersection)
    anchor_match_threshold: float = 0.7   # min score for an anchor pair to be accepted

    # Intersection name resolution (combines graph-derived confidence with fuzzy match score)
    intersection_fuzzy_weight: float = 0.7        # weight on the fuzzy-match component; graph weight = 1 - this
    intersection_verification_threshold: int = 80 # raw fuzz score above which an intersection is marked is_verified

    #TPS and RANSCAN
    RANSAC_THRESHOLD: float = 0.01
    TPS_SMOOTHING: float = 0.02

@dataclass
class SimilarityParameters:
    # Registration
    enable_registration: bool = True
    registration_min_matches: int = 8
    registration_strict_threshold: float = 0.25
    registration_probe_radius: float = 400.0   # metres; Pass 0 KDTree probe radius
    ransac_residual_threshold: float = 200.0
    
    # Search
    initial_search_radius: float = 150.0   
    max_candidates_edges: int = 15         
    max_paths_to_solve: int = 5            

    # Relaxation
    relaxation_search_radius: float = 350.0 
    relaxation_max_candidates: int = 40    
    relaxation_max_paths: int = 5          
    relaxation_length_min: float = 0.1     
    relaxation_length_max: float = 6.0     
    
    # Gates
    # bearing_hard_penalty_threshold: arccos(dot)/π fraction in [0, 1] (0.0=parallel,
    # 0.5=90°, 1.0=opposite). A soft penalty applies across the full range via
    # bearing_weight; this only triggers an *additional* hard penalty above the
    # threshold. Default 0.5 = hard spike only for >90° (catastrophic) errors.
    bearing_hard_penalty_threshold: float = 0.5
    pre_filter_bearing_diff: float = 45.0   # degrees; edge-level pre-filter
    min_len_ratio: float = 0.2        
    max_len_ratio: float = 5.0        
    
    # Optimization
    virtual_node_snap_threshold: float = 15.0
    early_exit_score: float = 0.1

    # Metric CRS for distance calculations. None = auto-pick the WGS84 UTM zone
    # for the historic graph's centroid (recommended). Override with an explicit
    # EPSG (e.g. "EPSG:3308" for NSW Lambert) only if you have a reason.
    crs_meters: Optional[str] = None

    # Weights (Updated for V5.3)
    shape_weight: float = 0.30       
    bearing_weight: float = 0.30      # Slightly reduced to allow topology to weigh in
    displacement_weight: float = 0.10 
    sinuosity_weight: float = 0.20    # INCREASED: Punish zig-zags
    topology_weight: float = 0.10     # NEW: Punish sharp 90-degree turns
    
    # Absolute source-to-path displacement penalty.
    # shape_weight alone cannot catch a parallel-street match: shape_score saturates
    # at 1.0 once the deviation passes half the link length, so 50 m off and 500 m off
    # score identically, and a saturated shape term contributes only shape_weight to a
    # score that must reach medium_confidence_threshold to be rejected. This term is
    # unsaturated and absolute. cap = 0 disables it.
    source_offset_cap_m: float = 0.0
    source_offset_weight: float = 0.0

    # Thresholds
    high_confidence_threshold: float = 0.25
    medium_confidence_threshold: float = 0.60
    low_confidence_threshold: float = 0.75
    very_low_confidence_threshold: float = 0.90
    
    drift_tolerance_meters: float = 80.0
    flow_magnitude_weight: float = 0.3
    flow_direction_weight: float = 0.7

    highway_types: Tuple[str] = ('tertiary', 'secondary', 'trunk', 'primary', 'residential', 'unclassified')



@dataclass
class VisualizationParameters:
    """Parameters for visualization stage."""
    # Output settings
    output_format: str = 'png'
    dpi: int = 300
    figure_size: tuple = (12, 8)


@dataclass
class ModelParameters:
    """Complete model parameters configuration for all pipeline stages."""
    
    preprocessing: PreprocessingParameters = field(default_factory=PreprocessingParameters)
    road_extraction: RoadExtractionParameters = field(default_factory=RoadExtractionParameters)
    road_vectorization: RoadVectorizationParameters = field(default_factory=RoadVectorizationParameters)
    detection: DetectionParameters = field(default_factory=DetectionParameters)
    ocr: OCRParameters = field(default_factory=OCRParameters)
    geolocation: GeolocationParameters = field(default_factory=GeolocationParameters)
    visualization: VisualizationParameters = field(default_factory=VisualizationParameters)
    similarity: SimilarityParameters = field(default_factory=SimilarityParameters)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'preprocessing': asdict(self.preprocessing),
            'road_extraction': asdict(self.road_extraction),
            'road_vectorization': asdict(self.road_vectorization),
            'detection': asdict(self.detection),
            'ocr': asdict(self.ocr),
            'geolocation': asdict(self.geolocation),
            'visualization': asdict(self.visualization),
            'similarity': asdict(self.similarity),
        }
    
    # Legacy → current renames applied during from_dict, per stage.
    _LEGACY_KEY_MAP = {
        'similarity': {'max_bearing_diff': 'bearing_hard_penalty_threshold'},
    }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelParameters':
        """Create from dictionary, migrating any known legacy keys."""
        def _migrate(stage: str) -> Dict[str, Any]:
            section = dict(data.get(stage, {}))
            for legacy, current in cls._LEGACY_KEY_MAP.get(stage, {}).items():
                if legacy in section:
                    legacy_value = section.pop(legacy)
                    section.setdefault(current, legacy_value)
            return section

        return cls(
            preprocessing=PreprocessingParameters(**_migrate('preprocessing')),
            road_extraction=RoadExtractionParameters(**_migrate('road_extraction')),
            road_vectorization=RoadVectorizationParameters(**_migrate('road_vectorization')),
            detection=DetectionParameters(**_migrate('detection')),
            ocr=OCRParameters(**_migrate('ocr')),
            geolocation=GeolocationParameters(**_migrate('geolocation')),
            visualization=VisualizationParameters(**_migrate('visualization')),
            similarity=SimilarityParameters(**_migrate('similarity')),
        )
    
    def save(self, filepath: str) -> None:
        """Save parameters to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'ModelParameters':
        """Load parameters from JSON file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            # Return defaults if file doesn't exist
            return cls()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    def update(self, **kwargs) -> None:
        """Update parameters from keyword arguments.
        
        Example:
            params.update(road_extraction__threshold=0.4, preprocessing__overlap_pct=0.1)
        """
        for key, value in kwargs.items():
            if '__' in key:
                stage, param = key.split('__', 1)
                if hasattr(self, stage):
                    stage_obj = getattr(self, stage)
                    if hasattr(stage_obj, param):
                        setattr(stage_obj, param, value)
                    else:
                        raise ValueError(f"Parameter '{param}' not found in {stage}")
                else:
                    raise ValueError(f"Stage '{stage}' not found")
            else:
                # Direct attribute update
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise ValueError(f"Parameter '{key}' not found")


# Global instance - can be imported and used by all modules
_global_params: Optional[ModelParameters] = None


def get_parameters() -> ModelParameters:
    """Get the global parameters instance."""
    global _global_params
    if _global_params is None:
        _global_params = ModelParameters()
    return _global_params


def set_parameters(params: ModelParameters) -> None:
    """Set the global parameters instance."""
    global _global_params
    _global_params = params


def load_parameters(filepath: str) -> ModelParameters:
    """Load parameters from file and set as global."""
    params = ModelParameters.load(filepath)
    set_parameters(params)
    return params


def save_parameters(filepath: str, params: Optional[ModelParameters] = None) -> None:
    """Save parameters to file."""
    if params is None:
        params = get_parameters()
    params.save(filepath)


# Default parameters file path
DEFAULT_PARAMS_FILE = "config/model_parameters.json"

