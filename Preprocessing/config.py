"""
Pipeline configuration loaded from preprocessing_config.yaml.

Module-level variables are exposed for backward compatibility.
CINE_ROOT and OUTPUT_ROOT can be changed at runtime via the GUI.
"""

from pathlib import Path
from typing import Optional

from config_loader import load_config

_config = load_config()

# --- Paths ---
PROJECT_ROOT: Path = Path(__file__).parent
CINE_ROOT: Path = PROJECT_ROOT / _config.paths.cine_root
OUTPUT_ROOT: Path = PROJECT_ROOT / _config.paths.output_root
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# --- Crop size bounds (pixels) ---
MAX_CNN_SIZE: int = _config.crop.max_cnn_size
MIN_CNN_SIZE: int = _config.crop.min_cnn_size
CROP_SAFETY_PIXELS: int = _config.crop.safety_pixels

# --- Sampling ---
CINE_STEP: int = _config.sampling.cine_step

# --- Geometry detection ---
GEOM_MIN_AREA: int = _config.geometry.min_area
SPHERE_WIDTH_RATIO: float = _config.geometry.sphere_width_ratio
SPHERE_CENTER_TOLERANCE: float = _config.geometry.sphere_center_tolerance

# --- Best frame selection ---
N_CANDIDATES: int = _config.best_frame.n_candidates
DARKNESS_THRESHOLD_PERCENTILE: float = _config.best_frame.darkness_threshold_percentile
DARKNESS_WEIGHT: float = _config.best_frame.darkness_weight

# --- Calibration ---
CALIBRATION_PERCENTILE: float = _config.calibration.percentile

# --- Focus metrics ---
FOCUS_METRICS_ENABLED: bool = _config.focus.enabled
FOCUS_PRIMARY_METRIC: str = _config.focus.primary_metric
FOCUS_SHARP_THRESHOLD: Optional[float] = _config.focus.sharp_threshold
FOCUS_BLUR_THRESHOLD: Optional[float] = _config.focus.blur_threshold
