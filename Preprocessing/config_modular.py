"""Configuration for the droplet cropping pipeline.

Loads settings from preprocessing_config.yaml and exposes them as
module-level variables for backward compatibility with existing code.

All tuneable parameters are defined in preprocessing_config.yaml.
Edit that file to modify pipeline behavior.

Note: CINE_ROOT and OUTPUT_ROOT are defaults that can be changed
via the GUI at runtime.
"""

from pathlib import Path
from typing import Optional

from config_loader import load_config, get_config

# ============================================================
# LOAD CONFIGURATION FROM YAML
# ============================================================
_config = load_config()

# ============================================================
# PATHS (defaults - configurable in GUI)
# ============================================================
PROJECT_ROOT: Path = Path(__file__).parent
CINE_ROOT: Path = PROJECT_ROOT / _config.paths.cine_root
OUTPUT_ROOT: Path = PROJECT_ROOT / _config.paths.output_root
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ============================================================
# CROP SIZE BOUNDS (pixels)
# ============================================================
MAX_CNN_SIZE: int = _config.crop.max_cnn_size
MIN_CNN_SIZE: int = _config.crop.min_cnn_size

# ============================================================
# SAMPLING
# ============================================================
CINE_STEP: int = _config.sampling.cine_step  # Process every Nth droplet

# ============================================================
# CROPPING GEOMETRY
# ============================================================
CROP_SAFETY_PIXELS: int = _config.crop.safety_pixels  # Vertical margin to keep sphere out of crop

# ============================================================
# GEOMETRY ANALYSIS (geom_analysis_modular.py)
# ============================================================
GEOM_MIN_AREA: int = _config.geometry.min_area  # Min pixels for connected component
SPHERE_WIDTH_RATIO: float = _config.geometry.sphere_width_ratio  # Sphere must span this fraction of width
SPHERE_CENTER_TOLERANCE: float = _config.geometry.sphere_center_tolerance  # Sphere center tolerance from image center

# ============================================================
# BEST FRAME SELECTION (darkness_analysis_modular.py)
# ============================================================
N_CANDIDATES: int = _config.best_frame.n_candidates  # Max candidate frames for geometric analysis
DARKNESS_THRESHOLD_PERCENTILE: float = _config.best_frame.darkness_threshold_percentile  # Darkness percentile threshold
DARKNESS_WEIGHT: float = _config.best_frame.darkness_weight  # Weight of darkness vs centring error

# ============================================================
# CALIBRATION (crop_calibration_modular.py)
# ============================================================
CALIBRATION_PERCENTILE: float = _config.calibration.percentile  # Percentile for outlier-robust calibration

# ============================================================
# FOCUS METRICS (focus_metrics_modular.py)
# ============================================================
FOCUS_METRICS_ENABLED: bool = _config.focus.enabled  # Compute focus metrics for each crop
FOCUS_PRIMARY_METRIC: str = _config.focus.primary_metric  # Primary metric for classification
# Thresholds set to None = auto-determine from dataset statistics
FOCUS_SHARP_THRESHOLD: Optional[float] = _config.focus.sharp_threshold  # Scores above this are "sharp"
FOCUS_BLUR_THRESHOLD: Optional[float] = _config.focus.blur_threshold  # Scores below this are "blurry"
