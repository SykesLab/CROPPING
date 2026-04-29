"""
Pipeline configuration loaded from preprocessing_config.yaml.

Module-level variables are exposed for backward compatibility.

Path semantics:
- OUTPUT_ROOT: parent directory under which timestamped runs live
  (default: Preprocessing/output). Every Start creates a fresh run dir
  under OUTPUT_ROOT/runs/.
- RUN_ROOT: the active run's directory. Set by the GUI immediately
  before spawning workers. Defaults to OUTPUT_ROOT for legacy callers
  that don't go through run_io.

Both are env-var-overridable (CROPPING_OUTPUT_ROOT, CROPPING_RUN_ROOT)
so worker subprocesses on Windows (which re-import config from scratch
under spawn semantics) inherit whatever the parent process set.
"""

import os
from pathlib import Path
from typing import Optional

from config_loader import load_config

_config = load_config()

# --- Paths ---
PROJECT_ROOT: Path = Path(__file__).parent

CINE_ROOT: Path = PROJECT_ROOT / _config.paths.cine_root

# OUTPUT_ROOT is the parent dir for runs. Lowercase 'output/' by default
# unless the YAML says otherwise (legacy default in the YAML is './OUTPUT').
_env_output = os.environ.get("CROPPING_OUTPUT_ROOT")
OUTPUT_ROOT: Path = (
    Path(_env_output) if _env_output else PROJECT_ROOT / _config.paths.output_root
)

# RUN_ROOT is the current run's dir. Set by the GUI on Start. Defaults
# to OUTPUT_ROOT so any legacy callers (focus_analysis CLI, ad-hoc
# scripts) keep writing to the flat layout they expect.
_env_run = os.environ.get("CROPPING_RUN_ROOT")
RUN_ROOT: Path = Path(_env_run) if _env_run else OUTPUT_ROOT

# --- Crop size bounds (pixels) ---
MAX_CNN_SIZE: int = _config.crop.max_cnn_size
MIN_CNN_SIZE: int = _config.crop.min_cnn_size
CROP_SAFETY_PIXELS: int = _config.crop.safety_pixels

# --- Sampling ---
CINE_STEP: int = _config.sampling.cine_step

# --- Parallelisation ---
N_CORES: Optional[int] = None  # None = use all available cores

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
