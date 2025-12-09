# config_modular.py
#
# Central configuration for the droplet cropping pipeline.
# All tuneable parameters in one place.

from pathlib import Path

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(r"C:\Users\justi\Downloads\coursework\testing_4mm\Modular")
CINE_ROOT = Path(r"F:\spheres")
OUTPUT_ROOT = PROJECT_ROOT / "OUTPUT"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ============================================================
# CROP SIZE BOUNDS
# ============================================================
MAX_CNN_SIZE = 512      # Maximum crop size (pixels)
MIN_CNN_SIZE = 64       # Minimum crop size (pixels)

# ============================================================
# SAMPLING
# ============================================================
CINE_STEP = 10          # Process every Nth droplet

# ============================================================
# CROPPING GEOMETRY
# ============================================================
CROP_SAFETY_PIXELS = 3  # Vertical margin to keep sphere out of crop

# ============================================================
# GEOMETRY ANALYSIS (geom_analysis_modular.py)
# ============================================================
GEOM_MIN_AREA = 50              # Min pixels for a connected component
SPHERE_WIDTH_RATIO = 0.30       # Sphere must span this fraction of image width
SPHERE_CENTER_TOLERANCE = 0.35  # Sphere center must be within this fraction of image center

# ============================================================
# BEST FRAME SELECTION (darkness_analysis_modular.py)
# ============================================================
# Full output mode: darkness curve + candidate-based geometry
N_CANDIDATES = 20                   # Max candidate frames to analyse geometrically
DARKNESS_THRESHOLD_PERCENTILE = 70  # Only consider frames with darkness above this percentile
DARKNESS_WEIGHT = 0.05              # Weight of darkness in scoring (centring error is 1.0)

# ============================================================
# CALIBRATION (crop_calibration_modular.py)
# ============================================================
CALIBRATION_PERCENTILE = 5  # Use Nth percentile of allowed heights (robust to outliers)
