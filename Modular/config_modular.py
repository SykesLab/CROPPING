"""Configuration for the droplet cropping pipeline.

All tuneable parameters are centralised here for easy modification
and documentation purposes.
"""

from pathlib import Path

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT: Path = Path(r"C:\Users\justi\Downloads\coursework\testing_4mm\Modular")
CINE_ROOT: Path = Path(r"F:\spheres")
OUTPUT_ROOT: Path = PROJECT_ROOT / "OUTPUT"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ============================================================
# CROP SIZE BOUNDS (pixels)
# ============================================================
MAX_CNN_SIZE: int = 512
MIN_CNN_SIZE: int = 64

# ============================================================
# SAMPLING
# ============================================================
CINE_STEP: int = 10  # Process every Nth droplet

# ============================================================
# CROPPING GEOMETRY
# ============================================================
CROP_SAFETY_PIXELS: int = 3  # Vertical margin to keep sphere out of crop

# ============================================================
# GEOMETRY ANALYSIS (geom_analysis_modular.py)
# ============================================================
GEOM_MIN_AREA: int = 50  # Min pixels for connected component
SPHERE_WIDTH_RATIO: float = 0.30  # Sphere must span this fraction of width
SPHERE_CENTER_TOLERANCE: float = 0.35  # Sphere center tolerance from image center

# ============================================================
# BEST FRAME SELECTION (darkness_analysis_modular.py)
# ============================================================
N_CANDIDATES: int = 20  # Max candidate frames for geometric analysis
DARKNESS_THRESHOLD_PERCENTILE: float = 70.0  # Darkness percentile threshold
DARKNESS_WEIGHT: float = 0.05  # Weight of darkness vs centring error

# ============================================================
# CALIBRATION (crop_calibration_modular.py)
# ============================================================
CALIBRATION_PERCENTILE: float = 5.0  # Percentile for outlier-robust calibration
