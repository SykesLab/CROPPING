# config_modular.py
from pathlib import Path

# Where this modular code lives
PROJECT_ROOT = Path(r"C:\Users\justi\Downloads\coursework\testing_4mm\Modular")

# Where your .cine folders live (adjust if needed)
CINE_ROOT = Path(r"F:\spheres")

# Where outputs should go
OUTPUT_ROOT = PROJECT_ROOT / "OUTPUT"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Crop size bounds
MAX_CNN_SIZE = 512
MIN_CNN_SIZE = 64

# Every Nth droplet to use (global + per-folder)
CINE_STEP = 10

# Vertical safety margin (pixels) to keep sphere out of crops
CROP_SAFETY_PIXELS = 3
