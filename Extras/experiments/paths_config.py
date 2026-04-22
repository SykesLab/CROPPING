"""
Shared path configuration for experiment scripts.

Edit the paths below to match your local setup. All experiment scripts
import from this file instead of hardcoding paths.
"""

from pathlib import Path

# ── Root directories ──────────────────────────────────────────────────────
# Base directory containing preprocessing output (material subfolders)
CROP_BASE = Path(r"")

# Directory containing .cine files for geometry / preprocessing tests
CINE_FOLDER = Path(r"")

# Trained model checkpoint
MODEL_PATH = Path(r"")

# ── Calibration data ─────────────────────────────────────────────────────
# Directory containing calibration images (processed sphere frames)
CALIB_IMG_DIR = Path(r"")

# CSV with per-frame calibration measurements
CALIB_CSV = Path(r"")

# ── Batch processing ─────────────────────────────────────────────────────
# Input directory for batch runner
BATCH_INPUT_DIR = Path(r"")

# Output directory for batch runner
BATCH_OUTPUT_DIR = Path(r"")

# ── Stage 2 packager ─────────────────────────────────────────────────────
STAGE1_ROOT = Path(r"")
STAGE2_OUTPUT_DIR = Path(r"")

# ── Derived paths (don't edit) ───────────────────────────────────────────
SHARP_CSV = CROP_BASE / "Focus" / "focus_classified_all.csv" if str(CROP_BASE) else Path("")
