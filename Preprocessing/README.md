# Droplet Preprocessing Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Automated preprocessing of high-speed camera footage for droplet analysis. Extracts droplet crops from Phantom .cine files with focus quality assessment.

## Features

- **Automatic droplet detection** — Connected component analysis identifies droplets and spheres, with vignetting exclusion and robust filtering
- **Intelligent frame selection** — Weighted scoring balances droplet centring (gap symmetry between droplet-to-sphere and droplet-to-image-top) with darkness, ensuring pre-collision frames with optimal geometry
- **Sphere-excluding crops** — Crops shift upward automatically to exclude the injection sphere while maintaining consistent dimensions for CNN input
- **Global crop calibration** — Conservative percentile-based sizing ensures uniform crops across all folders
- **Six focus quality metrics** — Laplacian variance, Tenengrad, Brenner gradient, and three additional edge-based measures
- **Adaptive focus classification** — Per-folder percentile thresholds (75th/25th) ensure balanced sharp/medium/blurry distribution regardless of optical conditions
- **GUI with live preview** — Real-time thumbnail display, progress tracking with ETA, and configurable processing options
- **Flexible execution modes** — Quick test (validation), safe mode (single-process), step sampling (every Nth droplet), and profiling output
- **Parallel processing** — Droplet-level parallelisation across all CPU cores for maximum throughput

## Quick Start

### 1. Set Up Environment

```bash
# Create virtual environment (recommended)
python -m venv droplet_env
droplet_env\Scripts\activate  # Windows
# source droplet_env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Verify setup
python setup_environment.py
```

### 2. Install Phantom SDK

The Phantom SDK (pyphantom) is required to read .cine files:

1. Download from [Vision Research](https://www.phantomhighspeed.com/resourcesandsupport/)
2. Install following their instructions
3. Verify: `python -c "import pyphantom; print('OK')"`

### 3. Run Pipeline

```bash
python gui.py
```

### 4. Configure in GUI

1. **CINE Root** — Browse to either:
   - A parent folder containing subfolders with .cine files, OR
   - A single folder directly containing .cine files
2. **Output Folder** — Use default `./OUTPUT` or browse to custom location
3. Click **Continue**, configure options, then **Run Pipeline**

## Project Structure

```
droplet-preprocessing/
├── gui.py                   # GUI interface (main entry point)
├── config.py                # Default configuration
├── setup_environment.py     # Environment verification
├── requirements.txt         # Python dependencies
├── README.md
├── LICENSE
│
├── pipeline_global.py       # Global calibration pipeline
├── pipeline_folder.py       # Per-folder pipeline
│
├── cine_io.py               # .cine file reading and SDK import
├── darkness_analysis.py     # Best frame selection
├── geom_analysis.py         # Droplet geometry
├── crop_calibration.py      # Crop size calibration
├── cropping.py              # Image cropping
├── image_utils.py           # Image utilities
│
├── focus_metrics.py         # Focus quality metrics
├── focus_analysis.py        # Standalone focus analysis
│
├── output_writer.py         # CSV/output generation
├── plotting.py              # Visualisation
├── parallel_utils.py        # Multiprocessing
├── profiling.py             # Timing and profiling utilities
├── workers.py               # Parallel worker functions
│
└── OUTPUT/                  # Generated outputs (default)
    ├── {folder}/
    │   ├── *_crop.png       # Droplet crops
    │   ├── *_summary.csv    # Metadata
    │   └── *_overlay.png    # Visualisations
    │
    └── Focus/               # Focus classification results
        ├── sharp_crops.csv
        └── {folder}/        # Sharp images by folder
```

## Pipeline Modes

### Global Mode
- Calibrates crop size across ALL folders
- Consistent crop dimensions for CNN training
- Better for large datasets with multiple folders
- **Only available when selecting a parent folder with multiple subfolders**

### Per-Folder Mode
- Independent processing per folder
- Crop sizes may vary between folders
- **Automatically selected when processing a single folder**

## Focus Classification

Focus classification is **built into the pipeline** — enable "Focus classification" checkbox in the GUI.

When enabled, the pipeline automatically:
- Computes focus metrics (Laplacian, Tenengrad, Brenner) for each crop
- Classifies crops as sharp/medium/blurry using per-folder thresholds
- Copies sharp crops to `OUTPUT/Focus/{folder}/` for CNN training

### Per-Folder Thresholds

Focus classification uses per-folder thresholds to ensure diverse training data:
- Each folder contributes its sharpest ~25% to the training set
- Handles varying optical conditions between sessions
- Produces robust models that generalise to new setups

### Standalone Analysis

To rerun focus analysis on existing crops without reprocessing:
```bash
python focus_analysis.py path/to/OUTPUT
```

## Output Files

### Directory Structure
```
OUTPUT/
├── {folder}/
│   ├── {droplet}{cam}_crop.png       # Grayscale droplet crop
│   ├── {droplet}{cam}_darkness.png   # Darkness curve plot (full output mode)
│   ├── {droplet}{cam}_overlay.png    # Geometric overlay plot (full output mode)
│   └── {folder}_summary.csv          # Metadata for all crops
│
├── Focus/                             # Focus classification results
│   ├── sharp_crops.csv               # All sharp crops with metrics
│   ├── focus_classified_all.csv      # All crops with classifications
│   ├── focus_folder_stats.csv        # Per-folder threshold statistics
│   ├── focus_classification_summary.png  # Distribution visualisation
│   └── {folder}/                     # Sharp crop copies by folder
│
└── focus_metrics_computed.csv        # Combined metrics from all folders
```

### Crop Images
- `{droplet}{cam}_crop.png` — Grayscale droplet crop (e.g., `sphere0843g_crop.png`)
- Standard size across all folders when using Global mode (e.g., 388x388)

### Visualisation Plots (Full Output Mode)
- `{droplet}{cam}_darkness.png` — Darkness curve showing frame selection
- `{droplet}{cam}_overlay.png` — Geometric overlay showing detected droplet boundaries

### Summary CSV
Each folder produces `{folder}_summary.csv`:

| Column | Description |
|--------|-------------|
| droplet_id | Droplet identifier |
| camera | g (green) or v (violet) |
| cine_file | Source .cine filename |
| best_frame | Selected frame index |
| dark_fraction | Darkness metric at best frame |
| y_top, y_bottom, y_sphere | Detected geometry |
| crop_size_px | Crop dimensions |
| crop_path | Path to crop image |
| laplacian_var | Focus metric (edge strength) |
| tenengrad | Focus metric (gradient magnitude) |
| tenengrad_var | Focus metric (gradient variance) |
| brenner | Focus metric (local contrast) |
| norm_laplacian | Focus metric (normalised Laplacian) |
| energy_gradient | Focus metric (energy of gradient) |

## Configuration

Key parameters can be adjusted in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_CNN_SIZE` | 512 | Maximum crop size (pixels) |
| `MIN_CNN_SIZE` | 64 | Minimum crop size (pixels) |
| `CROP_SAFETY_PIXELS` | 3 | Margin between crop and sphere |
| `N_CANDIDATES` | 20 | Candidate frames for geometry analysis |
| `CALIBRATION_PERCENTILE` | 5.0 | Percentile for robust crop sizing |
| `DARKNESS_THRESHOLD_PERCENTILE` | 70.0 | Darkness percentile for candidate filtering |
| `DARKNESS_WEIGHT` | 0.05 | Weight of darkness vs centring in frame scoring |

## Troubleshooting

### "pyphantom not found"
Install the Phantom SDK from Vision Research.

### "No .cine files found"
Check your CINE Root path in the GUI points to folders containing .cine files.

### Out of memory
Increase the Step value (processes every Nth droplet) or process fewer folders.

### GUI not responding
The pipeline runs in a background thread. Check the console for progress.

## Dependencies

- Python 3.8+
- numpy, pandas, scipy
- opencv-python
- matplotlib
- tqdm
- tkinter (included with Python)
- pyphantom (Phantom SDK — from Vision Research)

## License

This project is licensed under the GNU General Public License v3.0 — see the [LICENSE](LICENSE) file for details.

## Citation

This pipeline implements preprocessing for droplet defocus estimation based on:

> Wang, J., et al. (2022). "Three-dimensional droplet sizing and tracking based on a single deep neural network." *Experiments in Fluids*, 63, 166.

If you use this software in your research, please cite appropriately.
