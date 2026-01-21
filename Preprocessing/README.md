# Droplet Preprocessing Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Extracts droplet crops from Phantom .cine files for CNN training. Handles frame selection, cropping, and focus-based filtering.

## How It Works

Each .cine file contains a high-speed recording of a droplet falling onto a sphere. The pipeline finds the best frame to crop from, where "best" means the droplet is fully visible, hasn't yet collided with the sphere, and is roughly centred in the frame.

Frame selection uses connected component analysis to locate the droplet and sphere in each frame. It scores frames by how symmetric the gaps are (droplet-to-image-top vs droplet-to-sphere), with a small weight on darkness to prefer frames where the droplet is more opaque. Frames where the droplet has already touched the sphere are rejected.

Crops are sized to fit the droplet while excluding the sphere. If the crop would include the sphere, it shifts upward. When processing multiple folders, a global calibration pass sets a single crop size (using a conservative low percentile) so all outputs are the same dimensions for CNN training.

Focus quality is measured using six edge-based metrics (Laplacian variance, Tenengrad, Brenner, etc). Classification into sharp/medium/blurry uses per-folder thresholds at the 75th and 25th percentiles, so each folder contributes its sharpest ~25% regardless of lighting conditions.

The GUI shows live thumbnails, progress, and ETA. Processing runs in parallel across all CPU cores, with options for quick validation runs, single-process mode for debugging, and step sampling to process every Nth droplet.

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

The Phantom SDK (pyphantom) is required to read .cine files. It's not publicly available - check with your department, camera owner, or contact Vision Research directly.

Verify installation: `python -c "import pyphantom; print('OK')"`

### 3. Run Pipeline

```bash
python gui.py
```

### 4. Configure in GUI

1. **CINE Root**: either a parent folder containing subfolders with .cine files, or a single folder with .cine files directly
2. **Output Folder**: defaults to `./OUTPUT`
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

**Global mode** calibrates crop size across all folders, so every crop has the same dimensions. Only available when you select a parent folder with multiple subfolders.

**Per-folder mode** processes each folder independently. Crop sizes may vary between folders. Automatically selected when processing a single folder.

## Focus Classification

Enable the "Focus classification" checkbox in the GUI. This computes focus metrics for each crop, classifies them as sharp/medium/blurry, and copies the sharp ones to `OUTPUT/Focus/{folder}/`.

### Per-Folder Thresholds

Each folder gets its own threshold based on its own distribution:
- Sharpest ~25% from each folder goes to training
- Accounts for different lighting/focus between sessions

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

Grayscale droplet crops, e.g. `sphere0843g_crop.png`. In Global mode, all crops are the same size (e.g. 388x388).

### Visualisation Plots (Full Output Mode)

Only generated when "All plots" is selected:
- `*_darkness.png`: darkness curve with the selected frame marked
- `*_overlay.png`: detected droplet/sphere boundaries

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
- opencv-python, Pillow
- matplotlib
- tqdm
- PyYAML
- customtkinter
- tkinter (included with Python)
- pyphantom (Phantom SDK, from Vision Research)

## License

GPL-3.0. See [LICENSE](LICENSE).

## Citation

Based on:

> Wang, Z. et al. (2022). "Three-dimensional measurement of the droplets out of focus in shadowgraphy systems via deep learning-based image-processing method." *Physics of Fluids*, 34(7), 073301.

If you use this software in your research, please cite:

```bibtex
@article{wang2022three,
  title={Three-dimensional measurement of the droplets out of focus in shadowgraphy systems via deep learning-based image-processing method},
  author={Wang, Zhendong and others},
  journal={Physics of Fluids},
  volume={34},
  number={7},
  pages={073301},
  year={2022},
  publisher={AIP Publishing}
}
```
