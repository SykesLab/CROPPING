# Droplet Preprocessing Pipeline

[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Extracts droplet crops from Phantom .cine files for CNN training. Handles frame selection, cropping, and focus-based filtering.

## How It Works

Each .cine file contains a high-speed recording of a droplet falling onto a sphere. The pipeline finds the best frame to crop from, where "best" means the droplet is fully visible, hasn't yet collided with the sphere, and is roughly centred in the frame.

Frame selection uses connected component analysis to locate the droplet and sphere in each frame. It scores frames by how symmetric the gaps are (droplet-to-image-top vs droplet-to-sphere), with a small weight on darkness to prefer frames where the droplet is more opaque. Frames where the droplet has already touched the sphere are rejected. When darkness analysis is disabled, an early-stop optimisation detects collision and stops scanning (~4x speedup).

Crops are sized to fit the droplet while excluding the sphere. If the crop would otherwise include the sphere, it shifts upward. In Global mode, a calibration pass computes the maximum safe crop height for each sample, then takes a low percentile (default 5th) as the uniform crop size. Set `calibration.percentile` to 0 for guaranteed sphere exclusion if detection is reliable. See [Calibration and Crop Sizing](#calibration-and-crop-sizing) for tuning.

Focus quality is measured using six edge-based metrics (Laplacian variance, Tenengrad, Brenner, etc). Classification into sharp/medium/blurry uses per-folder, per-camera thresholds at the 75th and 25th percentiles, so each camera within each folder contributes its sharpest ~25% regardless of lighting conditions or optical setup.

The GUI shows live thumbnails, progress, and ETA. Processing runs in parallel across all CPU cores, with options for quick validation runs, single-process mode for debugging, and step sampling to process every Nth droplet.

## Quick Start

### 1. Install Python

**IMPORTANT:** Python version requirements depend on your Phantom SDK version. Check your SDK documentation.

Example: Many SDK versions require Python 3.11.x

Download from [python.org](https://www.python.org/downloads/)

During installation, make sure to check "Add Python to PATH".

### 2. Install Phantom SDK (pyphantom)

The Phantom SDK (pyphantom) is required to read .cine files. It's not publicly available - obtain from Vision Research or your institution.

**Installation varies by SDK version.** Typically:
1. Install the pyphantom wheel from your SDK's Python folder
2. Add the SDK's DLL directory to your system PATH (Windows: `PhantomSDK\Bin\Win64`)
3. Install Visual C++ Redistributable if needed (usually in SDK's Bin folder)

```bash
# Example installation (paths vary by SDK version)
pip install "path/to/PhantomSDK/Python/pyphantom*.whl"

# Verify
python -c "import pyphantom; print('OK')"
```

**Windows PATH setup example:**
```powershell
[Environment]::SetEnvironmentVariable("Path", [Environment]::GetEnvironmentVariable("Path", [EnvironmentVariableTarget]::User) + ";C:\path\to\PhantomSDK\Bin\Win64", [EnvironmentVariableTarget]::User)
```

After setup, restart your terminal/IDE.

### 3. Set Up Environment

```bash
# Create virtual environment (recommended)
python -m venv droplet_env
droplet_env\Scripts\activate  # Windows
# source droplet_env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 4. Run Pipeline

```bash
python gui.py
```

### 5. Configure in GUI

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
├── focus_classification.py  # Focus classification for pipeline
├── focus_analysis.py        # Standalone focus analysis CLI
│
├── output_writer.py         # CSV/output generation
├── plotting.py              # Visualisation
├── parallel_utils.py        # Multiprocessing
├── profiling.py             # Timing and profiling utilities
├── workers.py               # Parallel worker functions
│
└── OUTPUT/                  # Generated outputs (default)
    ├── {material}/
    │   ├── {material}_summary.csv
    │   ├── g/               # Green camera outputs
    │   │   ├── crops/
    │   │   └── visualizations/
    │   ├── v/               # Violet camera outputs
    │   └── m/               # Mono camera outputs (if present)
    │
    └── Focus/               # Focus classification results
        ├── sharp_crops.csv
        └── {material}/{camera}/  # Sharp images by material and camera
```

## Pipeline Modes

**Global mode** calibrates crop size across all folders, so every crop has the same dimensions. Only available when you select a parent folder with multiple subfolders.

**Per-folder mode** processes each folder independently. Crop sizes may vary between folders. Automatically selected when processing a single folder.

## Focus Classification

Focus classification runs automatically after processing. It computes focus metrics for each crop, classifies them as sharp/medium/blurry, and copies the sharp ones to `OUTPUT/Focus/{material}/{camera}/`.

### Per-Folder, Per-Camera Thresholds

Each camera within each folder is classified independently:
- Each camera (g, v, m) contributes its sharpest ~25% to training
- Different cameras have different optical properties (wavelength, depth of field)
- Accounts for different lighting/focus between sessions and camera types
- Ensures balanced training data from each camera type for multi-view CNN training

### Standalone Analysis

To rerun focus analysis on existing crops without reprocessing:
```bash
python focus_analysis.py path/to/OUTPUT
```

## Calibration and Crop Sizing

### How It Works

In Global mode, the pipeline computes a single crop size across all samples so the CNN receives uniform input dimensions. For each droplet:

1. **Measure geometry**: diameter (droplet height), gap (space between droplet bottom and sphere top), top margin (space from image top to droplet top)
2. **Compute allowed height**: `diameter + 2 × max(0, gap - safety_pixels)`
3. **Take percentile**: Use the Nth percentile across all samples as the global crop size

### Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `calibration.percentile` | 5.0 | Which percentile of allowed heights to use. Lower = smaller crops, safer. |
| `crop.safety_pixels` | 3 | Extra margin subtracted from the gap before computing allowed height. |

### Tuning Guide

**If crops include the sphere** (sphere visible at bottom of crop):
- Lower `calibration.percentile` (e.g., 2.0 or 1.0)
- Or increase `safety_pixels` (e.g., 5 or 10)

**If crops are too small** (excessive padding around droplet):
- Raise `calibration.percentile` (e.g., 10.0)
- Or decrease `safety_pixels`

**For guaranteed sphere exclusion** (recommended if geometry detection is reliable):
- Set `calibration.percentile: 0` to use the minimum allowed height across all samples
- This ensures no crop ever includes the sphere, but produces smaller crops
- Only use if you're confident the detection has no outliers causing falsely small gaps

### The Trade-off

The percentile approach handles noisy/outlier data:
- **5th percentile**: ~95% of crops guaranteed sphere-free, ~5% may clip the sphere slightly
- **0th percentile (minimum)**: 100% sphere-free, but a single outlier with a tiny gap makes all crops small

If your geometry detection is solid (consistent frame selection, no false detections), using `percentile: 0` is safe and guarantees clean crops.

## Output Files

### Directory Structure
```
OUTPUT/
├── {material}/
│   ├── {material}_summary.csv              # Metadata for all crops in this material
│   ├── g/                                  # Green camera outputs
│   │   ├── crops/
│   │   │   └── sphere0843g_crop.png        # Grayscale droplet crop
│   │   └── visualizations/
│   │       ├── sphere0843g_darkness.png    # Darkness curve plot (full output mode)
│   │       └── sphere0843g_overlay.png     # Geometric overlay (full output mode)
│   ├── v/                                  # Violet camera outputs
│   │   ├── crops/
│   │   └── visualizations/
│   └── m/                                  # Mono camera outputs (if present)
│       ├── crops/
│       └── visualizations/
│
├── Focus/                                  # Focus classification results
│   ├── sharp_crops.csv                     # All sharp crops with metrics
│   ├── focus_classified_all.csv            # All crops with classifications
│   ├── focus_folder_stats.csv              # Per-folder+camera statistics
│   ├── focus_classification_summary.png    # Distribution visualisation
│   └── {material}/
│       ├── g/                              # Sharp crops from green camera
│       ├── v/                              # Sharp crops from violet camera
│       └── m/                              # Sharp crops from mono camera
│
└── focus_metrics_computed.csv              # Combined metrics from all folders
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
| camera | g (green), v (violet), or m (mono/main) |
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

All parameters are in `preprocessing_config.yaml`. GUI overrides path settings at runtime.

### Crop Sizing
| Parameter | Default | Description |
|-----------|---------|-------------|
| `crop.max_cnn_size` | 512 | Maximum crop size (pixels) |
| `crop.min_cnn_size` | 64 | Minimum crop size (pixels) |
| `crop.safety_pixels` | 3 | Margin subtracted from gap. Increase if sphere appears in crops. |

### Calibration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `calibration.percentile` | 5.0 | Percentile of allowed heights. Set to 0 for guaranteed sphere exclusion. |

### Sampling
| Parameter | Default | Description |
|-----------|---------|-------------|
| `sampling.cine_step` | 10 | Process every Nth droplet (1 = all) |

### Best Frame Selection
| Parameter | Default | Description |
|-----------|---------|-------------|
| `best_frame.n_candidates` | 20 | Candidate frames for geometry analysis (full output mode only) |
| `best_frame.darkness_threshold_percentile` | 70.0 | Darkness percentile for candidate filtering |
| `best_frame.darkness_weight` | 0.05 | Weight of darkness vs centring in scoring |

### Geometry Detection
| Parameter | Default | Description |
|-----------|---------|-------------|
| `geometry.min_area` | 50 | Minimum pixels for connected component detection |
| `geometry.sphere_width_ratio` | 0.30 | Sphere must span this fraction of frame width |
| `geometry.sphere_center_tolerance` | 0.35 | Sphere center tolerance from image center |

### Focus Classification
| Parameter | Default | Description |
|-----------|---------|-------------|
| `focus.enabled` | true | Enable/disable focus metric computation |
| `focus.primary_metric` | laplacian_var | Primary metric for sharp/blur classification |
| `focus.sharp_threshold` | null | Custom threshold (null = 75th percentile per folder+camera) |
| `focus.blur_threshold` | null | Custom threshold (null = 25th percentile per folder+camera) |

## Troubleshooting

### "pyphantom not found"
Install the Phantom SDK from Vision Research. See installation instructions above.

### "DLL load failed" or "The specified module could not be found"
1. Ensure the Phantom SDK DLLs are in your system PATH (see installation steps above)
2. Install Visual C++ Redistributable (usually located in SDK's Bin folder)
3. Restart your terminal/IDE after making PATH changes

### "numpy.core.multiarray failed to import" or "_ARRAY_API not found"
Check your SDK's NumPy compatibility. Some SDK versions require NumPy 1.x:
```bash
pip install "numpy<2"
```

### Wrong Python version
Check your SDK's Python version requirements. Common requirement: Python 3.11.x
```bash
python --version
```

### "No .cine files found"
Check your CINE Root path in the GUI points to folders containing .cine files.

### Out of memory
Increase the Step value (processes every Nth droplet) or process fewer folders.

### GUI not responding
The pipeline runs in a background thread. Check the console for progress.

## Dependencies

**Note:** Python and NumPy version requirements depend on your Phantom SDK version.

- Python 3.x (check your SDK requirements - commonly 3.11.x)
- numpy (version depends on SDK - commonly requires <2.0)
- pandas, scipy
- opencv-python, Pillow
- matplotlib
- tqdm
- PyYAML
- tkinter (included with Python)
- **pyphantom** (Phantom SDK, from Vision Research)

See `requirements.txt` for core project dependencies.

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
