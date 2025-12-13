# Droplet Preprocessing Pipeline

Automated preprocessing of high-speed camera footage for droplet analysis. Extracts droplet crops from Phantom .cine files with focus quality assessment.

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
python gui_modular.py
```

### 4. Select Paths in GUI

1. **CINE Root** — Browse to folder containing your .cine subfolders
2. **Output Root** — Browse to where you want outputs saved
3. Configure options (step, mode, etc.)
4. Click **Run Pipeline**

**Alternative (Command Line):**
```bash
python main_runner.py --mode global --step 1 --cine-root "C:\path\to\cines" --output-root "C:\path\to\output"
```

## Project Structure

```
Modular/
├── main_runner.py          # CLI entry point
├── gui_modular.py          # GUI interface
├── config_modular.py       # Default configuration
├── setup_environment.py    # Environment verification
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── pipeline_global.py      # Global calibration pipeline
├── pipeline_folder.py      # Per-folder pipeline
│
├── cine_io_modular.py      # .cine file reading
├── darkness_analysis_modular.py  # Best frame selection
├── geom_analysis_modular.py      # Droplet geometry
├── crop_calibration_modular.py   # Crop size calibration
├── cropping_modular.py     # Image cropping
├── image_utils_modular.py  # Image utilities
│
├── focus_metrics_modular.py  # Focus quality metrics
├── focus_analysis.py       # Standalone focus analysis
│
├── output_writer_modular.py  # CSV/output generation
├── plotting_modular.py     # Visualisation
├── parallel_utils_modular.py # Multiprocessing
├── profiling_modular.py    # Performance profiling
├── timing_utils_modular.py # Timing utilities
├── workers_modular.py      # Parallel worker functions
│
└── OUTPUT/                 # Generated outputs
    ├── {folder}/
    │   ├── *_crop.png      # Droplet crops
    │   ├── *_summary.csv   # Metadata
    │   └── *_overlay.png   # Visualisations
    │
    └── Focus/              # Focus classification results
        ├── sharp_crops.csv
        └── {folder}/       # Sharp images by folder
```

## Pipeline Modes

### Global Mode (Recommended)
- Calibrates crop size across ALL folders
- Consistent crop dimensions for CNN training
- Better for large datasets

### Per-Folder Mode
- Independent processing per folder
- Crop sizes may vary between folders
- Faster for single-folder processing

## Focus Classification

After running the pipeline, classify crops by focus quality:

```bash
python focus_analysis.py
```

This creates:
- `Focus/sharp_crops.csv` — List of sharp crops for CNN training
- `Focus/{folder}/` — Copies of sharp images organised by folder

### Per-Folder Thresholds

Focus classification uses per-folder thresholds to ensure diverse training data:
- Each folder contributes its sharpest ~25% to the training set
- Handles varying optical conditions between sessions
- Produces robust models that generalise to new setups

## Output Files

### Per-Droplet Crops
- `{droplet_id}{cam}_crop.png` — Grayscale droplet crop
- Standard size (e.g., 388×388) across all folders

### Summary CSV
Each folder produces `{folder}_summary.csv`:

| Column | Description |
|--------|-------------|
| droplet_id | Droplet identifier |
| camera | g (green) or v (violet) |
| cine_file | Source .cine filename |
| best_frame | Selected frame index |
| dark_fraction | Darkness metric |
| y_top, y_bottom, y_sphere | Geometry |
| crop_size_px | Crop dimensions |
| crop_path | Path to crop image |
| laplacian_var | Focus metric (edge strength) |
| tenengrad | Focus metric (gradient) |
| brenner | Focus metric (contrast) |

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
- customtkinter
- pyphantom (Phantom SDK)

## Citation

This pipeline implements preprocessing for droplet defocus estimation based on:

> Wang et al. (2022). "Three-dimensional droplet measurement using deep learning"
