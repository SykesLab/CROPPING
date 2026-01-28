# Calibration Module

This module calibrates the blur-to-depth conversion constant ρ (rho) for your camera system.

## Installation

```bash
pip install numpy opencv-python scipy pandas pyyaml matplotlib pillow
```

## Quick Start

```bash
cd coursework/calibration
python calibration_gui.py
```

## Why Calibrate?

Your deep learning model learns to estimate blur (σ) from images. Calibration provides the mapping from blur to physical depth:

```
depth_mm = σ / ρ
```

Where ρ is your camera's blur-to-depth constant (pixels per mm).

## Workflow Overview

1. **Training**: Generate synthetic data with arbitrary optical parameters - model learns blur patterns
2. **Calibration**: Measure real σ vs depth for YOUR camera in the lab
3. **Inference**: Model predicts σ → use calibrated ρ to convert to physical depth (mm)

## Lab Procedure

### Equipment Needed
- Linear translation stage (0.1mm precision)
- Calibration sphere (1-3mm diameter)
- Your camera system

### Steps
1. Mount sphere on translation stage
2. Find focal plane (sharpest image), mark as z=0
3. Move stage in 0.5mm increments from -12mm to +12mm
4. Capture 5-10 frames at each position
5. Load images into GUI
6. Run calibration

## GUI Tabs

### Tab 1: Data
- Browse to folder containing calibration images
- Set z-range (e.g., -12 to +12 mm, 0.5mm steps)
- Preview images with slider
- Auto-detect focal plane (sharpest image)
- Auto-crop to sphere region

### Tab 2: Calibrate
Three-column layout: Measure | Fit | Results

**Step 1 - Measure Blur:**
- Choose method: Sigmoid (recommended), Gradient, or Laplacian
- Auto-detect sphere or enter manual coordinates
- Click "Measure All" to process z-stack

**Step 2 - Fit ρ:**
- Select approach: Hybrid (recommended), Direct, or Optical Formula
- Enter optical parameters if needed
- Click "Calibrate ρ" to fit the blur-depth relationship

**Results:**
- View σ vs z plot (V-shape expected)
- See fit quality (R²)
- Get calibrated ρ value

### Tab 3: Multi-Camera
For sign resolution with 2+ cameras (e.g., cameras g, m, v):

- Add calibration for each camera with focal plane offset
- First camera becomes reference (offset = 0)
- Test sign resolution: enter σ from two cameras, calculate signed depth

**How sign resolution works:**
- Single camera only gives |depth| (magnitude)
- Two cameras at different focal planes can determine if droplet is in front (+) or behind (-) the reference plane
- Compare which camera sees sharper image to determine sign

### Tab 4: Export
- Save YAML config (for Training GUI)
- Save CSV measurements
- Save calibration plots
- Copy ρ to clipboard

## Output Format

```yaml
camera: "g"
approach: "hybrid"

direct:
  rho_px_per_mm: 3.6
  sigma_0: 0.8
  r_squared: 0.994

optical_params:
  focal_length_mm: 50.0
  f_number: 4.0
  focus_distance_mm: 300.0
  pixel_size_mm: 0.01
formula_rho: 5.3

focal_plane_offset_mm: 0.0
```

## Using Results

After calibration, use ρ to convert model predictions to depth:

```python
# Model predicts blur sigma
sigma_predicted = model.predict(image)

# Convert to depth using calibrated rho
depth_mm = sigma_predicted / rho_px_per_mm
```

For multi-camera signed depth, compare blur from both cameras and use focal offsets.

## Module Structure

```
calibration/
├── calibration_gui.py     # Main GUI application
├── blur_measurement.py    # Blur measurement methods
├── calibration_core.py    # Core calibration logic
├── validation.py          # Validation utilities
└── README.md              # This file
```

## References

Wang, Z. et al. (2022). "Three-dimensional measurement of the droplets out of focus in shadowgraphy systems via deep learning-based image-processing method." *Physics of Fluids*, 34(7), 073301.
