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

Even if you know your camera's optical parameters (focal length, aperture, etc.), the theoretical blur formula doesn't perfectly match reality. Calibration determines ρ, which corrects theory to match your actual camera.

**Everyone needs calibration** - whether you know your optical parameters or not.

## Three Calibration Approaches

### Approach A: Direct Empirical
**Best for:** Unknown optical parameters, simplicity

```
σ = ρ × |d|
```
- ρ has units: pixels per mm
- No optical parameters needed
- Simple linear fit

### Approach B: Optical Formula + ρ
**Best for:** Known parameters, accuracy at large defocus

```
σ = ρ × CoC(d, f, N, D, pixel_size)
```
- ρ is dimensionless (typically 0.5-2.0)
- Uses theoretical Circle of Confusion formula
- More accurate far from focus

### Hybrid (Recommended)
**Best for:** Using the Training GUI

1. Calibrate with Approach A (simple lab procedure)
2. Automatically convert to Approach B format
3. Get compatible values for Training GUI

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

## GUI Workflow

### Tab 1: Load Z-Stack
- Browse to folder containing calibration images
- Set z-range (e.g., -12 to +12 mm, 0.5mm steps)
- Preview images and auto-detect focal plane

### Tab 2: Measure Blur
- Choose measurement method:
  - **Sigmoid** (recommended): Fits edge profile
  - **Gradient**: Uses Sobel magnitude
  - **Laplacian**: Simple variance metric
- View σ vs z plot

### Tab 3: Calibrate ρ
- Select approach (A, B, or Hybrid)
- For B/Hybrid: Enter optical parameters
- Run calibration
- View fit quality (R²)

### Tab 4: Multi-Aperture
If your historical data was captured with unknown aperture:
1. Calibrate at multiple aperture settings
2. Load validation images (e.g., from 2021 dataset)
3. Compare which aperture gives plausible depths
4. Select the best match

### Tab 5: Multi-Camera
For sign resolution with 2+ cameras:
1. Add calibration for each camera
2. Record focal plane offsets
3. Test sign resolution algorithm

### Tab 6: Export
- Save YAML config (for Training GUI)
- Save CSV measurements
- Save plots
- Copy ρ to clipboard

## Output Format

```yaml
camera: "g"
aperture_setting: "position_2"
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

## Using Results with Training GUI

1. Open Training GUI
2. In "Scan & Configure" tab, enter:
   - `focal_length_mm`, `f_number`, `focus_distance_mm`, `pixel_size_mm` from calibration
   - `rho` = `formula_rho` value
3. Generate synthetic training data
4. Train model

The synthetic blur will now match your real camera system!

## Module Structure

```
calibration/
├── calibration_gui.py     # Main GUI application
├── blur_measurement.py    # Blur measurement methods
├── calibration_core.py    # Core calibration logic
├── validation.py          # Historical data validation
└── README.md              # This file
```

## Command Line Usage

For scripting:

```python
from calibration_core import calibrate_hybrid, OpticalParams
from blur_measurement import measure_blur_batch

# Load your images and positions
images = [...]  # List of numpy arrays
positions = [-12, -11.5, ..., 12]  # z positions in mm

# Measure blur
_, sigmas, _ = measure_blur_batch(images, positions, method='sigmoid')

# Calibrate
optical = OpticalParams(
    focal_length_mm=50,
    f_number=4,
    focus_distance_mm=300,
    pixel_size_mm=0.01
)
result = calibrate_hybrid(positions, sigmas, optical)

print(f"ρ (direct): {result.direct_result.rho_px_per_mm:.3f} px/mm")
print(f"ρ (formula): {result.formula_result.rho:.3f}")
```

## References

Wang, Z. et al. (2022). "Three-dimensional measurement of the droplets out of focus in shadowgraphy systems via deep learning-based image-processing method." *Physics of Fluids*, 34(7), 073301.
