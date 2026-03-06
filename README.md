# Calibration — Blur-to-Depth Mapping

This module characterises the relationship between image blur and physical depth for a specific camera setup. The resulting calibration file is used by both the training pipeline (to generate physically accurate synthetic data) and the inference pipeline (to convert model predictions to millimetres).

---

## Position in Pipeline

```
Lab Z-Stack Images
  (lab_capture/)
        ↓
  [ Calibration ]
        ↓
calibration_results.yaml
        ↓
 Training (Direct Mode)      Inference (Direct Mode)
```

Calibration is required before Direct Mode training and inference. It is optional for Optical Formula Mode, where the conversion is derived analytically from known camera parameters.

---

## The Calibration Problem

The neural network outputs a normalised blur value. To convert this into a physically meaningful depth measurement in millimetres, a mapping function must be known for the specific camera and lens in use.

### Optical Formula Approach

In theory, the relationship between defocus distance and blur radius follows from the thin-lens equation:

```
CoC (px) = ρ × A × |d_i − d_o'| / d_o'
```

where `A` is the aperture diameter, `d_i` is the imaging distance, `d_o'` is the object distance, and `ρ` is a per-system scaling constant. If the camera's optical parameters are precisely known, this formula gives the blur at any depth.

In practice, small errors in focal length, focus distance, or pixel size compound into significant depth measurement errors. This motivated the development of the direct calibration approach.

### Direct Calibration Approach

Rather than relying on the optical formula, the direct approach measures the actual blur at known physical depths in the lab:

```
σ (px) = ρ_direct × |z| + σ₀
```

where:
- `z` is the physical displacement from the focal plane (mm)
- `ρ_direct` is the blur rate (pixels per millimetre), fitted from measured data
- `σ₀` is the residual blur at the focal plane (pixels), accounting for diffraction and sensor noise

This linear model is fitted by least-squares regression to a set of (z, σ) measurements taken across a range of known depths. The fit quality is assessed using R². A value above 0.99 indicates a reliable calibration.

The direct approach was introduced to resolve a sim-to-real gap observed during early experiments, where the optical-mode model predicted defocus values significantly different from ground truth.

---

## Inputs

### Direct Calibration Mode

| Input | Location | Description |
|-------|----------|-------------|
| Z-stack images | `lab_capture/captures/<timestamp>/` | Images of a calibration sphere at known depths, captured with `lab_capture/capture_calibration.py` |
| `positions.csv` | Same folder | Stage position (mm) for each image, produced automatically by the capture script |

### Optical Formula Mode

| Input | Description |
|-------|-------------|
| Focal length (mm) | Camera lens focal length |
| F-number | Aperture f-stop |
| Focus distance (mm) | Distance to the in-focus reference plane |
| Pixel size (mm) | Physical size of one sensor pixel |

These parameters are entered manually in the GUI.

---

## Outputs

```
calibration/calibration_output/calibration_results.yaml
```

```yaml
camera: "g"
approach: "hybrid"

direct:
  rho_px_per_mm: 0.699    # Blur rate: pixels of blur per mm of defocus
  sigma_0: 0.28            # Residual blur at the focal plane (pixels)
  r_squared: 0.99          # Fit quality

optical_params:
  focal_length_mm: 200.0
  f_number: 4.0
  focus_distance_mm: 400.0
  pixel_size_mm: 0.020

focal_plane_offset_mm: 0.0
```

The `direct` section is used by Direct Calibration Mode training and inference. The `optical_params` section is produced when using the Hybrid approach and serves as a reference.

---

## How It Works

### Step 1 — Blur Measurement

For each image in the z-stack, the blur of the sphere edge is measured in pixels (σ). The result across all positions forms a V-shape, with the minimum at the focal plane and increasing blur on either side.

Three measurement methods are available:

| Method | How It Works | When to Use |
|--------|-------------|-------------|
| **Sigmoid** (recommended) | Fits a sigmoid function to the intensity profile across the sphere edge. The transition width corresponds directly to σ. | Most cases — most robust to noise and lighting variation |
| **Gradient** | Measures blur from the gradient magnitude along the sphere boundary. | Faster; lower accuracy at high blur levels |
| **Laplacian** | Uses the Laplacian variance of the image region. | Simple; sensitive to texture and noise |

### Step 2 — Focal Plane Detection

The image with the smallest measured σ is identified as the focal plane and assigned `z = 0`. All other positions are expressed as displacements from this reference. This step can be performed automatically (GUI auto-detect) or manually by inspection.

### Step 3 — Model Fitting

A linear model `σ = ρ × |z| + σ₀` is fitted to the (z, σ) measurements. The fitted parameters and R² value are displayed in the GUI.

Three fitting approaches are available:

| Approach | Description | When to Use |
|----------|-------------|-------------|
| **Hybrid** (recommended) | Runs both Direct and Optical Formula approaches, exports both sets of parameters | Default; provides complete output for both training modes |
| **Direct** | Fits the linear empirical model only | When optical parameters are not known |
| **Optical Formula** | Fits using the thin-lens equation with user-provided parameters | When optical parameters are trusted |

---

## How to Run

Launch the calibration GUI:

```bash
cd calibration
python calibration_gui.py
```

### Tab 1 — Data

1. Click **Browse** and navigate to the z-stack image folder (`lab_capture/captures/<timestamp>/`)
2. Set the z-range: start position, end position, and step size (mm)
3. Use the image preview slider to inspect the z-stack and confirm images are correctly ordered
4. Click **Auto-detect focal plane** to identify the sharpest image
5. Click **Auto-crop** to isolate the sphere region, or manually enter crop coordinates

### Tab 2 — Calibrate

1. Select the blur measurement method: **Sigmoid** (recommended for most cases)
2. Confirm the sphere detection region shown in the preview
3. Click **Measure All** to process every image in the z-stack
4. Review the σ vs z plot — a symmetric V-shape indicates consistent measurements
5. Select the fitting approach: **Hybrid** (recommended)
6. Click **Calibrate ρ** to fit the model
7. Review R² in the Results panel. Values below 0.99 should be investigated before proceeding.

### Tab 3 — Multi-Camera (Optional)

For experiments using two cameras at different focal planes, this tab enables **sign resolution** — determining whether a particle is in front of (+z) or behind (−z) the reference plane, rather than only its absolute distance.

A single camera measures `|z|` only. Two cameras at different focal planes determine sign by comparing which camera sees the sharper image: the camera whose focal plane is closer to the particle will always produce a sharper image.

### Tab 4 — Export

1. Click **Save YAML** to write `calibration_output/calibration_results.yaml`
2. Optionally save the σ vs z CSV and calibration plots for records and the dissertation

---

## Caveats

### Calibration Must Precede Direct Mode Training

The `calibration_results.yaml` file must exist before Direct Mode training can be started. The training GUI validates this on startup and will not proceed without a loaded calibration file.

### Direct Mode Enables Recalibration Without Retraining

The trained neural network learns blur patterns, not the specific conversion constants `ρ` and `σ₀`. If the camera setup changes (different focus distance, repositioned lens), the calibration YAML can be updated with new measurements and the same trained model used for inference. Retraining is only required if the blur pattern itself changes significantly — for example, with a different lens or sensor.

### Calibration Sphere Size

A small calibration sphere (1–3 mm diameter) gives cleaner edge measurements than larger spheres. The sphere should be similar in material and size to the particles used in the actual experiment, to ensure that the blur model is representative of the real data.

### Valid Depth Range

The linear blur-depth model is an approximation that is valid close to the focal plane. At very large defocus distances, non-linear optical effects may become significant. The calibration should be performed over the same depth range expected in the actual experiment.

---

## Module Structure

```
calibration/
├── calibration_gui.py      ← Main GUI application
├── calibration_core.py     ← Core fitting logic (Approaches A, B, Hybrid)
├── blur_measurement.py     ← Blur measurement algorithms
├── validation.py           ← Validation utilities
├── cine_loader.py          ← Optional: load images directly from .cine files
├── calibration_output/
│   └── calibration_results.yaml   ← Output consumed by training and inference
└── README.md               ← This file
```

---

## Reference

Wang, Z. et al. (2022). "Three-dimensional measurement of the droplets out of focus in shadowgraphy systems via deep learning-based image-processing method." *Physics of Fluids*, 34(7), 073301.
https://doi.org/10.1063/5.0090714
