# Calibration

Given a z-stack of a calibration sphere at known stage positions, fits a
`CalibrationModel` that converts blur σ ↔ defocus mm for a specific camera
and lens. The output `calibration_results.yaml` is baked into Training
checkpoints and read at Inference.

This is the bridge between what the model learns (normalised blur,
unitless) and what the user wants (depth, mm). Every prediction the
pipeline makes — and every uncertainty bar it shows — flows through the
fitted model in this module.

`physics.CalibrationModel` itself lives at the **repo root**, not under
`Calibration/`. Training and Inference both need to instantiate it
without dragging in the calibration GUI; this module fits and exports
it, then the others consume it.

## Quick start

Three equivalent launch commands:

```bash
cropping-calibrate               # console script (pip install -e .)
python -m Calibration            # module form
python Calibration/calibration_gui.py     # F5-friendly direct launch
```

Minimal flow through the four tabs:

1. **Data** — pick an image folder OR a `.cine` folder, set positions
   (CSV or generated range), set the **Sphere diameter (mm)** field
   (this calibrates the camera scale `s_calib_px_per_mm`), click
   **Process Spheres**.
2. **Calibrate** — confirm sphere detection, click **Measure All**, pick
   **Direct Calibration** mode and a fit method
   (`linear` / `quadrature` / `hybrid`), click **Calibrate ρ**.
3. **Multi-Camera** *(optional)* — only if you have two cameras at
   different focal planes and want signed depth.
4. **Export** — Output Location auto-fills with
   `Calibration/runs/<timestamp>_camera-<id>/`. Click **Export
   Calibration**.

The run dir contains `calibration_results.yaml`, `measurements.csv`, two
diagnostic PNGs, and the `processed_images/` that ERF was actually run on.

## The pipeline

```
Z-stack input  (image folder OR .cine folder)
   ↓
Stage positions → defocus z   (re-zeroed around sharpest frame)
   ↓
Sphere processing  (per frame):
   ├── consensus sphere detect — median over the stack
   │       (defeats per-frame mis-detections on out-of-focus frames)
   ├── mirror about sphere centre — kills stage reflection
   ├── blacken interior — median-of-deep-interior pipette fill
   │       (kills lighting hot spots that contaminate edge measurement)
   ├── crop to square at 1.2× sphere radius
   └── optional flatten (default / simple / inference modes)
   ↓
Blur measurement at sphere edge   (ERF default; gradient or Laplacian alts)
   ↓
σ vs z curve   (V-shape, minimum at the focal plane)
   ↓
Pre-fit filter:
   ├── near-focus exclusion (|z| < 0.5 mm dropped — ERF unreliable here)
   └── plateau detection (3-consecutive-point rule on Δσ < 0.3 × median Δσ)
   ↓
Fit:  linear  /  quadrature  /  hybrid   →   CalibrationModel
   ↓
Trust bounds + LOO cross-validation
   ↓
Export:  calibration_results.yaml + measurements.csv + 2 PNGs + processed_images/
```

The two non-obvious steps:

- **Consensus sphere detect.** Per-frame Canny + `minEnclosingCircle`
  fails on the most-defocused frames (the sphere edge is wider than the
  blur kernel can tolerate). Detecting on every frame and taking the
  median `(cx, cy, radius)` defeats this — the sphere doesn't move, only
  focus changes.
- **Pre-fit filter.** Near-focus is dropped because ERF can't
  distinguish σ < ~1 px reliably. Plateau is dropped because at extreme
  defocus the blur exceeds the crop extent, ERF saturates, and Δσ
  collapses to noise. Kept points land in
  `fit_metadata.z_kept_mm` / `sigma_kept_calib_px` so the diagnostic
  PNG can colour-code them.

## GUI tabs

### Tab 1 — Data

| Field | Purpose |
|---|---|
| Source type | `Image Folder` or `.cine Folder`. The header shows `(✓ pyphantom)` or `(⚠ pyphantom)` so you know whether `.cine` is even available. |
| Image Folder | Folder of pre-saved images (one per z-position) |
| Positions CSV | Optional — `filename, stage_position_mm` columns. If absent, generate with the Z min/max/step row. |
| `.cine` Folder | Folder of `.cine` files, one per z-position |
| Stage Range | Fallback when no positions CSV is supplied — assumes evenly spaced positions |
| Frame index | Which frame to extract from each `.cine` (default 0) |
| Camera | `g` / `m` / `v` / `custom` — appears in YAML and run-dir name |
| Sphere diameter (mm) | Physical size of the calibration sphere — used to compute `s_calib_px_per_mm` (the camera scale baked into the model) |
| Upper contour only | Use only the upper portion of the detected contour for circle fitting — avoids the stage reflection on the lower edge |
| Process Spheres / Save Processed | Run the sphere-processing pipeline + optionally save the processed crops |
| Detect Sphere / Verify / Re-zero | Manual aids — re-detect on the current frame; re-zero z around the sharpest frame |

### Tab 2 — Calibrate

| Field | Purpose |
|---|---|
| Auto-detect / X, Y, R | Sphere region for blur measurement — auto by default; manual override available |
| Measure All | Run blur measurement (ERF) on every frame; produces `σ vs z` |
| Calibration Mode | `Optical Formula` (analytic Wang formula) or `Direct Calibration` (fit to data) |
| Fit method (Direct only) | `linear` / `quadrature` / `hybrid` — see "Calibration methods" below |
| Calibrate ρ | Run the fit + LOO + trust-bounds computation; updates the Results panel |
| Results panel | Shows ρ, σ₀ or σ_floor, R², LOO uncertainties, per-side slopes |

### Tab 3 — Multi-Camera

For setups with two cameras at different focal planes. A single camera
measures `|z|` only — the same blur is produced at `+z` and `−z`. With
two cameras at different focal offsets, sign falls out by comparing
which camera sees the sharper image (whichever camera's focal plane is
closer to the particle).

| Field | Purpose |
|---|---|
| Camera Calibrations table | Each row: camera label, ρ, focal offset (mm). First camera added becomes the reference (offset = 0). |
| Add Current Calibration | Pulls ρ from the active calibration on Tab 2 |
| Test Sign Resolution | Enter σ from each camera for the same droplet, click Calculate Signed Depth — returns a signed `z` |

This tab is optional. Single-camera calibrations don't need it.

### Tab 4 — Export

| Field | Purpose |
|---|---|
| YAML config | Write `calibration_results.yaml` (the deliverable) |
| CSV | Write `measurements.csv` (`filename, z_mm, sigma_px` per frame) |
| Plots | Write `calibration_curve.png` + `calibration_report.png` (multi-panel quality report with kept/excluded colour-coding, residuals, LOO summary) |
| Output Location | Auto-fills with `Calibration/runs/<YYYYMMDD_HHMMSS>_camera-<id>/`. Editable. |
| Export Calibration | Writes everything to the output folder + a copy of the processed images |
| Copy ρ to Clipboard | Convenience for pasting into other tools |
| YAML Preview | Right pane — live preview of what the export will write |

## Calibration methods

Three forward models are supported. All produce a unified
`CalibrationModel` instance with the same trust-bounds + LOO machinery.

### Linear (submitted-dissertation method)

```
σ = ρ × |z| + σ₀
```

- Two parameters; simplest; what the dissertation reports.
- `ρ` = blur rate (px/mm), `σ₀` = floor at the focal plane.
- Submitted values: ρ = 1.548 px/mm, σ₀ = 0.125 px, R² = 0.997, n = 50.
- YAML keys (legacy): `direct: {rho_px_per_mm, sigma_0, r_squared}`.
- YAML keys (unified): `calibration_model: {method: linear, rho_px_per_mm, sigma_0_calib_px, …}`.

### Quadrature (post-submission)

```
σ = √((ρ × |z|)² + σ_floor²)
```

- Smooth at z = 0 (no kink); better matches diffraction-limited PSF
  behaviour where σ approaches a finite floor as `|z| → 0`.
- Same number of parameters as linear, different curve shape.
- YAML keys: `calibration_model: {method: quadrature, rho_px_per_mm, sigma_floor_calib_px, …}`.

### Hybrid (post-submission, default for the user's current model)

```
σ = √((ρ × |z|)² + σ_floor²)  +  Δσ(|z|)
```

where `Δσ(|z|)` is a per-`|z|` residual look-up table averaged across ±z.
See "The hybrid residual LUT" below for how it's built and used.

- More accurate at extreme defocus where parametric models miss
  non-linearity. More complex.
- YAML keys: `calibration_model: {method: hybrid, rho_px_per_mm, sigma_floor_calib_px, residual_lut_mm_px: [[0.6, 0.038], [0.8, 0.024], …], …}`.

The submitted dissertation reports linear only. Quadrature and hybrid
were added post-submission and live alongside, never replacing linear —
defend either path in the viva.

## Trust bounds, the residual LUT, and LOO uncertainty propagation

This is the load-bearing section for understanding what the model
actually claims to know.

### Pre-fit filter

Two stages, both applied by `_filter_for_fit` in
[calibration_core.py:1016](calibration_core.py#L1016):

1. **Near-focus exclusion.** Default: drop `|z| < 0.5 mm`. ERF can't
   distinguish σ values below ~1 px reliably; including these would
   bias the fit's estimate of `σ₀` (or `σ_floor`).
2. **Plateau detection.** Walk in from each end of the sorted-by-z
   array. While `Δσ < 0.3 × median(Δσ)`, count consecutive low-delta
   points. Once a count of 3 or more is reached, drop the entire run.
   This catches the saturation tail at extreme defocus where blur
   exceeds the crop extent and ERF readings collapse to noise.

The kept arrays are persisted into `fit_metadata.z_kept_mm` and
`fit_metadata.sigma_kept_calib_px` in the YAML so the diagnostic PNG can
colour-code kept vs excluded points.

### Trust bounds

After filtering, `_bounds_from_fit`
([calibration_core.py:1111](calibration_core.py#L1111)) computes:

| Field | Meaning |
|---|---|
| `sigma_min_trusted_calib_px` | Smallest σ the fit covers — predictions below this flag `BELOW_FLOOR` |
| `sigma_max_trusted_calib_px` | Largest σ the fit covers, **cross-side conservative** (min of the per-side maxes) — predictions above this flag `SATURATED` |
| `sigma_max_trusted_neg_calib_px` | Per-side ceiling on the negative-z side (diagnostic, captures lens asymmetry) |
| `sigma_max_trusted_pos_calib_px` | Same on the positive-z side |
| `z_min_trusted_mm` | Smallest `|z|` the fit covers |
| `z_max_trusted_mm` | Largest `|z|` overall (for diagnostics) |
| `z_max_trusted_neg_mm` / `z_max_trusted_pos_mm` | Per-side range (asymmetric — lens asymmetry shows up here too) |

A sign-blind model uses the cross-side conservative `sigma_max_trusted_calib_px`. If you know the sign at inference (e.g. via the multi-camera tab), you can trust the matching per-side ceiling.

### BoundsFlag at inference

Every prediction is classified by `CalibrationModel.inverse()`:

- **`IN_RANGE`** — σ is within `[sigma_min_trusted, sigma_max_trusted]`. Use the prediction with confidence.
- **`BELOW_FLOOR`** — σ is at or below the resolution limit (or below `σ_floor` for quadrature/hybrid). Returns `(z = 0, BELOW_FLOOR)`.
- **`SATURATED`** — σ is at or above `min(sigma_max_trusted_calib, sigma_max_model_observed)`. Returns the **best-guess `|z|`** from the inverse, not NaN. Caller decides how much to trust it.

The `sigma_max_model_observed_px` is set later (during training) — it's the empirical max σ the trained model produces on the calibration stack. The SATURATED threshold is the min of "what the calibration covers" and "what the model actually outputs", which catches the case where the model's plateau is below the calibration's.

### The hybrid residual LUT — how it's built and used

The hybrid method adds a non-parametric correction on top of the
quadrature fit. Built in two passes
([calibration_core.py:1280](calibration_core.py#L1280)):

1. **Fit quadrature first** to the kept `(z, σ)` points. Predict σ at
   every kept point. Compute the residuals `Δσ_i = σ_actual_i − σ_predicted_i`.
2. **Group points by `|z|`** (rounded to 4 decimals, so `−0.6` and `+0.6`
   collide cleanly). Average the residuals within each `|z|` bin. Sort
   by `|z|`. Store as `[(|z|, Δσ), …]` in YAML.

A real LUT from the user's current hybrid calibration:

```yaml
residual_lut_mm_px:
  - [0.6, +0.0385]
  - [0.8, +0.0243]
  - [1.0, +0.0175]
  - [1.2, +0.0013]
  - [1.4, -0.0035]
  ...
  - [5.4, +0.1796]
  - [5.6, +0.2076]
  - [5.8, +0.2193]
  - [6.0, +0.0175]
  - [6.4, -0.4423]
```

The mid-range residuals are tiny (~10⁻²) — quadrature is doing the
heavy lifting. The endpoint residuals get larger because that's exactly
where parametric models miss: the pre-saturation regime where the
edge profile starts to deviate from the diffraction-limited shape.

**Forward use.** `CalibrationModel.forward(z)` computes
`quadrature(z) + Δσ(|z|)`. `Δσ(|z|)` is linearly interpolated between
LUT entries (clamped to the endpoints for out-of-range queries — see
`_residual_at` at [physics.py:676](../physics.py#L676)).

**Inverse use.** `CalibrationModel.inverse(σ)` runs Newton iteration on
the forward model, seeded with the quadrature inverse as starting point.
The numerical derivative handles the LUT's piecewise-linear discontinuities.
If Newton fails to converge (rare — happens at LUT endpoints where the
derivative is discontinuous), it falls back to the quadrature root.
The first fallback emits a logger warning; subsequent ones are counted
silently in `CalibrationModel._newton_fallback_count` for diagnostics.

Net effect: hybrid captures non-linearity at extreme defocus that the
parametric models miss, **without forcing a higher-order parametric form**
that would over-fit elsewhere.

### LOO cross-validation and uncertainty propagation

The trust bounds tell you whether a prediction is in range. **LOO tells you how much to trust it.**

`loo_cv_for_method(method, z, sigma)`
([calibration_core.py:777](calibration_core.py#L777)) runs once per fit method:

1. For each kept point `i`: refit the chosen method on the remaining
   `n − 1` points using `calibrate_to_model`. Predict σ at the held-out
   `z[i]`. Record the residual.
2. Aggregate across all `n` folds:

| Field | Meaning |
|---|---|
| `rho_std` | Standard deviation of `ρ` across the `n` refits |
| `aux_param_name` | `'sigma_0'` for linear, `'sigma_floor'` for quadrature/hybrid |
| `aux_param_std` | Standard deviation of the auxiliary parameter across refits |
| `loo_mae` | Mean absolute residual over the `n` held-out predictions |
| `num_folds` | `n` |

This dict lands in `CalibrationModel.loo_cv` and gets written to YAML.

**At inference time**, `CalibrationModel.defocus_uncertainty()`
propagates these into per-prediction error bars by perturbing `ρ` and
the auxiliary parameter by their LOO standard deviations and recomputing
`|z|` through the inverse. For linear, the propagation is closed-form
(`dz/dρ = −σ / ρ²`, `dz/dσ₀ = −1/ρ`); for quadrature/hybrid, the
derivatives are computed numerically. The result is the `±mm` shown in
the Inference GUI's result panel.

**Without LOO**, you'd get a point estimate with no honest uncertainty —
the user just has to trust the number. With LOO, every prediction comes
with a defensible error bar derived from the actual fit's stability.

Concrete numbers from the user's current hybrid calibration:

```yaml
loo_cv:
  rho_std: 0.0056          # ρ stable to ~0.5 % across folds
  aux_param_name: sigma_floor
  aux_param_std: 0.021     # σ_floor stable to ~2 %
  loo_mae: 0.089           # mean held-out error: ~0.09 px
  num_folds: 56
```

For a typical mid-range prediction these translate to roughly
`±0.05 mm` of defocus uncertainty.

### Three pixel spaces

The model's σ values live in `calib_px` — the calibration camera's
native pixel scale. Three spaces are tracked through every conversion:

- **`calib_px`** — where σ was measured (calibration camera native).
- **`native_px`** — source image native (could differ from calibration
  if you're inferring on a different camera).
- **`model_px`** — network input (after resizing native → `model_size`).

`CalibrationModel.forward_at` and `inverse_at` are the boundary wrappers
that handle cross-camera and resolution scaling. Callers never multiply
by scale ratios directly.

### Cross-camera scaling

`s_calib_px_per_mm` is baked into the model from the **Sphere diameter
(mm)** field on Tab 1. `s_inf_px_per_mm` is set at inference time from
the user's `s_c` field in inference settings.

For a prediction made by a model trained on calibration-camera data
but evaluated on a different camera:

```
σ_native = σ_calib × (s_inf / s_calib)
```

This makes calibrating on one camera and inferring on another
mathematically clean. If both scales are unset, the math falls back to
`s_inf = s_calib = 1` (i.e. assume same camera).

## Outputs

```
Calibration/runs/20260427_204947_camera-g/
├── calibration_results.yaml     # primary deliverable
├── measurements.csv             # filename, z_mm, sigma_px (one row per frame)
├── calibration_curve.png        # σ-vs-z plot snapshot
├── calibration_report.png       # multi-panel quality report (fit + residuals + LOO)
└── processed_images/            # post-pipeline crops ERF actually ran on
```

### `calibration_results.yaml` — schema

Every export writes both legacy and unified blocks for backward
compatibility. Top-level fields:

| Key | Purpose |
|---|---|
| `camera` | Camera label (`g` / `m` / `v` / custom) |
| `aperture_setting` | User-supplied; `unknown` if not entered |
| `calibration_mode` | `direct` or `optical` |
| `direct: {rho_px_per_mm, sigma_0, r_squared, num_points, scale_calib_px_per_mm, loo_cv}` | **Legacy block** — what older Training/Inference code reads. Always written. |
| `optical_params: {focal_length_mm, f_number, focus_distance_mm, pixel_size_mm}` | Only present in optical mode |
| `formula_rho` | Optical-formula ρ, when computed |
| `focal_plane_offset_mm` | Distance from stage zero to the focal plane |
| `defocus_range_mm: [z_min, z_max]` | Range covered by the calibration |
| `reference_resolution` | Image dimension the σ measurements live in (px) |
| `training_config: {…}` | Ready-to-paste block for Training's config YAML |
| `pixel_size_mm` | Physical pixel size, when known |
| `calibration_model: {…}` | **The unified block** — `physics.CalibrationModel.to_dict()` |

The `calibration_model:` sub-fields:

| Key | Purpose |
|---|---|
| `method` | `linear` / `quadrature` / `hybrid` |
| `rho_px_per_mm` | The slope (px/mm) |
| `sigma_0_calib_px` | Linear only |
| `sigma_floor_calib_px` | Quadrature / hybrid |
| `residual_lut_mm_px: [[|z|, Δσ], …]` | Hybrid only |
| `sigma_min_trusted_calib_px`, `sigma_max_trusted_calib_px` | Trust bounds (cross-side) |
| `sigma_max_trusted_neg_calib_px`, `sigma_max_trusted_pos_calib_px` | Per-side σ ceilings (diagnostic) |
| `sigma_max_model_observed_px` | Set later by Training; null until then |
| `z_min_trusted_mm`, `z_max_trusted_mm` | Trust bounds in z |
| `z_max_trusted_neg_mm`, `z_max_trusted_pos_mm` | Per-side z ranges |
| `s_calib_px_per_mm` | Calibration-camera scale, from sphere-diameter input |
| `loo_cv: {rho_std, aux_param_name, aux_param_std, loo_mae, num_folds}` | Method-aware LOO uncertainties |
| `fit_metadata: {…}` | Provenance: `num_points_total`, `num_points_kept`, exclusion counts, `rho_per_side`, `fit_timestamp`, `z_kept_mm`, `sigma_kept_calib_px`, `r_squared`, `max_abs_residual_px`, `n_lut_points` |

`fit_metadata.z_kept_mm` and `sigma_kept_calib_px` are the actual arrays
the fit ran on — useful for re-plotting and debugging without re-loading
the source z-stack.

## Caveats and gotchas

- **`physics.CalibrationModel` lives at the repo root**, not under
  `Calibration/`. Training and Inference both import it.
  [calibration_core.py:1006](calibration_core.py#L1006) injects the
  repo root into `sys.path` before importing it.
- **Sphere consensus matters.** Per-frame sphere detection misses on
  the most-defocused frames; the consensus (median across stack) circle
  is what gets used for cropping. Don't expect the per-frame detection
  box to look right on the most-blurred images.
- **Plateau exclusion is automatic and silent.** If your trust ceiling
  comes out lower than you expected, check `fit_metadata.num_plateau_excluded`
  in the YAML — extreme-defocus frames that ERF couldn't measure
  reliably get dropped.
- **Near-focus exclusion is a hard `±0.5 mm` by default.** If your
  z-stack is unusually fine-grained or your ERF measurements are
  unusually clean, consider lowering this — but the default is
  conservative for good reason.
- **The submitted dissertation reports linear only** (ρ = 1.548 px/mm,
  σ₀ = 0.125 px, R² = 0.997, n = 50). Quadrature and hybrid live
  alongside, never replacing it. Defend either.
- **Three pixel spaces matter for cross-camera work.** If you skip the
  Sphere diameter input, `s_calib` defaults to nothing and cross-camera
  scaling falls back to `s_inf = s_calib`. Fine if you never change
  cameras; broken if you do.
- **`validation.py` is dead code.** Defined but never called. Slated
  for removal in a cleanup pass.
- **No multiprocessing here** — the calibration GUI is single-process.
  53 frames, not 1500 droplets; doesn't need it.
- **Legacy fit functions still exist.** `calibrate_approach_a`,
  `calibrate_approach_b`, `calibrate_hybrid` predate the unified
  `calibrate_to_model` dispatcher. They're kept for backward
  compatibility with the older YAML schema; new code should use
  `calibrate_to_model`.

## Where `physics.py` lives — the import path

The `CalibrationModel` class lives at `<repo>/physics.py`, not under
this module. Reason: Training and Inference both need to instantiate it
without depending on the calibration GUI. The path is:

```
<repo>/physics.py                              ← the class
   ↑
Calibration/calibration_core.py:1006-1013      ← injects repo root, imports it
Training/calibration_editor.py:155             ← edits a saved model
Training/train.py:1068                         ← bakes it into the checkpoint
Training/inference_real_crops.py:417           ← reads from checkpoint
Inference/inference_engine.py:414-417          ← reads from checkpoint
```

If you ever move `physics.py`, fix the
[`calibration_core.py:1006`](calibration_core.py#L1006) sys.path
injection and the imports in Training and Inference. None of them
import via package syntax — they all do `from physics import
CalibrationModel` after a sys.path adjustment.

## File map

```
Module entry
  __main__.py                  console-script entry, sys.path bootstrap
  calibration_gui.py           4-tab Tk app (Data / Calibrate / Multi-Camera / Export)

Core fit machinery
  calibration_core.py          calibrate_to_model dispatcher, _filter_for_fit,
                                _bounds_from_fit, loo_cv_for_method,
                                YAML export, generate_quality_report
  blur_measurement.py          ERF / gradient / Laplacian σ-at-edge measurement,
                                detect_sphere, get_sphere_mask, measure_blur_auto

Sphere processing
  sphere_processing.py         find_sphere_center, mirror_from_center,
                                blacken_sphere_interior, flatten_sphere_crop,
                                find_consensus_sphere, process_sphere_stack
                                (also imported by Preprocessing, Inference, Training)

I/O
  cine_loader.py               CineLoader, CineFolderLoader,
                                parse_position_from_filename, check_pyphantom

Hardware capture (separate sub-module)
  lab_capture/                 Phantom + ThorLabs + Arduino driving (own README)

Legacy
  validation.py                defined but unused; slated for removal
```

## Where it sits in the pipeline

**Upstream:** Z-stack input — either from
[`lab_capture/`](lab_capture/) (the hardware-capture sub-module) or
hand-collected images with a `positions.csv`.

**Downstream:**

- **Training** ([Training/training_gui.py](../Training/training_gui.py))
  reads `calibration_results.yaml` via the "Load from Calibration"
  button, instantiates `CalibrationModel`, and bakes it into the
  checkpoint at training time. Auto-detects the most recent run from
  `Calibration/runs/` if not pointed elsewhere
  ([Training/train.py:202](../Training/train.py#L202)).
- **Inference** ([Inference/inference_engine.py](../Inference/inference_engine.py))
  reads the `calibration_model:` block from the checkpoint (or YAML
  directly), uses `CalibrationModel.inverse_at()` for σ → mm conversion,
  displays `BoundsFlag` per prediction, and renders the LOO-derived
  uncertainty bar.
- **Preprocessing** + **Inference** + **Training** also import
  [`Calibration/sphere_processing.py`](sphere_processing.py) as a code
  module — `flatten_sphere_crop` (mandatory in Preprocessing's output
  pipeline) and `find_sphere_center` (used in Inference's
  auto-preprocess heuristic).
