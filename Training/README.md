# Training

Trains DefocusNet — a scalar-head CNN that maps a blurred crop to a
normalised blur value σ — on synthetic data generated from a fitted
`CalibrationModel`. The CalibrationModel is **baked into every
checkpoint** so Inference is self-contained.

Two operating modes:

- **Direct calibration** — synthesis uses the `CalibrationModel` from
  `Calibration/runs/<latest>/calibration_results.yaml` (linear,
  quadrature, or hybrid).
- **Optical formula** — synthesis uses the Wang CoC formula from
  focal length / f-number / focus distance / pixel size.

Both modes use the same scalar architecture and the same log-space MSE
loss; only the synthesis function and the inference inversion differ.

## Quick start

Three equivalent launch commands:

```bash
cropping-train                                    # console script
python -m Training                                # module form
python Training/training_gui.py                   # F5-friendly direct
```

CLI training is also supported:

```bash
python Training/train.py --config <yaml> --data-dir <dataset_dir> --output-dir <run_dir> [--resume <checkpoint>]
```

Minimal flow through the 5-tab GUI:

1. **Scan & Configure** — Browse to a sharp-crops directory (output of
   Preprocessing). Click **Load from CSVs** to scan + auto-load
   `sharp_crops.csv`. Click **Load from Calibration** to auto-fill ρ,
   σ₀, scale, and defocus range from the latest `calibration_results.yaml`.
2. **Generate** — Set num samples, blur range, image size; click
   **Generate Dataset**. Output lands at
   `Training/training_output/datasets/<YYYYMMDD_HHMMSS>_<name>/`.
3. **Train** — Pick the dataset (auto-fills with latest), set epochs +
   batch size + learning rate, click **Start Training**. Output lands
   at `Training/training_output/models/<YYYYMMDD_HHMMSS>_<name>/`.
4. **Validation** — Optional: run the trained model on a synthetic test
   set to get scatter plots, error histograms, and worst-case visualisations.
5. **Inference** — Run on real preprocessed crops; per-image predictions
   with `BoundsFlag` and LOO-derived uncertainty bars.

Two best checkpoints are saved per run: `dme_best.pth` (best by
weighted MAE on the synthetic validation set) and `dme_calib_best.pth`
(best by mean absolute gap on the real calibration stack). Inference
defaults to `dme_best.pth`.

## The model: DefocusNet

A small Conv backbone with two residual blocks, a 4×4 adaptive average
pool, and a single fully-connected layer to a sigmoid output:

```
Input: blurred crop  (B, 1, H, W)              default H=W=256
   ↓
Conv 7×7, 32 filters, padding=3
   ↓
LeakyReLU(α=0.2)
   ↓
ResBlock × 2          (Conv → LReLU → Conv + skip, 3×3 kernels)
   ↓
AdaptiveAvgPool(4×4)                            preserves spatial gradient
   ↓
Flatten → Linear(32 × 4 × 4 → 1)
   ↓
Sigmoid                                          output ∈ [0, 1]
```

~39,105 trainable parameters. The output is a normalised blur scalar;
multiplying by `max_blur` (stored in the checkpoint) recovers σ in
pixels.

**Loss** is a single log-space MSE in pixel space:

```
L = MSE( log(σ_pred + ε),  log(σ_target + ε) )

  σ_pred   = pred_norm   × max_blur
  σ_target = target_norm × max_blur
  ε        = log_eps     (default 0.01)
```

Operating in log space penalises **relative** error: a 50% error at
σ=1 px and a 50% error at σ=12 px produce the same loss. Without log
space, the loss would be dominated by the highest-σ samples and the
model would learn poorly at low blur.

`max_blur` is the single source of truth, computed once from the
dataset's `metadata.csv` and persisted into the checkpoint. Never
recomputed downstream.

## Two training modes

| | Direct calibration | Optical formula |
|---|---|---|
| Synthesis source | `CalibrationModel.forward(z)` | Wang CoC formula from optical params |
| Inputs needed | `calibration_results.yaml` from Calibration | focal length, f-number, focus distance, pixel size |
| Inversion at inference | `CalibrationModel.inverse_at()` | inverse Wang formula |
| Calibration baked into checkpoint? | Yes (full `CalibrationModel` block) | No (uses optical params from training_config) |
| Recalibrate without retraining? | Yes — swap the embedded `calibration_model` block | No — retrain with new optical params |
| Supports linear / quadrature / hybrid? | All three | n/a (analytic) |

Direct mode is the path most users will use, and the only one that
supports the Calibration module's quadrature and hybrid fits. Optical
mode is supported as an analytic fallback when calibration data isn't
available.

## Synthetic data generation

`synthetic_blur.py` is the generator. Given a `CalibrationModel` (or
optical params), it produces a dataset of blurred sphere crops with
ground-truth σ labels.

### Sphere appearance sampling

Generated samples don't render against an arbitrary background — they
match the real data they'll be deployed on. `SphereAppearanceStats`
samples the user's real sharp crops to learn the distributions of:

- sphere diameter
- contrast (sphere vs background intensities)
- edge profile shape
- background texture

Each generated sample then draws a sphere matching one of those
distributions before blur is applied.

### Blur synthesis

Gaussian convolution at the synthesis-target σ:

- **Direct mode:** `σ = CalibrationModel.forward(|z|)` for each sampled
  defocus `z`. The full `CalibrationModel` (linear/quadrature/hybrid)
  is used — for hybrid, this includes the residual LUT.
- **Optical mode:** `σ` from the Wang CoC formula at the sampled
  defocus.

Kernel radius factor configurable (default 4× σ). The result is a
blurred crop at the target σ ready to feed the model.

### Diameter binning (stratified)

Samples are split into three diameter tertiles (small / medium /
large), with bin boundaries derived from the real-crop diameters. The
training validation split is stratified across these bins so the model
sees balanced training across droplet sizes. Without stratification the
model overfits the dominant size and fails on the others.

### Two-source mixing

The dataset can blend pure-synthetic crops with a fraction of real
preprocessed crops at matching diameters. Configurable in the
**Synthetic Data Config** dialog on Tab 2. Sharpens the sim-to-real
bridge for users whose synthesis doesn't perfectly match their lens.

### Tertiary real-calibration pool

A small pool (default ~50) of real calibration sphere images with
ERF-derived σ labels, drawn from `Calibration/runs/<latest>/processed_images/`.
Used both as a third sample source in the dataset and as the pool the
**calibration loss anchor** evaluates against during training (next
section).

### Per-sample ERF validation

When enabled (`generation.erf_validation: true`), every generated
sample is auto-validated by re-measuring σ via ERF and checking it
matches the synthesis target within tolerance. Catches bugs in the
synthesis path. Failed samples are logged and excluded.

### Per-camera scale correction

When mixing crops from cameras with different `s_calib_px_per_mm`,
σ values are scaled to a common pixel space. Without this, mixing
crops from a high-resolution camera with crops from a lower-resolution
one would make the model see inconsistent σ for the same defocus.

### Outputs

```
datasets/<ts>_<name>/
├── blur/                      PNG crops the model sees during training
├── sharp/                     original sharp crops (reference, not used by training)
├── blur_map/                  per-pixel blur maps (reference, not used by training)
├── metadata.csv               filename, sigma_px, defocus_mm, diameter_px, source, camera
├── generation_config.yaml     resolved config used for generation
├── dataset_summary.json       counts per source + bin distribution
└── calibration_model.yaml     the CalibrationModel that drove synthesis (direct mode)
```

`metadata.csv` is the contract the dataloader reads. The `source`
column distinguishes `synthetic` / `real_mix` / `real_calibration`
samples; the `camera` column is set for mix samples.

## The training loop

`train.py` runs `Trainer.train()`. End-to-end:

### Data loading

`DMEDataset` reads `metadata.csv`, returns `(blur_img, blur_norm, blur_px)`
per item. Optional in-memory cache (`blur_cache.npy`) for small datasets.
On Windows the DataLoader defaults to `num_workers=0` for stability;
override via the `TRAIN_NUM_WORKERS` environment variable.

### Validation split

Stratified across the three diameter bins by default. Falls back to a
uniform split when `min_blur_px < 0.5` (the bins lose meaning at very
low blur).

### Optimiser and scheduler

| Setting | Options | Default |
|---|---|---|
| Optimiser | `adam`, `adamw`, `sgd` | `adam` |
| Adam β₁/β₂ | configurable | 0.9 / 0.999 |
| Weight decay | configurable | 0.0 |
| LR schedule | `cosine`, `exponential`, `step` | `step` |
| LR decay start epoch | configurable | run-specific |
| Gradient clipping | configurable | off (0.0) |

### Bin-weighted MAE

The validation MAE is computed per blur-bin (4 equal-width bins by σ)
and weighted by Beta-distribution sampling weights (configurable α, β).
"Best" is tracked against this weighted MAE rather than overall MAE so
the model isn't dominated by the lowest-σ bin.

### Calibration evaluation per epoch

If `Calibration/runs/` contains a bundle, the trainer auto-discovers
the latest, runs the model on every frame in `processed_images/`, and
computes the mean absolute gap (px) between model predictions and
ERF-derived ground truth from `measurements.csv`. The frames are cached
on first call to avoid repeated disk reads.

The empirical maximum σ the model produces on this stack is recorded as
`sigma_max_model_observed_px`. This becomes the model's plateau
ceiling — the SECOND trust bound used by the SATURATED flag at
inference, alongside the calibration's `sigma_max_trusted_calib_px`.

### Calibration loss anchor (optional)

When enabled (`calibration_anchor_enabled: true`), an additional term
is added to every training step:

```
L_total = L_main + α × MSE( model(real_calibration_pool), ERF_labels )
```

Pulls the model toward physical truth on real data while still training
mostly on synthetic. Default `α = 0.5` works well; higher values
increase the pull but also the variance in training curves.

### Two best checkpoints

| File | Selection criterion |
|---|---|
| `dme_best.pth` | Best weighted MAE on the synthetic validation set |
| `dme_calib_best.pth` | Best mean absolute calibration gap on real ERF-truth frames |

These are often different. Use `dme_best.pth` as the default for
inference; switch to `dme_calib_best.pth` if calibration-stack
performance matters more than synthetic-val metrics for your use case.

Per-epoch recovery checkpoints (`dme_epoch_<N>.pth`) are saved unless
`save_only_best: true`.

### Resuming

`--resume <path>` loads weights + optimizer state. If the checkpoint's
`training_mode` differs from the current config, the loader logs a
warning and proceeds with the checkpoint's mode.

## Calibration baking and the inference handoff

The checkpoint is self-contained — Inference reads everything it needs
from the `.pth` file alone. The chain:

1. **Calibration module** fits a `CalibrationModel`, exports
   `calibration_results.yaml` to `Calibration/runs/<ts>_camera-<id>/`.
2. **Dataset generation** reads the YAML and writes a copy as
   `calibration_model.yaml` next to `metadata.csv` in the dataset dir.
3. **Trainer** auto-loads the dataset's `calibration_model.yaml` (or
   builds a linear `CalibrationModel` from `rho_direct` + `sigma_0` for
   datasets that predate the Phase-5 emit).
4. **Trainer** runs calibration eval per epoch → records
   `sigma_max_model_observed_px`.
5. **At every checkpoint save** (`train.py:1115`), the Trainer mutates
   a copy of `config['training']` to include:

   ```yaml
   training:
     # ... existing fields
     inversion_method: hybrid                    # or linear / quadrature
     calibration_model:
       method: hybrid
       rho_px_per_mm: 1.0478
       sigma_floor_calib_px: 0.9274
       residual_lut_mm_px: [[0.6, 0.038], …]
       sigma_min_trusted_calib_px: 1.131
       sigma_max_trusted_calib_px: 6.431
       # … all other CalibrationModel fields
       loo_cv:
         rho_std: 0.0056
         aux_param_name: sigma_floor
         aux_param_std: 0.021
         loo_mae: 0.089
         num_folds: 56
     calibration_source_sha256: <12-hex prefix>
     sigma_max_model_observed_px: <empirical plateau>
   ```

6. **Inference** ([Inference/inference_engine.py](../Inference/inference_engine.py))
   reads the checkpoint, finds `config['training']['calibration_model']`,
   instantiates `physics.CalibrationModel.from_dict()`, and uses
   `inverse_at()` for σ → mm + `BoundsFlag` for trust.

The SHA256 prefix lets you verify the checkpoint's calibration matches
the source `calibration_results.yaml` you intended (the checkpoint logs
the prefix at save time).

## GUI tabs

### Tab 1 — Scan & Configure

| Field | Purpose |
|---|---|
| Sharp Crops path | Browse to a Preprocessing run's `Focus/` directory (or any folder containing crops + `sharp_crops.csv`) |
| Camera filter | `all` / `g` / `m` / `v` — restricts the scan to one camera type |
| Load from CSVs | Auto-discovers `sharp_crops.csv` (walks up 3 parent dirs, mtime sort) and populates the folder list with focus stats, scale, native blur σ, diameter ranges per folder |
| Folder list | Per-folder `OpticalConfig` overrides — useful when different folders were captured with different optics |
| Load from Calibration | Auto-fills `rho_direct`, `sigma_0`, `scale_calib_px_per_mm`, `defocus_range_mm` from the latest `Calibration/runs/<ts>_camera-<id>/calibration_results.yaml`. Disabled if no calibration runs exist. |

### Tab 2 — Generate

| Field | Purpose |
|---|---|
| Training mode | `direct` (uses CalibrationModel) or `optical` (uses Wang formula) |
| Num samples | Total samples to generate (typical: 50k–200k) |
| Blur range / image size | Defines the σ range and crop dimension (256×256 default) |
| Droplet diameter range | Min/max diameter (px) for synthesised spheres |
| Synthetic Data Config dialog | Two-source mixing percentages, tertiary real-calibration pool size, ERF validation toggle |
| Output dir | Auto-fills with `datasets/<ts>_<name>/`. Editable. |
| Generate Dataset | Writes the dataset; progress + sample preview shown live |

### Tab 3 — Train

| Field | Purpose |
|---|---|
| Dataset path | Auto-fills with the latest dataset. Browse to override. |
| Epochs / batch size / learning rate | Standard knobs |
| Optimiser / LR schedule | Combo dropdowns — see Training Loop section above |
| Advanced settings dialog | Adam β₁/β₂, weight decay, gradient clipping, log_eps, calibration anchor (enable + α), seed, save_only_best toggle |
| Output dir | Auto-fills with `models/<ts>_<name>/`. Editable. |
| Resume from checkpoint | Optional path to a prior `.pth` |
| Start Training | Runs in a background thread; live log + TensorBoard launch button |

### Tab 4 — Validation

Pick a model checkpoint, optionally pick a separate test dataset (or
use the original training dataset's val split). Outputs:

- predicted-vs-true scatter
- error histograms (overall + per blur bin)
- worst-case visualisations (the N samples with largest absolute error)
- summary stats (MAE, RMSE, R², per-bin breakdown)

Lands in `models/<run>/tests/synthetic/test_<ts>/`.

### Tab 5 — Inference

| Field | Purpose |
|---|---|
| Model checkpoint | Path to a `.pth` (auto-fills with the run's `dme_best.pth`) |
| Real crops folder | Folder of preprocessed PNGs (e.g. a Preprocessing run's `Focus/<material>/<cam>/`) |
| Pre-pipeline | Optional sphere processing via `Calibration.sphere_processing.process_sphere_stack` (consensus detect → mirror → blacken → flatten → crop → resize) |
| Run Inference | Per-image predictions with σ_px, defocus_mm, BoundsFlag, LOO-derived ±mm uncertainty |
| Calibration Editor | Opens the post-hoc editor (next section) |

Outputs: `models/<run>/tests/real_crop/test_<ts>/` containing per-image
results, comparison plots, and a CSV manifest.

## The calibration editor

Edits the `(ρ, σ₀)` baked into a trained checkpoint without retraining.
**Linear-only** — quadrature/hybrid checkpoints can't be edited
post-hoc because the LUT and σ_floor depend on the fit data, not just a
slope/intercept.

Why this exists: in direct mode, the calibration parameters live in
the checkpoint's *config*, not in the *weights*. If you recalibrate
(lens swap, focus shift, lab move), you can patch the checkpoint
without regenerating the synthetic dataset and retraining.

| Mode | What it does |
|---|---|
| Manual values | Type new `(ρ, σ₀)` directly |
| Fit from points | Provide new (z, σ) pairs; the editor fits a linear model and applies it |
| Apply linear correction | Subtract a known slope/intercept correction from the existing values |

The editor writes a **new** checkpoint at
`models/<m>/edits/<edit_name>/dme_best.pth`. It never overwrites the
source — `save_corrected_checkpoint` enforces this. The new checkpoint
records a `calibration_history` entry (timestamp, before/after values,
optional notes) so you can trace the lineage.

To "undo" an edit, delete the edit dir.

## Outputs

```
Training/training_output/
├── datasets/
│   └── <YYYYMMDD_HHMMSS>_<name>/
│       ├── blur/                   PNG crops the model sees
│       ├── sharp/                  original sharp crops (reference)
│       ├── blur_map/               per-pixel blur maps (reference)
│       ├── metadata.csv            filename, sigma_px, defocus_mm, diameter_px, source, camera
│       ├── generation_config.yaml  resolved config used for generation
│       ├── dataset_summary.json    counts per source + bin distribution
│       └── calibration_model.yaml  the CalibrationModel that drove synthesis (direct mode)
└── models/
    └── <YYYYMMDD_HHMMSS>_<name>/
        ├── training_config.yaml    resolved config (incl. embedded calibration block)
        ├── checkpoints/
        │   ├── dme_best.pth                     best by weighted MAE on synthetic val
        │   ├── dme_calib_best.pth               best by mean abs gap on calibration stack
        │   ├── dme_best_session_<ts>.pth        best within the current invocation
        │   └── dme_epoch_<N>.pth                per-epoch (skipped when save_only_best)
        ├── logs/
        │   ├── events.out.tfevents.*            TensorBoard event files
        │   ├── run_metadata.json                status / timing / env / dataset summary / best metrics
        │   ├── training_history.yaml            per-epoch loss + MAE + LR
        │   └── training_curves.png              loss + MAE plots
        ├── tests/
        │   ├── synthetic/<test_<ts>>/
        │   └── real_crop/<test_<ts>>/
        └── edits/
            └── <edit_name>/
                ├── dme_best.pth                 edited copy with patched (ρ, σ₀)
                └── tests/
                    ├── synthetic/<test_<ts>>/
                    └── real_crop/<test_<ts>>/
```

### `training_config.yaml` schema

Fields, organised by section:

| Section | Key fields |
|---|---|
| `blur` | `kernel_radius_factor`, `rho` (legacy) |
| `calibration` | `reference_resolution` |
| `data` | `blur_distribution`, `defocus_range_mm`, `diameter_bins` (with tertile boundaries), `droplet_diameter_range_px`, `image_size_px`, `min_blur_px`, `num_samples`, `blur_range_px` |
| `generation` | `erf_validation`, `save_blur_trace_metadata` |
| `optics` | `focal_length_mm`, `f_number`, `focus_distance_mm`, `pixel_size_mm`, `sensor_height_px`, `sensor_width_px` (used by optical mode) |
| `training` | `batch_size`, `crop_size_px`, `epochs_dme`, `lr`, `rho_direct`, `scale_calib_px_per_mm`, `sigma_0`, `training_mode`, `optimizer`, `lr_schedule`, `weight_decay`, `grad_clip_norm`, `log_eps`, `seed`, `calibration_anchor_enabled`, `calibration_anchor_alpha`, `save_only_best` |

When the trainer saves a checkpoint, it adds these to `training`:

| Key | Meaning |
|---|---|
| `inversion_method` | `linear` / `quadrature` / `hybrid` — what to use for σ → mm at inference |
| `calibration_model` | Full `CalibrationModel.to_dict()` block |
| `calibration_source_sha256` | 12-hex prefix of the CalibrationModel for verification |
| `sigma_max_model_observed_px` | Empirical plateau — the model's max σ output on the calibration stack |

### Checkpoint dict keys

| Key | Type | Purpose |
|---|---|---|
| `epoch` | int | Last completed epoch |
| `global_step` | int | Total optimiser steps |
| `config` | dict | Full resolved config (with embedded calibration block) |
| `max_blur` | float | Single source of truth for σ normalisation |
| `max_coc` | float | Back-compat alias for `max_blur` |
| `log_eps` | float | ε used in the log-space loss |
| `training_mode` | str | `direct` or `optical` |
| `dme_state_dict` | dict | Network weights |
| `optimizer_state_dict` | dict | Optimiser state (for resume) |
| `val_loss`, `val_mae_px` | float | Validation metrics at this checkpoint |
| `calibration_history` | list | Audit trail of any post-hoc edits (only on edited checkpoints) |

## Caveats and gotchas

- **`physics.CalibrationModel` lives at the repo root**, not under
  `Training/`. The trainer imports it via `from physics import
  CalibrationModel` after the `__main__.py` sys.path injection
  (root + Training + Preprocessing + Calibration).
- **Calibration is baked into every direct-mode checkpoint.** Inference
  doesn't read `calibration_results.yaml` — it reads from the
  checkpoint's `config.training.calibration_model` block. The SHA256
  prefix verifies provenance.
- **Two best checkpoints exist for a reason.** `dme_best.pth` minimises
  synthetic-val weighted MAE; `dme_calib_best.pth` minimises real
  calibration-stack gap. They're usually different runs of "best".
  Inference defaults to `dme_best.pth`; point it at `dme_calib_best.pth`
  if calibration-stack performance matters more for your use case.
- **`max_blur` is the single source of truth.** Computed once from
  `metadata.csv` at training start, stored in the checkpoint, never
  recomputed downstream. The `max_coc` key in the checkpoint is just a
  back-compat alias for the same value.
- **`TRAIN_NUM_WORKERS` env var.** On Windows the DataLoader's
  multi-process mode is fragile; default workers is 0. Set
  `TRAIN_NUM_WORKERS=4` (or whatever) before launching to override.
- **Calibration loss anchor changes the loss surface.** When on, the
  trainer is jointly minimising synthetic log-MSE + α·MSE on real ERF
  labels. Higher α pulls harder toward real data but adds variance to
  the training curves. `α = 0.5` is the working default.
- **Edits never overwrite the source checkpoint.**
  `calibration_editor.save_corrected_checkpoint` enforces this — edits
  always write to a new path under `edits/<edit_name>/`. To revert, delete the
  edit dir.
- **Calibration editor is linear-only.** Quadrature/hybrid checkpoints
  can't be patched post-hoc because the LUT and σ_floor depend on the
  fit data, not just a slope/intercept. Workaround: re-fit calibration
  with the new sphere data and retrain.
- **Mode mismatch on resume warns and proceeds.** If you resume an
  optical-mode checkpoint with a direct-mode config (or vice versa),
  the loader logs a warning and uses the checkpoint's mode. Start a
  fresh run if that isn't what you want.
- **CLI inference lives in `inference_real_crops.py`**, not a separate
  `predict.py`. Use the GUI's Tab 5 or
  `python Training/inference_real_crops.py` for headless runs.
- **Auto-discovery uses mtime.** "Latest dataset" / "latest model" /
  "latest calibration bundle" all sort by directory mtime. If you
  manipulate folders externally and want a specific one to be picked up,
  `touch` it.

## File map

```
Module entry
  __main__.py                  console-script entry, sys.path bootstrap
                                (Training, repo root, Preprocessing, Calibration)
  training_gui.py              5-tab Tk app — Scan/Configure / Generate / Train / Validation / Inference

Model + loss
  model.py                     DefocusNet scalar-head architecture
                                (DMESubnet, ResBlock, count_parameters, model_summary)
  losses.py                    DMELoss (log-space MSE in pixel space), compute_psnr, compute_ssim

Training
  train.py                     Trainer class — train()/train_dme()/train_dme_only(),
                                checkpoint baking, calibration eval + anchor, CLI main()
  dataset.py                   DMEDataset (returns blur_img, blur_norm, blur_px),
                                max/min-blur from metadata, stratified split, in-memory cache
  utils.py                     calculate_bin_weights_from_beta()

Synthetic data
  synthetic_blur.py            BlurParams, BlurCalculator, SphereAppearanceStats,
                                SyntheticBlurGenerator (two-source mixing, tertiary real pool,
                                ERF validation, per-camera scale correction)

Real-data inference + testing
  inference_real_crops.py      DiameterMeasurer, RealCropInference, per-material grouping,
                                comparison plots, calibration-aware inversion. Also CLI main().
  test_model.py                TestDataset, ModelTester — synthetic-data evaluation,
                                predicted-vs-true scatter, error histograms, worst-case visualisations

Calibration editor
  calibration_editor.py        apply_linear_correction, read/write_calibration,
                                save_corrected_checkpoint, CalibrationSnapshot,
                                next_edit_dirname, audit-trail history
  calibration_editor_dialog.py CalibrationEditorDialog (Tk modal wrapping the editor)

Path utilities
  run_paths.py                 Timestamped dataset/model/test dir helpers,
                                parse_true_z_from_filename, detect_model_name, detect_variant
```

## Where it sits in the pipeline

**Upstream:**

- **Preprocessing** —
  `Preprocessing/output/runs/<latest>/Focus/sharp_crops.csv` provides
  the real sharp crops that seed `SphereAppearanceStats`, drive the
  optional two-source mix, and supply per-camera scale + diameter info.
  Auto-discovered by Tab 1's "Load from CSVs".
- **Calibration** — `Calibration/runs/<latest>/calibration_results.yaml`
  provides the `CalibrationModel`. Auto-discovered by Tab 1's
  "Load from Calibration" and by the trainer's
  `_find_latest_calibration_bundle()` (used for per-epoch calibration
  eval and the optional loss anchor).

**Downstream:**

- **Inference** ([Inference/inference_engine.py](../Inference/inference_engine.py))
  consumes the checkpoint `.pth` only. Does not read `metadata.csv`,
  `training_config.yaml`, or any other Training output. Imports
  `Training.model.DefocusNet` (the class) and reads everything else
  from the checkpoint dict — including the embedded `CalibrationModel`,
  `max_blur`, `log_eps`, and `sigma_max_model_observed_px`.

No other modules in this repo import from Training.
