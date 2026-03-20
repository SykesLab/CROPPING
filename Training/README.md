# Training — DefocusNet Neural Network Pipeline

This module trains DefocusNet, generates the synthetic training dataset, evaluates model performance, and runs inference on real images. It is the central computational stage of the pipeline.

**Based on:** Wang, Z. et al. (2022). "Three-dimensional measurement of the droplets out of focus in shadowgraphy systems via deep learning-based image-processing method." *Physics of Fluids*, 34(7), 073301.

---

## Position in Pipeline

```
Sharp Sphere Crops           Calibration Results
(from Preprocessing)    +   (Direct Mode only)
          │                        │
          └──────────┬─────────────┘
                     ↓
          [ Data Generation ]
                     ↓
         Synthetic Training Dataset
         (blur / sharp / CoC map)
                     ↓
          [ Model Training ]
                     ↓
         Model Checkpoint (.pth)
                     ↓
            [ Inference ]
                     ↓
        defocus_mm per image
```

---

## Architecture

DefocusNet consists of two sequential subnetworks that are trained in separate stages.

```
Blurred Image ──► DME-subnet ──► CoC Map ──┬──► DD-subnet ──► Deblurred Image
                                            │
                                            └──► Defocus Distance (mm)
```

### DME-subnet — Defocus Map Estimator

A U-Net encoder-decoder network that receives a blurred sphere image and outputs a per-pixel map of the Circle of Confusion (CoC). The CoC value at each pixel indicates the local blur radius in pixels.

- **Input:** single-channel blurred image
- **Output:** normalised CoC map in the range [−1, 1]
- **Architecture:** encoder-decoder with residual blocks and LeakyReLU(0.2) activation

### DD-subnet — Defocus Deblurring

A deblurring network that takes the blurred image concatenated with the CoC map produced by the DME-subnet, and reconstructs a sharp image. The CoC map guides the deblurring — regions with high CoC receive more aggressive correction.

- **Input:** blurred image + CoC map (2 channels combined)
- **Output:** reconstructed sharp image
- **Constraint:** DME weights are frozen during DD training

### Two-Stage Training Principle

**Stage 1 — DME training:** The DME-subnet is trained alone to minimise CoC prediction error.

**Stage 2 — DD training:** The DME weights are frozen. The DD-subnet is then trained to produce sharp output conditioned on the frozen DME's predictions.

This sequencing is essential: the DD-subnet learns to deblur using the specific CoC predictions the DME produces. If the DME is changed after DD training, the deblurring quality degrades because DD was never trained to handle the new DME's prediction style.

**Practical implication:** Train DME to convergence before starting DD. A well-trained DME is the single most important factor for good end-to-end performance.

---

## Dual-Mode Architecture

The training mode determines how synthetic blur is generated and how inference converts model output to depth.

### Optical Formula Mode

Synthetic blur is computed from the thin-lens equation using known camera parameters (focal length, f-number, focus distance, pixel size). At inference, the inverse formula converts predicted CoC (pixels) to defocus depth (mm).

**Use when:** Optical parameters are accurately known and trusted.

### Direct Calibration Mode

Synthetic blur is computed from the empirical linear model fitted in the Calibration module:

```
σ (px) = ρ_direct × |z| + σ₀
```

The calibration parameters `ρ_direct` and `σ₀` are loaded from `calibration_results.yaml`. At inference, the inverse relationship converts predicted blur (pixels) to defocus depth (mm).

**Use when:** Optical parameters are uncertain, or when maximum accuracy for a specific camera is required.

The mode is selected in Tab 1 of the GUI, stored in every checkpoint, and detected automatically at inference time. Checkpoints from different modes are not interchangeable.

---

## Inputs

| Input | Location | Required For |
|-------|----------|-------------|
| Sharp sphere crops | `Preprocessing/CROPPING/Preprocessing/OUTPUT*/` | Both modes |
| `optical_config.yaml` | `training_output/` | Both modes (written by Tab 1) |
| `calibration_results.yaml` | `calibration/calibration_output/` | Direct Mode only |

---

## Outputs

| Output | Location | Description |
|--------|----------|-------------|
| Synthetic dataset | `training_output/synthetic_data/` | Blur/sharp/CoC map triples for training |
| Model checkpoints | `training_output/checkpoints/` | Saved model weights (`.pth` files) |
| TensorBoard logs | `training_output/checkpoints/logs/` | Training metrics for monitoring |
| Inference results | `training_output/inference_results/` | Per-image defocus estimates and visualisations |
| Test results | `training_output/test_results/` | Validation metrics and plots |

### Synthetic Dataset Structure

```
training_output/synthetic_data/
├── blur/           ← Synthetically blurred sphere images
├── sharp/          ← Original sharp sphere images (reference)
├── coc_map/        ← Ground truth CoC maps (normalised 0–1)
└── metadata.csv    ← Per-sample: filename, CoC (px), defocus (mm)
```

### Checkpoint Contents

Each checkpoint file stores:
- Model weights (DME alone, or DME + DD for full pipeline)
- Optimiser state and current learning rate
- Training epoch and best validation loss
- Full training configuration
- `training_mode` — `"optical"` or `"direct"`, used to route inference correctly
- `max_coc` — maximum CoC value used for output normalisation

---

## How to Run

### GUI (Recommended)

```bash
cd training/Training
python training_gui.py
```

Work through the five tabs in order. Tabs 1–3 are for training; Tabs 4–5 are for evaluation and inference.

### CLI — Inference Only

```bash
# Optical mode
python training/Training/predict.py \
    --checkpoint training_output/checkpoints/best_checkpoint.pth \
    --image path/to/image.png

# Direct mode
python training/Training/predict.py \
    --checkpoint training_output/checkpoints/best_checkpoint.pth \
    --image path/to/image.png \
    --calibration calibration/calibration_output/calibration_results.yaml
```

### CLI — Training Only

```bash
python training/Training/train.py --config training_output/optical_config.yaml
```

---

## GUI Reference

### Tab 1 — Scan and Configure

Load the preprocessing output and set the training configuration.

**Step 1: Load crops**

Click **Browse** and navigate to the folder containing preprocessed sphere crops (the `OUTPUT*/` directory from preprocessing). The GUI counts the images found and reports the scan result.

**Step 2: Select mode**

Choose between **Optical Mode** and **Direct Calibration Mode** using the radio buttons.

- *Optical Mode:* Enter camera parameters manually in the fields that appear.
- *Direct Calibration Mode:* Click **Browse** to load `calibration_results.yaml`. The fields `ρ_direct`, `σ₀`, and R² are populated automatically and are read-only.

**Optical parameters (Optical Mode):**

| Parameter | Description |
|-----------|-------------|
| Focal Length (mm) | Camera lens focal length |
| F-number | Aperture f-stop (e.g. 4.0 for f/4) |
| Focus Distance (mm) | Distance to the in-focus reference plane |
| Pixel Size (mm) | Physical size of one sensor pixel |
| Defocus Min/Max (mm) | Range of defocus to simulate during data generation |
| Rho (ρ) | Gaussian blur scaling constant (typically 1.0) |

**Step 3: Save configuration**

Click **Save Config** to write `optical_config.yaml` to the crops directory. This file is required before proceeding to Tab 2.

---

### Tab 2 — Generate

Create the synthetic training dataset from the loaded sharp crops.

The generation process applies a range of synthetic blur levels to each sharp reference image, producing blur/sharp/CoC map triples with precisely known ground truth. This is necessary because it is impossible to obtain perfectly labelled blur data from real images.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| Output directory | Where to write the synthetic dataset |
| Samples per image | Number of blurred variants to generate per reference crop. Recommended: 20–50. |
| Distribution | `Uniform` (equal sampling) or `Weighted` (Beta distribution) |
| Beta α and β | Shape parameters for the Beta distribution (see table below) |
| Min CoC filter | Skip samples below this CoC threshold. Set to 0 to include all. |

**Beta distribution guide — choosing α and β:**

| Goal | α | β | Effect |
|------|---|---|--------|
| General purpose | 2.0 | 2.0 | Bell curve centred at mid-range |
| Real data is mostly sharp | 2.0 | 5.0 | More samples near focus |
| Real data is mostly blurred | 5.0 | 2.0 | More samples at high defocus |
| Emphasise extremes | 0.5 | 0.5 | U-shape: sharp and very blurred |
| Uniform (equivalent) | 1.0 | 1.0 | Equal across full range |

The GUI provides forward and reverse Beta calculators to help choose parameters before generating.

Click **Generate** and wait for the progress bar to complete.

---

### Tab 3 — Train

Train the neural networks on the generated dataset.

**Training pipeline options:**

| Mode | Description | When to Use |
|------|-------------|-------------|
| Full Pipeline | Train DME, then freeze it and train DD | Starting from scratch — standard workflow |
| DME Only | Train only the CoC prediction network | Improving defocus estimation without retraining DD |
| DD Only | Train deblurring with a frozen pre-trained DME | Improving deblurring quality without repeating DME training |

**Key training parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| Batch Size | 8 | Samples per training step. Reduce if GPU runs out of memory. |
| DME Epochs | 100 | Training epochs for the DME stage |
| DD Epochs | 100 | Training epochs for the DD stage |
| Learning Rate | 0.0001 | Initial learning rate (Adam optimiser) |
| Validation Split | 0.2 | Fraction of data held out for validation |

**Checkpoint management:**

- **Auto-detect:** Scans the data directory for the most recent checkpoint and loads it automatically. Use this to resume an interrupted training run.
- **Browse:** Select a specific checkpoint file manually.
- **Clear:** Start fresh with no pre-loaded weights.

When loading a checkpoint, you can optionally set a new learning rate (Learning Rate Override). Use this for fine-tuning at a lower learning rate after initial convergence.

**Validation split strategy:**

| Strategy | Description |
|----------|-------------|
| Random | Random 80/20 split. Set a seed for reproducibility. |
| Stratified | Split balanced across CoC bins. Ensures all blur levels are represented in validation. |

**Monitoring training:**

```bash
tensorboard --logdir training/Training/training_output/checkpoints/logs
```

Open [http://localhost:6006](http://localhost:6006) in a browser to view loss curves and metrics in real time.

---

### Tab 4 — Validation and Testing

Evaluate the trained model on held-out data with detailed metrics.

**Test modes:**

| Mode | Metrics Produced |
|------|-----------------|
| DME Only | CoC MAE (px), CoC RMSE, error by CoC bin |
| Full Pipeline | PSNR, SSIM, diameter error (px), defocus error (%) |

**Output files:**

```
test_results/
├── test_summary.txt           ← Overall metrics and run configuration
├── test_metrics.csv           ← Per-sample metrics
├── worst_cases/               ← Visualisations of worst-performing samples
├── coc_scatter.png            ← Predicted vs ground truth CoC scatter plot
├── error_histogram.png        ← Distribution of prediction errors
└── binned_metrics.png         ← Metrics grouped by CoC range
```

---

### Tab 5 — Inference and Analysis

Run the trained model on real (non-synthetic) sphere images, organised by material.

**Input structure:**

```
input_directory/
├── 8mm-borosilicate/
│   ├── sphere0001v_crop.png
│   └── ...
└── 6mm-steel/
    └── ...
```

**Output structure:**

```
inference_results/inference_YYYYMMDD_HHMMSS/
├── <material>/
│   ├── blur_results.csv            ← Per-crop results
│   ├── crops_deblurred/            ← Deblurred reconstructions
│   └── visualizations/             ← 3-panel comparison images
├── summary_all_materials.csv
└── summary_analysis.png            ← 4-panel analysis plot
```

**CSV columns:**

| Column | Description |
|--------|-------------|
| `filename` | Input crop filename |
| `coc_px` | Predicted Circle of Confusion (pixels) |
| `defocus_mm` | Corresponding defocus distance (mm) |
| `focus_status` | `in_focus` or `out_of_focus` |
| `diameter_original_px` | Measured diameter from original image |
| `diameter_deblurred_px` | Measured diameter from deblurred image |
| `material` | Material folder name |

---

## Configuration Reference

### optical_config.yaml (Optical Mode)

```yaml
# Optical system parameters
focal_length_mm: 200.0       # Lens focal length (mm)
f_number: 4.0                # Aperture f-stop
focus_distance_mm: 400.0     # Distance to in-focus reference plane (mm)
pixel_size_mm: 0.020         # Sensor pixel size (mm)

# Defocus range for synthetic data generation
defocus_min_mm: -12.0        # Negative: closer to camera than focal plane
defocus_max_mm: 12.0         # Positive: farther from camera than focal plane

# Blur scaling constant
rho: 1.0                     # Gaussian blur scaling factor (typically 1.0)
```

### optical_config.yaml (Direct Mode additions)

```yaml
training_mode: direct
rho_direct: 0.699            # From calibration: pixels of blur per mm of defocus
sigma_0: 0.28                # From calibration: residual blur at focal plane (pixels)
```

### CoC Formula (Optical Mode)

```
CoC (px) = ρ × A × |d_i − d_o'| / d_o'
```

Where:
- `ρ` = rho scaling constant
- `A` = aperture diameter = `focal_length / f_number`
- `d_i` = imaging distance (from thin-lens equation)
- `d_o'` = object distance at defocus plane = `focus_distance + defocus`

---

## Troubleshooting

| Symptom | Likely Cause | Resolution |
|---------|-------------|------------|
| Poor CoC prediction | Optical config does not match the actual camera | Verify all parameters; consider switching to Direct Calibration Mode |
| Blurry deblurred output | DME predictions are inaccurate | Improve DME first; DD quality is bounded by DME quality |
| CUDA out-of-memory | Batch size too large for available GPU memory | Reduce batch size in Tab 3 |
| Mode mismatch warning on checkpoint load | Checkpoint was trained in a different mode | Confirm `training_mode` field in the checkpoint matches the current config |
| Direct Mode blocked at training start | Calibration YAML not loaded | Return to Tab 1 and use Browse to load `calibration_results.yaml` |
| Training not improving after many epochs | Learning rate too high or data distribution issue | Try loading the checkpoint with a lower LR override; check validation loss is not flat |

---

## Project Files

| File | Description |
|------|-------------|
| `training_gui.py` | Main 5-tab GUI application |
| `train.py` | Training engine (Trainer class; used by GUI and CLI) |
| `model.py` | DefocusNet architecture (DMESubnet, DDSubnet, ResBlock) |
| `synthetic_blur.py` | Blur synthesis engine with dual-mode support (OpticalParams, CoCCalculator, SyntheticBlurGenerator) |
| `dataset.py` | PyTorch Dataset classes and DataLoader construction |
| `losses.py` | Loss functions (DMELoss, DDLoss) |
| `predict.py` | CLI inference entry point (DefocusEstimator class) |
| `inference_real_crops.py` | GUI inference engine (RealCropInference class) |
| `test_model.py` | Model evaluation utilities |
| `test_cuda.py` | CUDA availability and device diagnostics |

---

## Reference

Wang, Z. et al. (2022). "Three-dimensional measurement of the droplets out of focus in shadowgraphy systems via deep learning-based image-processing method." *Physics of Fluids*, 34(7), 073301.
https://doi.org/10.1063/5.0090714
