# Defocus Depth Estimation for High-Speed Shadowgraphy

A complete end-to-end pipeline for estimating the 3D position of particles in high-speed shadowgraphy experiments using deep learning.

This project was developed as part of a mechanical engineering dissertation investigating depth measurement in particle flow experiments.

---

## Background

### What is Shadowgraphy?

Shadowgraphy is an imaging technique used in fluid dynamics research in which objects are backlit and captured as silhouettes by a high-speed camera. It is widely used to image droplets, bubbles, and particles in flow experiments because it requires no fluorescent dyes or laser illumination — the particles are simply imaged against a bright background.

### The Depth Estimation Problem

When a particle lies exactly at the focal plane of the camera, its silhouette appears sharp. When the particle is displaced from the focal plane — in either direction along the optical axis — its image becomes blurred. The amount of blur is a direct function of the physical displacement, a quantity known as the **defocus distance**.

The blurred appearance is characterised by the **Circle of Confusion (CoC)**: the diameter of the blur disk formed on the camera sensor when a point source is out of focus. A larger CoC means greater defocus and greater physical distance from the focal plane.

This relationship between blur and depth means that a single camera can estimate 3D particle positions from 2D images — provided the blur can be accurately measured and the blur-to-depth conversion is known.

### Why Deep Learning?

Classical approaches to blur measurement (edge-gradient fitting, Laplacian variance) are sensitive to noise and imperfect sphere geometry. A deep learning model trained on physics-based synthetic data provides more robust blur estimation, generalises better across particle sizes, and can simultaneously produce a deblurred image for more accurate size measurement.

---

## System Architecture

The system is built on **DefocusNet**, a two-subnet deep learning model:

```
Blurred Image ──► DME-subnet ──► CoC Map ──┬──► DD-subnet ──► Deblurred Image
                                            │
                                            └──► Defocus Distance (mm)
```

- **DME-subnet** (Defocus Map Estimator): a U-Net encoder-decoder that predicts the per-pixel Circle of Confusion from a blurred input image.
- **DD-subnet** (Defocus Deblurring): takes the blurred image alongside the predicted CoC map and recovers a sharp reconstruction.

The CoC value maps to physical defocus distance via a conversion model, which is where the two operating modes differ.

---

## Dual-Mode Architecture

A central design decision is support for two operating modes, which differ in how the blur-to-depth conversion is characterised.

### Optical Formula Mode

The blur produced by a defocused object follows the thin-lens equation. If the camera's optical parameters are precisely known (focal length, f-number, focus distance, pixel size), the Circle of Confusion at any defocus distance can be computed analytically.

The model is trained on synthetic data generated from this formula, and at inference the formula is inverted to recover depth from predicted CoC.

**Use when:** Optical parameters are accurately known.

### Direct Calibration Mode

In practice, exact optical parameters are often uncertain. The direct mode bypasses the formula entirely. A calibration sphere is physically moved through a range of known depths in the lab, and the blur at each position is measured. A linear model is fitted to this data:

```
σ (px) = ρ × |z| + σ₀
```

where `z` is physical displacement from the focal plane (mm), `ρ` is the fitted blur rate (px/mm), and `σ₀` is the residual blur at the focal plane. The model is trained using this empirical relationship, and inference uses the inverse to recover depth.

**Use when:** Optical parameters are uncertain, or when maximum accuracy for a specific camera setup is required.

The direct calibration mode was introduced to resolve a sim-to-real gap observed during development, where the optical-mode model predicted defocus values significantly different from ground truth. A key advantage of this mode is that the trained model does not need to be retrained if the camera is recalibrated — only the calibration YAML needs to be updated.

Both modes use the same neural network architecture. The mode is selected at training time, stored in the model checkpoint, and detected automatically at inference.

---

## Pipeline Overview

The pipeline consists of five stages. Each stage is independent and communicates with the next through intermediate files.

```
Step 0 — Lab Capture (Direct Mode only, one-time per camera setup)
         Hardware: Phantom camera + ThorLabs KDC101 stage + Arduino
         Output:   tiff images + positions.csv
              ↓
Step 1 — Preprocessing
         Input:    Raw .cine video files
         Output:   Cropped sphere images (PNG)
              ↓
Step 2 — Calibration
         Input:    Lab z-stack (Direct Mode) or optical parameters (Optical Mode)
         Output:   calibration_results.yaml
              ↓
Step 3 — Training
         Input:    Cropped sphere images + calibration YAML (Direct Mode)
         Output:   Model checkpoint (.pth)
              ↓
Step 4 — Inference
         Input:    Real images + model checkpoint
         Output:   defocus_mm per image, deblurred images, analysis plots
```

---

## Repository Structure

```
coursework/
│
├── README.md                              ← This file
│
├── Preprocessing/                         ← Frame extraction and sphere cropping
│   ├── CROPPING/Preprocessing/            ← Python scripts (entry point: gui.py)
│   └── README.md                          ← Preprocessing documentation
│
├── calibration/                           ← Blur-to-depth calibration
│   ├── calibration_gui.py                 ← Entry point
│   ├── calibration_core.py                ← Calibration fitting logic
│   ├── blur_measurement.py                ← Blur measurement algorithms
│   ├── calibration_output/
│   │   └── calibration_results.yaml       ← Calibration output (input to training)
│   └── README.md                          ← Calibration documentation
│
├── training/Training/                     ← Neural network training and inference
│   ├── training_gui.py                    ← Primary entry point (5-tab GUI)
│   ├── train.py                           ← Training engine (Trainer class)
│   ├── model.py                           ← DefocusNet architecture
│   ├── synthetic_blur.py                  ← Blur synthesis engine
│   ├── predict.py                         ← CLI inference
│   ├── inference_real_crops.py            ← GUI inference engine
│   ├── training_output/
│   │   ├── optical_config.yaml            ← Training configuration
│   │   ├── checkpoints/                   ← Saved model checkpoints
│   │   ├── synthetic_data/                ← Generated training dataset
│   │   └── inference_results/             ← Inference outputs
│   └── README.md                          ← Training documentation
│
├── lab_capture/                           ← Automated calibration data capture
│   ├── capture_calibration.py             ← Entry point
│   ├── check_arduino.py                   ← Hardware pre-flight check
│   └── README.md                          ← Lab setup and operating guide
│
├── spheres/                               ← Raw .cine video data
│   ├── 4mm-borosilicate/
│   ├── 6mm-borosilicate-*/
│   ├── 8mm-borosilicate-*/
│   └── ...
│
└── .planning/                             ← Project planning documents
```

---

## How to Run the Pipeline

### Step 0 — Capture Calibration Data (Direct Mode only)

*Skip this step if using Optical Mode, or if calibration data has already been collected.*

```
Hardware required: Phantom camera, ThorLabs KDC101 linear stage, Arduino Uno
Script:            lab_capture/capture_calibration.py
Output:            lab_capture/captures/<timestamp>/*.tiff + positions.csv
```

See [lab_capture/README.md](lab_capture/README.md) for full hardware setup and operating instructions.

---

### Step 1 — Preprocessing

Convert raw `.cine` video files into cropped sphere images.

```
Input:   spheres/<material>/*.cine
Script:  Preprocessing/CROPPING/Preprocessing/gui.py
Output:  Preprocessing/CROPPING/Preprocessing/OUTPUT*/<material>/
```

```bash
cd Preprocessing/CROPPING/Preprocessing
python gui.py
```

See [Preprocessing/README.md](Preprocessing/README.md) for configuration and usage.

---

### Step 2 — Calibration

Measure blur vs depth for your camera and fit the conversion model.

```
Input:   lab_capture/captures/ (Direct Mode)  OR  manual parameters (Optical Mode)
Script:  calibration/calibration_gui.py
Output:  calibration/calibration_output/calibration_results.yaml
```

```bash
cd calibration
python calibration_gui.py
```

See [calibration/README.md](calibration/README.md) for the full calibration procedure.

---

### Step 3 — Training

Generate synthetic training data and train the DefocusNet model.

```
Input:   Sharp sphere crops + calibration_results.yaml (Direct Mode)
Script:  training/Training/training_gui.py
Output:  training/Training/training_output/checkpoints/best_checkpoint.pth
```

```bash
cd training/Training
python training_gui.py
```

Work through the GUI tabs in order: Scan & Configure → Generate → Train.

See [training/Training/README.md](training/Training/README.md) for the complete training workflow.

---

### Step 4 — Inference

Run the trained model on real images to estimate defocus depth.

```
Input:   Real sphere images + model checkpoint + calibration YAML (Direct Mode)
Script:  training/Training/training_gui.py (Tab 5)
Output:  training/Training/training_output/inference_results/
```

Via GUI: use Tab 5 (Inference & Analysis) in `training_gui.py`.

Via CLI:

```bash
# Optical mode
python training/Training/predict.py \
    --checkpoint training/Training/training_output/checkpoints/best_checkpoint.pth \
    --image path/to/image.png

# Direct mode
python training/Training/predict.py \
    --checkpoint training/Training/training_output/checkpoints/best_checkpoint.pth \
    --image path/to/image.png \
    --calibration calibration/calibration_output/calibration_results.yaml
```

---

## Software Dependencies

### Core Requirements

```bash
pip install torch torchvision numpy opencv-python pandas pyyaml scipy scikit-learn matplotlib Pillow tqdm tensorboard
```

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11 | Required by the Phantom SDK |
| NumPy | < 2.0 | Phantom SDK compatibility |
| PyTorch | 1.9.0+ | GPU training strongly recommended |
| CUDA | 11.8, 12.1, or 12.4 | Optional but gives 10–100× speedup |

### Hardware-Specific Dependencies

These are required only for preprocessing and lab capture. Training and inference run independently.

**Phantom SDK (`pyphantom`)** — reads `.cine` video files. Not available on PyPI; must be installed from the Vision Research SDK:

```bash
pip install "path/to/pyphantom-3.11.11.806-py311-none-any.whl"
```

**ThorLabs Kinesis** — controls the KDC101 linear stage. Install the Kinesis software package from ThorLabs, then:

```bash
pip install pythonnet
```

DLLs must be present at `C:\Program Files\Thorlabs\Kinesis\`.

**Arduino (pyfirmata)** — hardware triggering during calibration capture:

```bash
pip install pyfirmata
```

> pyfirmata 1.1.0 uses `inspect.getargspec`, which was removed in Python 3.11. The lab capture scripts patch this automatically at startup — no manual fix is needed.

---

## Important Notes

### Pipeline Ordering is Strict

Each stage depends on the outputs of the previous stage:

- Preprocessing must be run before training. Sharp crops are required as training reference images.
- For Direct Mode, calibration must be run before training. The calibration YAML is loaded into the training configuration and cannot be generated on the fly.
- Training must be completed before inference. A model checkpoint is required.

### Training Mode is Permanent for a Checkpoint

The training mode (`optical` or `direct`) is selected at the start of training and stored in the model checkpoint. A checkpoint trained in one mode cannot be used for inference in the other. To change mode, training must be repeated. The inference scripts detect the mode automatically from the checkpoint.

### Crop Size Must Remain Consistent

The preprocessing pipeline crops spheres to a configured pixel size. If this size is changed, all synthetic training data must be regenerated and the model must be retrained. The crop size used during preprocessing must match the configuration used during data generation.

### Direct Mode Enables Recalibration Without Retraining

One of the key advantages of Direct Calibration Mode is that the fitted parameters `ρ` and `σ₀` are not baked into the model weights. If the camera setup changes (e.g. different focus distance or lens), the `calibration_results.yaml` can be updated with new measurements and the same trained model used for inference. Retraining is only necessary if the blur pattern itself changes significantly.

### No Formal Test Suite

This is a research project. Validation is performed through visual inspection, R² scoring, and TensorBoard monitoring. Ad-hoc test scripts (`test_cuda.py`, `test_model.py`) are provided but there is no automated test framework.

---

## Reference

Wang, Z. et al. (2022). "Three-dimensional measurement of the droplets out of focus in shadowgraphy systems via deep learning-based image-processing method." *Physics of Fluids*, 34(7), 073301.
https://doi.org/10.1063/5.0090714
