# CROPPING — Defocus Depth Estimation for High-Speed Shadowgraphy

Estimate the depth of small particles — droplets, bubbles, beads — from
a single shadowgraph by measuring the blur of their image. Implemented
as a four-stage pipeline: preprocess raw `.cine` recordings into curated
crops; calibrate the blur-to-depth relationship in the lab with a known
sphere; train a small CNN on synthetic data generated from that
calibration; deploy the model on new recordings.

The calibration is **baked into the trained checkpoint**, so the
inference stage is self-contained. Ship the `.pth` to a colleague and
they can infer anywhere — no separate config file required.

## The pipeline at a glance

```
              HARDWARE INPUTS
   ┌────────────────────┴────────────────────┐
   │                                         │
.cine recordings              z-stack of calibration sphere
(droplets falling onto a       (motorised stage steps through
 calibration sphere)            known defocus positions)
   │                                         │
   ↓                                         ↓
┌──────────────┐                 ┌──────────────────────┐
│ Preprocessing│                 │     Calibration      │
│              │                 │                      │
│ best frame   │                 │ measure σ at edge,   │
│  → crop      │                 │ fit CalibrationModel │
│  → flatten   │                 │ (linear/quad/hybrid) │
│  → focus     │                 │ + LOO uncertainty    │
│    classify  │                 │                      │
└──────┬───────┘                 └──────────┬───────────┘
       │                                    │
       │  Focus/sharp_crops.csv             │  calibration_results.yaml
       │  + matching crops                  │  (with embedded CalibrationModel)
       │                                    │
       └──────────────┬─────────────────────┘
                      ↓
              ┌───────────────┐
              │   Training    │
              │               │
              │ synthesise    │
              │ blurred data  │
              │ from calib →  │
              │ train scalar  │
              │ DefocusNet    │
              │               │
              │ CalibModel    │
              │ baked into    │
              │ checkpoint    │
              └───────┬───────┘
                      │  checkpoints/dme_best.pth
                      │  (self-contained — no external files needed)
                      ↓
              ┌───────────────┐
              │   Inference   │
              │               │
              │ load .pth →   │
              │ predict on    │
              │ new data →    │
              │ defocus_mm    │
              │ + trust flag  │
              │ + ±mm bar     │
              └───────────────┘
```

Each stage is a self-contained Python module with its own GUI and
README:

- **[Preprocessing/](Preprocessing/README.md)** — turns raw Phantom
  `.cine` files into the curated, per-pixel-flattened crops that
  Training reads. Output: `Focus/sharp_crops.csv` plus matching PNG
  crops sorted by camera.
- **[Calibration/](Calibration/README.md)** — fits a `CalibrationModel`
  that converts blur σ ↔ defocus mm for a specific camera + lens.
  Output: `calibration_results.yaml` with the full model serialised,
  including trust bounds and LOO uncertainty.
- **[Training/](Training/README.md)** — generates synthetic blurred
  crops from the `CalibrationModel`, trains DefocusNet, and bakes the
  calibration into every checkpoint. Output: `.pth` files under
  `Training/training_output/models/<run>/checkpoints/`.
- **[Inference/](Inference/README.md)** — loads a checkpoint and runs
  it on new data. Two input modes (`.cine` and pre-cropped PNG); per-
  prediction uncertainty bars; session-based output organisation.

The hardware-capture sub-module
[Calibration/lab_capture/](Calibration/lab_capture/) drives a Phantom
camera + ThorLabs KDC101 stage + Arduino to record the calibration
z-stack — only needed if you're producing your own calibration data.

## Install

```bash
python install.py                # one-shot: creates venv + installs everything
python install.py --cuda 12.1    # same, with CUDA-enabled PyTorch
```

See [INSTALL.md](INSTALL.md) for the full guide — three install paths,
CUDA setup, the optional Phantom SDK and lab-capture hardware
dependencies, and troubleshooting.

## Run end-to-end

The four console scripts, in pipeline order:

```bash
cropping-preprocess     # Raw .cine → curated crops + sharp_crops.csv
cropping-calibrate      # Z-stack    → calibration_results.yaml + CalibrationModel
cropping-train          # crops + calibration → trained checkpoint (calibration baked in)
cropping-infer          # checkpoint + new data → defocus_mm per image
```

Each stage auto-discovers the most recent output of the previous stage
by directory mtime — point the GUI at a parent dir, or use the
"Load latest" buttons. No path-juggling required for the happy path.

CLI training is also supported for headless / batch runs:

```bash
python Training/train.py --config <yaml> --data-dir <dataset> --output-dir <run> [--resume <ckpt>]
```

A Windows convenience wrapper exists at
[run_training_cli.bat](run_training_cli.bat).

## Architecture you should know up front

The four modules aren't fully independent — they share a small amount
of cross-cutting code, and their data handoffs follow strict contracts.
Worth understanding before you start moving things around.

### Cross-cutting code

**[`physics.py`](physics.py)** lives at the repo root, not under any
module. It defines the `CalibrationModel` class and every σ↔z
conversion in the system. Three modules import it:

- Calibration **writes** it (fits the model, exports to YAML)
- Training **bakes** it into every checkpoint
- Inference **reads** it from the checkpoint and runs the inverse

It lives at the root because all three need it without any one of them
depending on the other two. If you move it, fix the `sys.path`
injection in
[`Calibration/calibration_core.py`](Calibration/calibration_core.py)
and the equivalent imports in Training and Inference.

**[`Calibration/sphere_processing.py`](Calibration/sphere_processing.py)**
is the most-imported leaf module in the repo. `flatten_sphere_crop` is
mandatory in Preprocessing's output pipeline; `find_sphere_center`
drives Inference's auto-preprocess heuristic; `process_sphere_stack`
is used by Training when the inference tab pre-processes real crops.

### Data handoffs

Three contracts bind the modules together:

| Producer → Consumer | File | Format |
|---|---|---|
| Preprocessing → Training | `Focus/sharp_crops.csv` + matching PNGs | CSV with `filename`, `sigma_px`, `diameter_px`, `native_blur_sigma`, `scale_px_per_mm`, `camera`. Training auto-discovers it via mtime by walking up parent dirs from any pointed-at folder. |
| Calibration → Training | `calibration_results.yaml` | YAML with embedded `calibration_model:` block (the full `CalibrationModel.to_dict()`). Training also reads the `direct:` back-compat block for older datasets. |
| Training → Inference | `checkpoints/dme_best.pth` | PyTorch dict containing weights + `config.training.calibration_model` (the CalibrationModel baked in by Training) + `max_blur` + `sigma_max_model_observed_px`. **No separate calibration file needed at inference.** |

### Why baking matters

Inference doesn't read `calibration_results.yaml` directly — it reads
the `calibration_model:` block from inside the checkpoint. The
checkpoint also carries the SHA256 prefix of the source CalibrationModel,
so you can verify provenance weeks after the fact.

Practical consequences:

- The `.pth` file is shippable as-is. A colleague can run inference
  without your `Calibration/runs/` directory being present.
- You can recalibrate without retraining (linear-mode checkpoints only)
  via [`Training/calibration_editor.py`](Training/calibration_editor.py),
  which writes a corrected copy under
  `models/<run>/edits/<edit_name>/dme_best.pth` and never overwrites
  the source.
- Quadrature and hybrid calibrations can't be edited post-hoc — re-fit
  with the new sphere data and retrain.

## Repository layout

```
CROPPING/
├── Preprocessing/         Raw .cine → curated crops + sharp_crops.csv
├── Calibration/           Z-stack → CalibrationModel + calibration_results.yaml
│   └── lab_capture/       Hardware capture (Phantom + ThorLabs + Arduino)
├── Training/              Synthesise + train + bake calibration into checkpoint
├── Inference/             Load checkpoint, predict defocus on new data
├── Extras/
│   ├── tests/             Pytest suite (224 tests at present)
│   ├── experiments/       Exploratory / one-off scripts
│   └── archive/           Historical material kept for reference
├── physics.py             CalibrationModel + σ↔z conversions (cross-cutting)
├── pyproject.toml         Source of truth for deps + console scripts
├── install.py             One-shot installer (creates venv, installs everything)
├── requirements.txt       Flat dep list (mirror of pyproject for plain pip install)
├── run_training_cli.bat   Windows convenience wrapper for the training CLI
├── INSTALL.md             Full install guide (CUDA, hardware deps, troubleshooting)
├── README.md              This file
└── LICENSE                GPL-3.0
```

The `calibration spheres/` directory at the repo root holds raw
calibration data for the user's own runs and is gitignored — it's not
part of the codebase.

## Tests

```bash
python -m pytest Extras/tests/ -v
```

Current coverage: auto-preprocess heuristic, calibration model
(linear/quadrature/hybrid forward + inverse, trust bounds, LOO),
synthetic blur generation, dataset loading, training-config validation,
end-to-end pipeline integration. See [Extras/tests/](Extras/tests/) for
the full file list.

## What's stable, what's experimental

This is research code from a dissertation project. The four module
pipelines are stable enough to use end-to-end — the GUIs handle the
common flows, the auto-discovery between stages works without manual
path juggling, and the checkpoint format is stable. The hybrid
calibration method, LOO uncertainty propagation, and SATURATED bound
are the most recently developed parts; the older linear-only path is
also fully supported.

Ad-hoc scripts under [Extras/experiments/](Extras/experiments/) are
exploratory and were written for specific investigations — useful as
references but not part of the maintained pipeline.
[Extras/archive/](Extras/archive/) is historical material kept for
provenance.

## Hardware requirements

- **Python** 3.10+ (3.11 specifically if you need pyphantom for `.cine`
  reading — the wheel is locked to 3.11).
- **PyTorch** 1.9+. CUDA optional but gives a 10–100× speedup for
  training. Inference is comfortable on CPU.
- **Phantom SDK / pyphantom** — only required for reading `.cine`
  files. Skip if you only have pre-cropped PNGs.
- **ThorLabs Kinesis + Arduino** — only required for the hardware lab
  capture sub-module ([Calibration/lab_capture/](Calibration/lab_capture/)).
  Skip if you have calibration images from another source.

See [INSTALL.md](INSTALL.md) for installation specifics.

## Citation

Built on the depth-from-defocus approach of:

> Wang, Z. et al. (2022). "Three-dimensional measurement of the droplets
> out of focus in shadowgraphy systems via deep learning-based
> image-processing method." *Physics of Fluids*, 34(7), 073301.
> [doi:10.1063/5.0090714](https://doi.org/10.1063/5.0090714)

```bibtex
@article{wang2022three,
  title   = {Three-dimensional measurement of the droplets out of focus
             in shadowgraphy systems via deep learning-based
             image-processing method},
  author  = {Wang, Zhendong and others},
  journal = {Physics of Fluids},
  volume  = {34}, number = {7}, pages = {073301}, year = {2022},
  publisher = {AIP Publishing},
}
```

## License

GPL-3.0. See [LICENSE](LICENSE).
