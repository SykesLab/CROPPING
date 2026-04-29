# Inference

UX-first GUI for running the trained defocus model on `.cine` recordings
or pre-cropped PNG images. Imports the model + calibration directly;
emits per-cine and per-folder results into timestamped session folders.

## Run

Three equivalent ways:

- **F5** `Inference/inference_gui.py` in your IDE
- `python -m Inference.inference_gui` from CROPPING root
- `python Inference/inference_gui.py` from CROPPING root

The entry-point file (`inference_gui.py`) has a single, explicit
sys.path bootstrap at the top so direct/F5 launches resolve the
top-level packages (`Preprocessing`, `Calibration`, etc.). Every
other module / widget / test in this folder uses clean absolute
imports.

## What it does

### Two input modes

A tab strip above the main view selects how data comes in:

- **`.cine input`** — select a `.cine` file (or watch a folder of
  them). Engine picks best frame, extracts droplet crop, runs
  preprocess + model + inversion. Suited for the original use case
  (droplet falling above a calibration sphere).
- **`Precropped PNG`** — select a folder of square PNGs. Each PNG is
  treated as already-cropped, fed straight to preprocess + model.
  Suited for any user with their own cropping pipeline.

Both modes share the same Process / Save / Batch / Prev / Next /
Settings actions; behaviour dispatches on mode.

### Always-visible top banner

Shows the loaded model, the calibration baked into its checkpoint
(method, ρ, σ_floor, sha256), and the deployment camera scale `s_c`.
The only field the user must set is `s_c` — everything else is locked
in by the checkpoint.

### Auto-pick preprocessing with live preview

Three thumbnails (original / processed / sphere-detect overlay) update
live whenever a cine or PNG is loaded. An auto-decide heuristic picks
between `calibration` (cfg1), `boundary_normalise` (cfg4), and other
flatten modes based on sphere fill ratio + edge clearance. Override
dropdown locks a specific mode for the session.

### Frame view tab

Overlay-on-best-frame canvas: zoom/pan, frame slider, four view modes:
- **Raw** — current frame at slider position
- **Overlay** — geometry annotations (droplet bounds, sphere line,
  best-frame crop box, distance labels). Disabled in PNG mode.
- **Crop** — the extracted region the model will see (pre-flatten)
- **Processed** — the exact image fed to the model (post-flatten)

### Persistent result panel + active-mode strip

Below the tabs:
- Result panel shows the last Process result in big bold text
  (defocus mm + bounds_flag + per-pixel diagnostics)
- "Active mode" strip shows which flatten mode the next Process will use
- Session indicator shows the current session folder + cine count

### Save flow with sessions

Save button pops a checkbox dialog (raw frame / overlay / crop /
processed / CSV). Files go into a session folder auto-grouped by
source folder + date:

```
Inference/output/sessions/2026-04-29_4mm-borosilicate/
├── session_metadata.yaml      # model / calibration / s_c / settings
├── results.csv                 # one row per saved cine, appended
├── summary.txt                 # regenerated each save
├── bounds_flag_distribution.png
└── per_cine/
    ├── sphere0959v_153022/
    │   ├── raw_frame.png
    │   ├── overlay.png
    │   ├── crop.png
    │   └── processed.png
    └── …
```

Working through one folder of cines from the same dataset
auto-accumulates into one session — no manual session management.

### Batch flow

`Batch folder…` asks the save-options dialog up-front (since batch
runs unattended), then processes every cine/png in the chosen folder
into a timestamped batch folder:

```
Inference/output/batch/2026-04-29_154120_4mm-borosilicate/
```

## File structure

```
Inference/
├── __init__.py
├── inference_engine.py         # core: select_best_frame, extract_crop,
│                                # preprocess_crop, run_inference
├── inference_gui.py            # composition root: state + widget wiring
├── auto_preprocess.py          # pure-logic flatten-mode picker
├── geometry_overlay.py         # overlay annotation drawing
├── run_io.py                   # session/batch folder layout, metadata
├── inference_settings.json     # auto-created on first save
├── widgets/
│   ├── locked_banner.py        # top: model + calibration + s_c
│   ├── input_mode_tabs.py      # .cine vs PNG input tab strip
│   ├── preview_row.py          # 3 thumbs + override + result line
│   ├── frame_canvas.py         # zoom/pan main viewer
│   ├── view_mode_toggle.py     # Raw/Overlay/Crop/Processed bar
│   ├── frame_slider.py         # frame scrubber
│   ├── settings_dialog.py      # 3-section reorganised dialog
│   ├── action_bar.py           # Process/Save/Batch/Prev/Next/Settings
│   ├── status_bar.py           # progress + validation banner
│   ├── result_panel.py         # big persistent result number
│   ├── active_mode_strip.py    # "Active mode: …" indicator
│   ├── session_indicator.py    # "Session: 4mm-borosilicate · 3 saved"
│   ├── save_dialog.py          # checkbox save options
│   ├── tooltips.py             # hover tooltip helper
│   ├── watch_folder_bar.py     # legacy widget, retained
│   └── results_panel.py        # placeholder
└── README.md
```

## Decision rule (auto-preprocess)

Pure function in [auto_preprocess.py](auto_preprocess.py):

```
def decide_flatten_mode(crop, cfg1_inner_margin=20):
    1. detect sphere via Calibration.sphere_processing.find_sphere_center
    2. compute fill_ratio = π·r² / crop_area
    3. compute edge_clearance = min(crop_dim/2 − sphere_radius)
    4. if detection failed:                → boundary_normalise
       elif edge_clearance < inner_margin+5: → boundary_normalise
       elif fill_ratio > 0.6:                → boundary_normalise
       else:                                  → calibration
```

Returns a structured `AutoDecision` with the mode + a human-readable
rationale + diagnostic info. Rationale is shown verbatim above the
preview thumbnails so users can sanity-check.

## Settings JSON

Stored at `Inference/inference_settings.json`. Key fields:

```json
{
  "model_path": "…/dme_best.pth",
  "input_mode": "cine",
  "last_cine": "…/some.cine",
  "watch_folder": "…/some_dir",
  "png_folder": "…/png_dir",
  "last_png": "…/some.png",
  "s_c": 89.55555555555556,
  "flatten_mode": "auto",
  "inner_margin_px": 20,
  "feather_px": 40,
  "crop_size": 299,
  "device": "cpu",
  "rho": 1.0478,
  "sigma_0": 0.9274,
  "s_calib": 89.5556,
  "rho_std": 0.0056,
  "sigma_0_std": 0.0211
}
```

When a checkpoint with a baked-in CalibrationModel is loaded, the
engine overrides `rho`, `sigma_0`, `s_calib`, `rho_std`, `sigma_0_std`
with the checkpoint values automatically — those fields in the JSON
are diagnostic, not the source of truth.

## Tests

```
python -m pytest Extras/tests/test_auto_preprocess.py -v
```

Eleven tests covering the four decision branches + the resolve_mode
helper + a real-data smoke test against
`calibration spheres/9mm/9mm_30.cine`.
