# Inference

Tk GUI for running a trained defocus checkpoint on shadowgraphy data.
Two input modes (`.cine` recordings, pre-cropped PNGs); auto-picks a
preprocessing recipe per input; outputs go to dated session folders
that accumulate as you work.

The locked banner at the top of the window shows the model file, the
`CalibrationModel` baked into the checkpoint (method, ρ, σ_floor or σ₀,
sha256), and the deployment camera scale `s_c`. When a checkpoint
loads, the engine **overwrites** the user's ρ/σ₀/scale settings from
the checkpoint values — what the banner shows is what's actually used
in the inversion. `s_c` is the only banner field the user can edit.

Each Save produces a session folder with PNGs + a CSV row + a
regenerated summary. Batch produces one timestamped folder per click.

## Quick start

Three equivalent launch commands:

```bash
cropping-infer                              # console script
python -m Inference.inference_gui           # module form
python Inference/inference_gui.py           # F5-friendly direct launch
```

`python -m Inference` does **not** work — there is no `Inference/__main__.py`.
Use `python -m Inference.inference_gui`.

Minimal flow:

1. Pick input mode on the tab strip — `.cine input` or `Precropped PNG`.
2. Browse to a file or folder. Set `s_c` (deployment camera scale,
   px/mm) in the locked banner if it isn't already correct.
3. Click **Process** — result panel populates with σ + defocus +
   bounds flag.
4. Click **Save** — checkbox dialog asks what to write — files dropped
   into `Inference/output/sessions/<YYYY-MM-DD>_<source-folder>/`.
5. Or click **Batch folder…** to process every file in a folder
   unattended into a fresh `Inference/output/batch/<YYYY-MM-DD_HHMMSS>_<source-folder>/`.

The session folder accumulates across the day for the same input
folder — no manual session management.

## The two input modes

| Mode | Use when | What the engine does |
|---|---|---|
| `.cine input` | You have raw Phantom recordings of droplets falling onto a sphere | Open `.cine` → pick best pre-collision frame → sphere-guarded crop → preprocess → model → invert |
| `Precropped PNG` | You have your own square-cropped PNGs from any source | Read PNG → preprocess → model → invert (no frame selection, no cropping) |

Both share the same preprocess → model → invert tail. The mode just
chooses what comes in.

`.cine` mode also supports a watch folder: tick "Auto", point at a
folder, drop new `.cine` files in, GUI polls every 10 s and processes
them automatically. PNG mode is per-file or per-folder, no watch
support.

## The locked banner — what it shows and why

Three rows across the top of the window:

```
MODEL          dme_calib_best.pth ✓
CALIBRATION    hybrid · ρ=1.048 · σ_floor=0.927 · sha=25b53…  [from checkpoint]
DEPLOYMENT     s_c = [102.57] px/mm   verify
```

The "honesty layer" is the mechanism behind the banner. When
`InferenceEngine.load_checkpoint()` finds a `CalibrationModel` baked
into the checkpoint config (`config.training.calibration_model`), it
mutates `self.settings` to overwrite:

- `rho` ← `CalibrationModel.rho_px_per_mm`
- `sigma_0` ← `CalibrationModel.sigma_floor_calib_px` (quad/hybrid) or
  `sigma_0_calib_px` (linear)
- `s_calib` ← `CalibrationModel.s_calib_px_per_mm`
- `rho_std` ← `CalibrationModel.loo_cv['rho_std']`
- `sigma_0_std` ← `CalibrationModel.loo_cv['aux_param_std']`

The banner reads back from engine state, not from the JSON file. What
the user sees is what was actually used at inference time — stale GUI
defaults can't lie.

`s_c` (deployment camera scale, px/mm) is the only banner field the
user can edit. It's not knowable from the checkpoint — different
deployment cameras have different pixel scales for the same physical
setup, and only the user can tell the GUI what camera the inference
data came from.

## Auto-preprocess decision

The model expects flattened input. `auto_preprocess.decide_flatten_mode()`
picks one mode per input via a four-rule cascade:

```
1. Detect sphere via Calibration.sphere_processing.find_sphere_center
2. Compute fill_ratio   = π·r² / crop_area
   Compute edge_clearance = (crop_dim/2) − sphere_radius
3. if detection failed:                  → boundary_normalise
   elif edge_clearance < inner_margin+5: → boundary_normalise
   elif fill_ratio > 0.6:                → boundary_normalise
   else:                                  → calibration
```

Returns an `AutoDecision` with the mode, a human-readable rationale,
and diagnostic info (`detection_succeeded`, `fill_ratio`,
`edge_clearance_px`, `sphere_radius_px`). The rationale is shown
verbatim above the preview thumbnails so you can sanity-check the pick.

The override dropdown above the thumbnails locks a specific mode for
the session. **Four modes** are reachable via override:

| Mode | What it does | Auto-pickable? |
|---|---|---|
| `calibration` | `flatten_sphere_crop(inner_margin=20, flatten_exterior=False)` — interior-only flatten with a margin to preserve the central caustic | Yes |
| `boundary_normalise` | Otsu + cosine feather flatten (handles tight droplet crops where the sphere fills most of the frame) | Yes |
| `simple` | `flatten_sphere_crop()` with default args — full flatten with zero inner margin | **Override-only** |
| `none` | Pass-through. No flatten — just resize to model size. | **Override-only** |

`simple` and `none` exist for the override dropdown only; the
auto-decide heuristic never returns them. Use `none` when feeding
already-preprocessed crops (e.g. a Calibration run's `processed_images/`
PNGs directly — they're already flattened).

## GUI layout

Top-down map of the window:

```
┌─────────────────────────────────────────────────────────────────┐
│ Locked banner    (model / calibration / s_c)                    │
├─────────────────────────────────────────────────────────────────┤
│ Input mode tabs  (.cine | Precropped PNG)                       │
│   .cine pane:    file browser, watch folder, auto-poll          │
│   PNG pane:      folder browser, current file index             │
├─────────────────────────────────────────────────────────────────┤
│ Preview row                                                     │
│   Auto-decision rationale text                                  │
│   [override dropdown:  Auto / calibration / boundary_… / …]     │
│   ┌───────┐  ┌───────┐  ┌───────┐                              │
│   │  raw  │  │ proc. │  │sphere │                              │
│   │ crop  │→ │ crop  │  │detect │                              │
│   └───────┘  └───────┘  └───────┘                              │
│   "σ predicted: 1.34 px → z = 1.28 mm  [IN_RANGE]"              │
├─────────────────────────────────────────────────────────────────┤
│ Frame view tab  (zoom/pan canvas)                               │
│   View modes: Raw / Overlay / Crop / Processed                  │
│   Frame slider: 0 ── current ── total                           │
├─────────────────────────────────────────────────────────────────┤
│ Result panel       (big bold defocus number + bounds flag)      │
│ Active mode strip  ("Active mode: calibration · auto-picked")   │
│ Session indicator  ("Session: 4mm-borosilicate · 3 saved")      │
├─────────────────────────────────────────────────────────────────┤
│ Action bar  [Process] [Save] [Batch…] [◀ Prev] [Next ▶] [⚙]    │
├─────────────────────────────────────────────────────────────────┤
│ Status bar  (progress bar, validation banner, status text)      │
└─────────────────────────────────────────────────────────────────┘
```

| Element | Purpose |
|---|---|
| Locked banner | Always-visible model + calibration + `s_c`. See the section above for the honesty-layer mechanism. |
| Input mode tabs | Switches between `.cine` and PNG input flows. Holds per-mode controls (file/folder browsers, watch-folder polling for `.cine`). |
| Preview row | Three live thumbnails (raw / processed / sphere overlay) plus the override dropdown and the auto-decision rationale text. Updates whenever a new input loads or a parameter changes. |
| Frame view tab | Main viewer — zoom/pan canvas with four view modes (next section). |
| Frame slider | Scrubs through the cine's full frame range. Best-frame index is marked on the slider track. PNG mode hides this. |
| Result panel | Big bold defocus number + bounds flag colour-coded (green / orange / red) + per-pixel diagnostics. Persists across input changes until the next Process. |
| Active mode strip | Shows which flatten mode the next Process will use, plus whether it was auto-picked or user-overridden. |
| Session indicator | Current session folder name + number of saves so far. |
| Action bar | Process / Save / Batch folder… / ◀ Prev / Next ▶ / Settings. Tooltips on every button. |
| Status bar | Bottom of window — progress bar (hidden when idle), status text, validation banner (red for critical, yellow for warning, green for ok). |

## The Frame view tab

Zoomable / pannable canvas with four view modes selected by a radio bar
above:

| Mode | What's shown |
|---|---|
| Raw | Current frame at the slider position |
| Overlay | Geometry annotations on the raw frame: droplet bounds, sphere line, best-frame crop box, distance labels. **Disabled in PNG mode** (no underlying frame to annotate). |
| Crop | The extracted region the model sees, pre-flatten |
| Processed | The exact image fed to the model — post-flatten, pre-resize |

Zoom: scroll wheel (centred on cursor). Pan: middle-click drag.
Reset: right-click on the canvas.

The frame slider scrubs through the full `.cine` frame range and
updates the canvas live. The best-frame index chosen by the engine is
marked on the slider track.

## Settings dialog

Three clearly-separated sections:

### Section 1 — From Checkpoint (read-only)

Greyed-out fields showing what the engine pulled from the checkpoint:

- `inversion_method` (`linear` / `quadrature` / `hybrid`)
- `rho_px_per_mm`
- `sigma_0` or `sigma_floor_calib_px` (whichever the method uses)
- `s_calib_px_per_mm`
- `loo_cv` standard deviations (`rho_std`, `aux_param_std`)
- `calibration_source_sha256` (12-hex prefix)

These are honest about what's in use. They cannot be edited from the
GUI — change them by training a new checkpoint or editing one via
[`Training/calibration_editor.py`](../Training/calibration_editor.py).

### Section 2 — Required (deployment-specific, prominent)

- `s_c` — deployment camera scale (px/mm). Bigger field with helper
  text and a validation indicator (yellow border if very different
  from `s_calib`).

### Section 3 — Advanced (preprocessing tweaks)

- `flatten_mode` dropdown (`auto` / `calibration` / `boundary_normalise` / `simple` / `none`)
- `inner_margin_px` — only enabled when `flatten_mode = calibration`
- `feather_px` — only enabled when `flatten_mode = boundary_normalise`
- `crop_size`
- `device` (`cpu` / `cuda` if available)

Live thumbnail previews refresh as you change fields.

## Save vs Batch flows

Two mutually exclusive output flows.

### Save

Pops a checkbox dialog: choose what to write per cine (raw frame /
overlay / crop / processed / CSV row). Writes into the active session
folder, auto-grouped by `(date, source folder)`:

```
Inference/output/sessions/{YYYY-MM-DD}_{source-folder}/
```

Multiple Save clicks across the same day for the same input folder
**accumulate into one session** — no manual session management. Closing
and reopening the GUI on the same day with the same input folder
resumes the same session.

Per Save: writes per-cine PNGs into a new sub-folder, appends a row to
`results.csv`, regenerates `summary.txt` and
`bounds_flag_distribution.png` from the full CSV.

### Batch folder…

Asks the save-options dialog up-front (since batch runs unattended),
then processes every cine/PNG in the chosen folder into a **fresh
timestamped batch folder**:

```
Inference/output/batch/{YYYY-MM-DD_HHMMSS}_{source-folder}/
```

One folder per Batch click. Never accumulates across clicks. Same
per-file outputs as Save, plus a `run_metadata.yaml` capturing the
batch settings.

## Outputs

```
Inference/output/
├── sessions/
│   └── 2026-04-29_4mm-borosilicate/             # auto-grouped per (date, parent dir)
│       ├── session_metadata.yaml                # written once on first save
│       ├── results.csv                          # 18 columns, appended per save
│       ├── summary.txt                          # regenerated every save
│       ├── bounds_flag_distribution.png         # regenerated every save
│       └── per_cine/
│           └── sphere0959v_153022/
│               ├── raw_frame.png                # if requested
│               ├── overlay.png                  # if requested (omitted in PNG mode)
│               ├── crop.png                     # if requested
│               ├── processed.png                # if requested
│               └── result.txt                   # always written
└── batch/
    └── 2026-04-29_154120_4mm-borosilicate/      # one per Batch click
        ├── run_metadata.yaml
        ├── results.csv
        ├── summary.txt
        ├── bounds_flag_distribution.png
        └── per_cine/{stem}_{HHMMSS}/...
```

### `results.csv` — schema

18 columns, written in this order:

| Column | Source | Meaning |
|---|---|---|
| `timestamp` | save time | `YYYY-MM-DD HH:MM:SS` |
| `cine_filename` | input | basename of the input file (`.cine` or `.png`) |
| `best_frame_idx` | engine | frame chosen (empty for PNG mode) |
| `sigma_native_px` | engine | predicted σ at native pixel scale |
| `sigma_model_px` | engine | predicted σ at model-input pixel scale |
| `pred_norm` | engine | raw network output ∈ [0, 1] |
| `defocus_mm` | engine | inverted via `CalibrationModel.inverse_at()` |
| `defocus_uncertainty_mm` | engine | LOO-propagated ±mm |
| `bounds_flag` | engine | `IN_RANGE` / `BELOW_FLOOR` / `SATURATED` |
| `inversion_method` | engine | `linear` / `quadrature` / `hybrid` |
| `auto_picked_mode` | auto-decide | mode the heuristic chose |
| `auto_rationale` | auto-decide | human-readable rationale |
| `detection_succeeded` | auto-decide | sphere detection true/false |
| `fill_ratio` | auto-decide | `π·r² / crop_area` |
| `edge_clearance_px` | auto-decide | `crop_dim/2 − sphere_radius` |
| `saved_views` | save dialog | comma list of what was saved (`raw,overlay,crop,processed`) |
| `calibration_sha256` | engine | 16-hex prefix of the embedded `CalibrationModel` |
| `model_sha256` | engine | 16-hex prefix of the `.pth` file |

The two SHA columns let you cross-check what model + what calibration
produced each row, weeks after the fact.

### `result.txt` — per-cine human summary

Always written, even if no PNG views were requested. Contains the
defocus + uncertainty + bounds flag, σ_native + σ_model + pred_norm,
calibration method + ρ + σ_floor + sha, the auto-picked preprocessing
mode + rationale, and the checkbox state from the save dialog.

### `session_metadata.yaml` / `run_metadata.yaml`

Written once per session (or per batch run): model filename + sha256,
calibration block, deployment `s_c`, settings snapshot, started_at.

### `summary.txt` and `bounds_flag_distribution.png`

Regenerated from `results.csv` on every save. The text file gives
aggregate bounds-flag counts, defocus statistics for `IN_RANGE` rows,
auto-preprocess decision counts, and a per-cine listing. The PNG plots
the distribution of bounds flags across all saves so far.

## Reading the results

What the numbers mean and how to interpret them.

### Prediction chain

```
Network output:  pred_norm  ∈ [0, 1]
   × max_blur (from checkpoint)
sigma_model_px  (σ at model-input pixel scale)
   × (native_size / model_size)
sigma_native_px  (σ at the native pixel scale)
   ├── invert via CalibrationModel.inverse_at(sigma_native, s_inf=s_c, ...)
   └── (|z|_mm,  BoundsFlag)
   ├── propagate uncertainty via CalibrationModel.defocus_uncertainty()
   └── ±mm error bar
```

`max_blur`, `model_size`, `s_calib_px_per_mm` and the full
`CalibrationModel` all come from the checkpoint. `s_c` and the
`flatten_mode` come from the GUI.

### The three trust flags

| Flag | Meaning | What to do |
|---|---|---|
| `IN_RANGE` | σ is within the calibration's trusted bounds (`sigma_min_trusted ≤ σ ≤ sigma_max_trusted`) | Use the prediction with confidence; uncertainty bar is honest |
| `BELOW_FLOOR` | σ ≤ `sigma_min_trusted` (or ≤ `σ_floor` for quadrature/hybrid) | Returns `z = 0`. The droplet is at or beyond your camera's diffraction-limited resolution. |
| `SATURATED` | σ ≥ `min(sigma_max_trusted_calib, sigma_max_model_observed)` | Returns the **best-guess `|z|`** from the inverse, not NaN. Treat with low confidence — this is the extrapolation regime. |

`SATURATED` happens when the input blur exceeds either the
calibration's σ ceiling or the model's empirical σ-output plateau
(`sigma_max_model_observed_px`, baked into the checkpoint at training
time). Two ceilings — whichever fires first.

### Uncertainty bars

The `±mm` shown in the result panel comes from
`CalibrationModel.defocus_uncertainty()`, which propagates the LOO
`rho_std` and `aux_param_std` from the checkpoint through the inverse
formula at the predicted σ. For a typical hybrid calibration this is
roughly `±0.05 mm` mid-range.

For checkpoints without a `loo_cv` block, the uncertainty bar reads
`0.000` — the prediction is a point estimate.

## Caveats and gotchas

- **No `__main__.py`.** `python -m Inference` doesn't work. Use
  `python -m Inference.inference_gui` or the `cropping-infer` console
  script.
- **`physics.CalibrationModel` lives at the repo root**, not under
  `Inference/`. The engine imports it via `from physics import
  CalibrationModel` after the sys.path bootstrap at the top of
  `inference_gui.py`.
- **The engine mutates `self.settings` when a checkpoint loads.**
  `rho`, `sigma_0`, `s_calib`, `rho_std`, `sigma_0_std` are overwritten
  from the checkpoint's `CalibrationModel`. The locked banner reads
  back from engine state, not the JSON file. `s_c` is the only field
  the user controls per deployment.
- **`simple` and `none` modes are user-override-only.** The auto-decide
  heuristic only ever picks `calibration` or `boundary_normalise`. To
  use `simple` or `none`, lock the override dropdown.
- **`none` mode is for already-preprocessed inputs.** Use it on
  Calibration's `processed_images/*.png` (they're already flattened) or
  anything else you've preprocessed externally. Don't use on raw crops.
- **`s_c` matters for cross-camera work.** `s_c` is your deployment
  camera's pixel scale (px/mm). If your inference camera differs from
  the calibration camera, `s_c ≠ s_calib`, and the inverse correctly
  rescales. If both equal, the math falls back to "same camera".
- **Sphere flattening can fail silently.** When
  `flatten_sphere_crop()`'s sphere detection misses, the engine falls
  back to `boundary_normalise` and prints a warning. Failures are
  visible in the third preview thumbnail (sphere overlay) — if no
  circle is drawn, detection failed.
- **Watch folder is `.cine`-mode only.** PNG mode doesn't poll.
- **PNG mode disables Overlay view.** Geometry annotations require the
  underlying frame and detected geometry, neither of which exists in
  PNG mode.
- **`widgets/watch_folder_bar.py` is unused.** Watch-folder controls
  live inside `input_mode_tabs.py`; the standalone widget is never
  imported anywhere.
- **Settings JSON persists between sessions.** Stored at
  `Inference/inference_settings.json`; auto-created on first save.
  Delete to reset to defaults.
- **The session indicator shows the *current* session, not the
  *latest*.** Each Save updates the count. Closing and reopening the
  GUI on the same day with the same input folder resumes the same
  session.

## Settings JSON

Stored at [`inference_settings.json`](inference_settings.json) — auto-created on first save.

| Field | Source of truth | Notes |
|---|---|---|
| `model_path` | GUI (Settings) | Path to `.pth` |
| `input_mode` | input mode tabs | `cine` or `png` |
| `last_cine`, `watch_folder`, `png_folder`, `last_png` | GUI | Most-recent paths for convenience |
| `s_c` | locked banner | Deployment camera scale (px/mm) — user-editable |
| `flatten_mode` | preview row dropdown | `auto` / `calibration` / `boundary_normalise` / `simple` / `none` |
| `inner_margin_px`, `feather_px` | Settings dialog (Advanced) | Preprocessing knobs (mode-gated) |
| `crop_size`, `device` | Settings dialog (Advanced) | Model crop size + `cpu` / `cuda` |
| `rho`, `sigma_0`, `s_calib`, `rho_std`, `sigma_0_std` | **engine** (overwrites GUI) | Diagnostic only — engine writes from checkpoint on load |

The five engine-overwritten fields are diagnostic. The active source of
truth at inference is always the `CalibrationModel` instance the engine
built from the checkpoint, not the JSON.

## File map

```
Module entry
  inference_gui.py             composition root: state + widget wiring + event handlers
                                (sys.path bootstrap at the top)
  __init__.py                  empty package marker

Engine + I/O
  inference_engine.py          InferenceEngine: load model + CalibrationModel,
                                select_best_frame, extract_crop, preprocess_crop (4 modes),
                                run_inference. Plus the legacy boundary_normalise function.
  auto_preprocess.py           pure-logic flatten-mode picker
                                (decide_flatten_mode, resolve_mode, AutoDecision)
  geometry_overlay.py          draw_geometry_overlay (BGR overlay drawing)
  run_io.py                    session/batch folder layout, SaveOptions,
                                CSV/YAML writers, summary.txt + bounds_flag_distribution.png
                                regeneration
  inference_settings.json      auto-created persisted user state

widgets/
  locked_banner.py             top: model + calibration + s_c
  input_mode_tabs.py           .cine vs Precropped PNG tab strip + watch-folder controls
  preview_row.py               three thumbnails + override dropdown + result line
  frame_canvas.py              zoom/pan main viewer
  view_mode_toggle.py          Raw / Overlay / Crop / Processed radio bar
  frame_slider.py              frame scrubber + best-frame marker
  settings_dialog.py           three-section settings dialog
  action_bar.py                Process / Save / Batch / Prev / Next / Settings + tooltips
  status_bar.py                progress + validation banner
  result_panel.py              big persistent result number
  active_mode_strip.py         "Active mode: …" indicator
  session_indicator.py         "Session: <name> · N saved"
  save_dialog.py               checkbox save-options modal
  tooltips.py                  hover tooltip helper

Unused
  widgets/watch_folder_bar.py  never imported anywhere — superseded by
                                input_mode_tabs.py's watch-folder controls
```

## Where it sits in the pipeline

**Upstream:**

- **Training** — consumes the `.pth` checkpoint via
  `Training.model.DefocusNet` (the class only) and reads everything
  else from the checkpoint dict (weights, `max_blur`, `log_eps`,
  embedded `CalibrationModel`, `sigma_max_model_observed_px`).
- **Calibration** — `Calibration.sphere_processing.flatten_sphere_crop`
  and `find_sphere_center` are the preprocessing primitives;
  `physics.CalibrationModel` (at the repo root) drives the σ → mm
  inversion.
- **Preprocessing** — Inference imports `cine_io`,
  `darkness_analysis`, `cropping`, `geom_analysis`, `image_utils` as
  code modules. Does **not** read Preprocessing's output files; works
  on its own input data.

**Downstream:** nothing. Inference is a leaf in the import graph — only
`Extras/tests/test_auto_preprocess.py` imports from it (and only the
mode constants + decision functions).
