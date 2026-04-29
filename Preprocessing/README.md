# Preprocessing

Turns raw Phantom `.cine` recordings of droplets falling onto a calibration
sphere into the curated, per-pixel-flattened crops that Training reads.

Three things in order, per droplet:

1. Pick the best pre-collision frame (droplet fully formed, not yet touching
   the sphere).
2. Crop the droplet at a fixed size while shifting the crop box upward to
   keep the sphere out.
3. Flatten the residual sphere-illumination gradient via
   `Calibration.sphere_processing.flatten_sphere_crop`. Every saved crop is
   post-flattened — this is undocumented in older versions but load-bearing.

Output is a `Focus/sharp_crops.csv` manifest plus the matching crops sorted
by camera. Training discovers it automatically.

## Quick start

Prereqs: pyphantom installed and the project installed in editable mode.
See the repo-root [INSTALL.md](../INSTALL.md) for both.

Three equivalent launch commands:

```bash
cropping-preprocess              # console script (pip install -e .)
python -m Preprocessing          # module form
python Preprocessing/gui.py      # F5-friendly direct launch
```

Minimal flow:

1. **Browse** to a CINE root (a folder containing `.cine` files OR a parent
   folder containing subfolders of `.cine` files).
2. **Continue** — the GUI scans, shows `N folders, ~M droplets`.
3. Edit the **Run label** field if you want (auto-fills with the source
   folder basename — e.g. `4mm-borosilicate`).
4. **Start.** A timestamped run dir is created at
   `Preprocessing/output/runs/<YYYY-MM-DD_HHMMSS>_<label>/` and crops start
   appearing in the Recent Outputs row as workers finish.
5. When the pipeline completes, point Training at this run's `Focus/`
   directory — Training auto-discovers `sharp_crops.csv` from there.

Each Start creates its own run dir. Re-runs never overwrite earlier runs.

## The pipeline

Default mode is **Global** (single calibrated crop size across all folders).
Three phases plus a focus-classification pass at the end:

```
Phase 1 — Analyse droplets         (parallel, all cores by default)
   ↓ per-droplet droplet & sphere geometry
   ↓ produces (y_top, y_bottom, y_sphere, cx) per camera

Phase 2 — Calibrate crop size      (instant, single pass on Phase 1 output)
   ↓ Nth percentile of allowed heights across all droplets (default 5th)
   ↓ produces one cnn_size used for every output

Phase 3 — Generate outputs         (parallel)
   ↓ reload frame → crop → flatten via Calibration → save PNG
   ↓ compute six legacy focus metrics → write per-folder summary CSV

Focus classification               (single-process, runs after the pipeline)
   ↓ measure ERF blur sigma on every saved crop
   ↓ classify per folder + per camera at 25th/75th percentile
   ↓ copy sharp crops into Focus/{material}/{cam}/
   ↓ write Focus/sharp_crops.csv (the Training contract)
```

**Per-folder mode** runs the same three phases but calibrates a separate
crop size for each folder. Use it when different folders were captured with
different optical setups; the GUI greys out Global mode automatically when
the input is a single folder.

**Quick test mode** processes only the first droplet of each folder and
generates the darkness curve + geometry overlay. Use it to confirm
detection works before committing to a full run.

## GUI fields

The main screen has five config groups, plus the Run label entry next to
Start.

### Execution mode

| Field | What it does |
|---|---|
| Quick test (1st droplet/folder) | Smoke-test detection; produces overlay + darkness PNGs only. Disables every other field except Safe mode. |
| Full pipeline | Real run. All three phases. |
| Parallel cores | Worker count for Phases 1 and 3. Defaults to all detected cores. |
| Safe mode (single process) | Forces sequential processing for debugging. Ignored in Quick test mode. |

### Sampling

`Every Nth droplet` — controls how aggressively to subsample. A folder of
1500 droplets at step=10 yields 150. Use 1 for small datasets where you
want everything; 5 or 10 for production runs.

### Calibration

| Field | When to use |
|---|---|
| Per-folder crop | Each folder gets its own crop size. Required when input is a single folder (Global is greyed out). |
| Global crop | One crop size across all folders. Cleaner training data when folders share an optical setup. Only enabled when the input has multiple subfolders. |

### Outputs

| Field | What it does |
|---|---|
| Crops only (fastest) | Skips overlay and darkness plots — just the crops + summary CSVs. |
| All plots | Adds `*_overlay.png` (geometry annotation) and `*_darkness.png` (darkness curve with selected frame marked) per droplet. ~3× slower. |
| Enable profiling | Writes `profiling.json` in the run dir with per-phase timings. |

### Frame Selection

| Field | What it does |
|---|---|
| Use darkness weighting (4× slower) | OFF by default. ON adds the dark-fraction curve to the best-frame scoring. The submitted dissertation reports the OFF (geometry-only) path; ~86% of best-frame picks agree between the two modes. |

### Run label

Auto-fills from the basename of your CINE root. Edit freely. Becomes the
suffix of the run dir name; the timestamp is prepended automatically.

## Outputs

Every Start creates a fresh run dir:

```
Preprocessing/output/runs/2026-04-29_153022_4mm-borosilicate/
├── run_metadata.yaml             # settings + outcomes for this run
├── Focus/                        # what Training reads
│   ├── sharp_crops.csv
│   ├── focus_classified_all.csv
│   ├── focus_folder_stats.csv
│   ├── focus_classification_summary.png
│   └── {material}/{cam}/*.png    # sharp crops, one folder per camera
├── {material}/                   # full output, all crops (sharp+medium+blurry)
│   ├── {material}_summary.csv
│   ├── g/crops/sphere0843g_crop.png
│   ├── v/crops/...
│   └── m/crops/...
├── flatten_failures.log          # only if any flatten attempts failed
└── profiling.json                # only if profiling was on
```

**The legacy `Preprocessing/OUTPUT/` directory is untouched** by this layout.
If you have historical runs there, they keep working as-is; new Starts go
to `output/runs/`.

### `Focus/sharp_crops.csv` columns

| Column | Source | Used by |
|---|---|---|
| `filename` | crop basename | Training (lookup key) |
| `crop_path` | absolute path on disk | Training |
| `camera` | `g` / `v` / `m` | Training (per-camera filtering) |
| `droplet_id` | numeric ID parsed from `.cine` filename | reference |
| `cine_file` | source `.cine` filename | reference |
| `best_frame` | frame index chosen | reference |
| `dark_fraction` | darkness at best frame (0 if geometry-only) | reference |
| `y_top`, `y_bottom`, `y_sphere` | detected geometry, pixels | reference |
| `crop_size_px` | crop dimension (square) | Training |
| `laplacian_var`, `tenengrad`, `tenengrad_var`, `brenner`, `norm_laplacian`, `energy_gradient` | six legacy focus metrics, kept for reference only | none |
| `erf_sigma` | ERF blur sigma — the actual classifier | reference |
| `focus_class` | `sharp` / `medium` / `blurry` | filtered to `sharp` only in this CSV |
| `diameter_px` | `y_bottom − y_top`, added by classifier | Training (synthetic-blur sizing) |
| `native_blur_sigma` | same as `erf_sigma`, added under the name Training expects | Training |

### `run_metadata.yaml` fields

| Key | What it captures |
|---|---|
| `label` | the user-supplied or auto-filled run label |
| `cine_root` | source `.cine` directory |
| `run_dir` | absolute path of this run dir |
| `started_at` / `finished_at` | ISO-8601 timestamps |
| `elapsed_seconds` | wall-clock duration |
| `mode` | `global` / `per-folder` / `quick-test` |
| `sampling_step` | the `Every Nth` value used |
| `use_darkness`, `full_output`, `safe_mode`, `profile` | all toggles as set on Start |
| `n_cores` | parallel-worker count |
| `n_folders`, `n_droplets_processed` | scope of the run (Global mode) |
| `calibrated_crop_size_px` | the single Global crop size (Global mode only) |
| `flatten_failures` | count of crops where sphere detection failed during flattening |

## Configuration

All knobs live in [preprocessing_config.yaml](preprocessing_config.yaml).
The GUI overrides `paths.cine_root` and `paths.output_root` at runtime;
everything else needs a YAML edit.

| Section | Key | Default | What it controls |
|---|---|---|---|
| `paths` | `cine_root` | `./data` | Default `.cine` source dir (overridden by GUI Browse) |
| `paths` | `output_root` | `./output` | Parent dir under which `runs/<timestamp>_<label>/` lives |
| `crop` | `max_cnn_size` | 512 | Hard cap on crop dimension |
| `crop` | `min_cnn_size` | 64 | Hard floor on crop dimension |
| `crop` | `safety_pixels` | 3 | Vertical margin subtracted from the droplet→sphere gap before crop sizing. Raise if crops include the sphere. |
| `sampling` | `cine_step` | 10 | Process every Nth droplet (overridden by GUI) |
| `geometry` | `min_area` | 50 | Minimum connected-component area (px) to count as a candidate droplet/sphere |
| `geometry` | `sphere_width_ratio` | 0.30 | Sphere candidate must span at least this fraction of frame width |
| `geometry` | `sphere_center_tolerance` | 0.35 | Sphere centre must lie within this fraction of frame centre |
| `best_frame` | `n_candidates` | 20 | Candidate frames considered per `.cine` (full-output / darkness mode) |
| `best_frame` | `darkness_threshold_percentile` | 70.0 | Darkness percentile above which frames qualify as candidates |
| `best_frame` | `darkness_weight` | 0.05 | Weight of darkness vs centring error in the best-frame score |
| `calibration` | `percentile` | 5.0 | Percentile of allowed heights used as the global crop size. 0 = guaranteed sphere exclusion at the cost of small crops. |
| `focus` | `enabled` | true | Compute the six legacy focus metrics (kept in CSV for reference) |
| `focus` | `primary_metric` | `laplacian_var` | Legacy field; ignored by the active ERF classifier |
| `focus` | `sharp_threshold`, `blur_threshold` | null | Legacy fields; ignored. ERF classifier always uses 25th/75th percentiles per folder+camera. |

## Caveats and gotchas

- **Filename contract.** Droplet grouping uses regex `(\d+)([gmv])$` on
  the `.cine` stem. Files matching `sphere0843g.cine`, `sphere0843v.cine`,
  `sphere0843m.cine` get grouped as droplet `0843` with cameras `g`, `v`,
  `m`. Anything else is silently skipped. The `t` suffix appears in
  `scale_lookup.py` defaults but is not in the regex.
- **Camera codes.** `g` green, `v` violet, `m` mono. Focus classification
  is independent per camera — each contributes its own sharpest 25%
  regardless of which camera saw the cleanest frames overall.
- **Sphere flattening is mandatory.** Every saved crop is post-processed by
  `Calibration.sphere_processing.flatten_sphere_crop`. When sphere detection
  fails inside the crop the unflattened crop is returned and the failure
  is logged to `flatten_failures.log`. A non-zero count is normal for the
  `m` camera, where the sphere is sometimes only partially in frame.
- **Focus classifier is ERF blur sigma.** Lower = sharper. The six
  Laplacian/Tenengrad/etc columns in the summary CSV are kept for
  reference only; the active classifier is `focus_classification.py`'s ERF
  pass.
- **Each Start = its own run dir.** Outputs within a run dir are
  destructively rewritten if you re-run on the same label within the same
  second (rare — the GUI retries after 1 second on collision). Prior runs
  are never touched.
- **Geometry-only is the default and faster.** "Use darkness weighting"
  adds the darkness curve to the scoring; the geometry-only fast path
  agrees with the darkness-weighted pick on ~86% of frames and runs ~4×
  faster. The submitted dissertation reports the geometry-only path.
- **Worker subprocesses on Windows.** The active run dir is threaded into
  spawned workers via the `CROPPING_RUN_ROOT` environment variable, set by
  the GUI on Start. Don't unset or override it mid-run.
- **Flat-name imports.** Internal modules use bare imports (`import config`,
  not `from . import config`). External callers (`Inference`) only safely
  import leaf modules without transitive flat-import dependencies:
  `cine_io`, `darkness_analysis`, `cropping`, `image_utils`,
  `geom_analysis`. Importing higher-level modules requires `Preprocessing/`
  on `sys.path` first (which `__main__.py` does).
- **`focus_analysis.py` is legacy.** It is a separate Laplacian-variance
  CLI that competes with the in-pipeline ERF classifier. Don't re-run it on
  fresh outputs; it produces a different `Focus/` content. Use the GUI's
  Start button.
- **`scales.csv` doesn't ship.** `scale_lookup.py` exists for filename →
  px/mm lookups but the CSV is not in the module directory. The default
  flatten path uses no scale; only ad-hoc retrofit scripts use it.

## File map

```
Pipeline core
  pipeline_global.py          3-phase pipeline with shared crop calibration
  pipeline_folder.py          same but per-folder calibration
  workers.py                  multiprocessing worker functions

Frame and droplet analysis
  cine_io.py                  .cine load + droplet-grouping regex
  darkness_analysis.py        best-frame selection (dark-weighted or geometry-only)
  geom_analysis.py            Otsu + connected-components droplet/sphere detection
  cropping.py                 sphere-guarded fixed-size crop
  image_utils.py              frame loaders and Otsu mask helpers

Crop sizing
  crop_calibration.py         Nth-percentile crop size across all droplets

Output writing
  output_writer.py            saves crops + flattens via Calibration + writes summary CSVs
  run_io.py                   timestamped run dirs + run_metadata.yaml

Focus classification
  crop_blur_measurement.py    ERF blur sigma per crop
  focus_classification.py     per-folder/per-camera classifier (writes Focus/*)
  focus_metrics.py            legacy Laplacian/Tenengrad metrics (reference only)
  focus_analysis.py           legacy standalone CLI (don't use)

Configuration
  config.py                   loaded module-level constants
  config_loader.py            YAML → typed dataclass
  preprocessing_config.yaml   knob source of truth

GUI and plumbing
  gui.py                      Tk app, landing + main screen
  __main__.py                 console-script entry, sys.path bootstrap
  parallel_utils.py           run_parallel + GUI progress markers
  phantom_utils.py            silent pyphantom import
  phantom_silence.py          FD-level stdout/stderr suppression for the SDK banner

Plotting and diagnostics
  plotting.py                 darkness curves + geometry overlays
  profiling.py                timers and per-phase aggregation
  logging_config.py           logger setup

Ad-hoc retrofit scripts (one-shot, not part of the pipeline)
  flatten_existing_crops.py     re-flatten an old OUTPUT directory
  add_diameter_to_sharp_crops.py    backfill diameter_px column into an old sharp_crops.csv
  scale_lookup.py                   filename → px/mm lookups (needs scales.csv)
```

## Troubleshooting

**Phantom SDK not installed.** `cropping-preprocess` shows "(SDK NOT FOUND!)"
in the header and refuses to process `.cine` files. Install pyphantom — see
the repo-root [INSTALL.md](../INSTALL.md). If you only need to feed an
already-cropped dataset to the model, use Inference's `Precropped PNG` mode
instead and skip Preprocessing entirely.

**`flatten_failures.log` is huge.** Flatten depends on
`Calibration.sphere_processing.find_sphere_center`. If the sphere isn't
detected inside your crop, the unflattened crop is saved and a failure is
logged. Check sphere visibility in your `.cine` files. Common on the `m`
camera where the sphere is partially out of frame.

**No `.cine` files found.** Filenames must match `sphereNNNN[g|v|m].cine`
(case-insensitive). Anything else is silently dropped by the
droplet-grouping regex. Rename your files or extend the regex in
[cine_io.py](cine_io.py).

**Crops include the sphere.** Drop `calibration.percentile` toward 0 (1.0
or 0.0) to use a smaller, safer crop size, or raise `crop.safety_pixels`
to widen the gap budget. See the Configuration table.

**GUI hangs / "not responding".** The pipeline runs in a background thread.
If unresponsive for more than 30s, check the in-GUI log box for stack
traces — SDK import errors don't always pop dialog boxes.

## Where it sits in the pipeline

**Upstream:** nothing. This is stage 1 — input is raw `.cine` files
straight off the camera.

**Imports from elsewhere in this repo:**
- `Calibration.sphere_processing.flatten_sphere_crop` (mandatory, every
  saved crop is flattened)

**Downstream consumers:**
- **Training** ([Training/training_gui.py](../Training/training_gui.py))
  auto-discovers `sharp_crops.csv` by walking up parent directories from
  any pointed-at folder and selecting the most recently modified match.
  Point the Training GUI at any new run dir's `Focus/` and it picks up the
  CSV.
- **Inference** ([Inference/inference_engine.py](../Inference/inference_engine.py))
  reuses `cine_io`, `darkness_analysis`, `cropping`, `geom_analysis`, and
  `image_utils` as code modules at runtime — but does not read
  Preprocessing's output files. Inference processes its own input data
  (either `.cine` recordings or pre-cropped PNGs).
