"""Inference main window — composition root.

Holds canonical application state and wires together the modular widgets
in ``Inference/widgets/``. Imports ``InferenceEngine`` from the
``Inference`` package without modification.

Launch any of these ways — they all work:
  - F5 this file directly in your IDE
  - ``python -m Inference.inference_gui`` from CROPPING root
  - ``python Inference/inference_gui.py`` from CROPPING root

The single sys.path bootstrap below is the only sys.path manipulation
in the entire Inference package — every other module / widget / test
uses clean absolute imports. The bootstrap only fires when CROPPING
isn't already importable (i.e. F5 / direct-run), and is a no-op when
the project is properly on the import path (e.g. via ``pip install -e .``).
"""

from __future__ import annotations

# ── F5 / direct-run import bootstrap (single source of sys.path manipulation) ─
# Adds the CROPPING repo root to sys.path so the absolute imports below
# resolve when this file is executed directly. Importing this module from
# an installed package or from CROPPING-root cwd skips the insert (the
# path is already there) — so it's truly a no-op in those cases.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, Optional

import cv2
import numpy as np


def _samepath(a: str, b: str) -> bool:
    """Robust path comparison. Handles Windows case / normalisation /
    forward-vs-backward slash mismatches between values stored in
    settings and values returned from Path.glob()."""
    if not a or not b:
        return False
    try:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(
            os.path.abspath(b))
    except Exception:
        return a == b

from Inference.inference_engine import InferenceEngine
from Preprocessing.geom_analysis import analyze_frame_geometric
from Preprocessing.image_utils import load_frame_gray

from Inference.auto_preprocess import resolve_mode
from Inference.geometry_overlay import draw_geometry_overlay
from Inference.run_io import (
    DEFAULT_RUN_ROOT_REL,
    SaveOptions,
    append_csv_row,
    batch_folder_for,
    build_csv_row,
    build_run_metadata,
    init_batch_run,
    save_image,
    save_single_cine,
    session_folder_for,
    write_bounds_flag_plot,
    write_session_summary,
    write_yaml,
    CSV_COLUMNS,
)
from Inference.widgets.action_bar import ActionBar
from Inference.widgets.active_mode_strip import (
    ActiveModeStrip, process_button_label,
)
from Inference.widgets.frame_canvas import FrameCanvas
from Inference.widgets.frame_slider import FrameSlider
from Inference.widgets.locked_banner import LockedBanner
from Inference.widgets.preview_row import PreviewRow
from Inference.widgets.result_panel import ResultPanel
from Inference.widgets.save_dialog import SaveDialog
from Inference.widgets.session_indicator import SessionIndicator
from Inference.widgets.settings_dialog import SettingsDialogV2
from Inference.widgets.status_bar import StatusBar
from Inference.widgets.view_mode_toggle import (
    MODE_CROP, MODE_OVERLAY, MODE_PROCESSED, MODE_RAW, ViewModeToggle,
)
from Inference.widgets.input_mode_tabs import (
    InputModeTabs, MODE_CINE, MODE_PNG,
)


_THIS_DIR = Path(__file__).resolve().parent


SETTINGS_FILE = _THIS_DIR / "inference_settings.json"

DEFAULT_SETTINGS: Dict[str, Any] = {
    "model_path": "",
    # Input mode + per-mode paths
    "input_mode": "cine",        # "cine" | "png"
    "last_cine": "",
    "watch_folder": "",
    "png_folder": "",
    "last_png": "",
    "auto_detect": False,
    # Deployment-specific (the only field the user must really set)
    "s_c": 89.55555555555556,        # default to s_calib value of the user's
                                       # current camera; correct value gets
                                       # populated from checkpoint on load
    # Preprocessing (Auto by default; sticky-for-session if user overrides)
    "flatten_mode": "auto",
    "inner_margin_px": 20,
    "feather_px": 40,
    "crop_size": 299,
    "device": "cpu",
}


def load_settings() -> Dict[str, Any]:
    """Load V2 settings, falling back to defaults if file missing/invalid."""
    settings = dict(DEFAULT_SETTINGS)
    if SETTINGS_FILE.is_file():
        try:
            with open(SETTINGS_FILE, "r") as f:
                saved = json.load(f)
            settings.update(saved)
        except Exception as e:
            import logging
            logging.warning(f"V2 settings load failed ({e}); using defaults.")
    return settings


def save_settings(settings: Dict[str, Any]) -> None:
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass


class InferenceApp(tk.Tk):
    """Composition root. Holds state, instantiates widgets, wires events."""

    def __init__(self):
        super().__init__()
        self.title("Defocus Inference Tool — V2")
        self.geometry("1200x800")

        # ── Application state ─────────────────────────────────────────
        self.settings: Dict[str, Any] = load_settings()
        self.engine: Optional[InferenceEngine] = None
        self.current_crop = None              # np.ndarray | None
        self.current_results: Optional[Dict[str, Any]] = None
        self.auto_decision = None             # AutoDecision | None
        self._cine_obj = None
        self._cine_range = (0, 0)
        # Frame view state (Tab 2 — set when a cine opens)
        self.current_best_frame_idx: Optional[int] = None
        self.current_geo: Optional[Dict[str, Any]] = None
        self.current_frame_gray = None         # np.ndarray | None
        self.current_view_mode: str = "overlay"
        # Watch-folder polling state
        self._last_auto_mtime: Optional[float] = None
        self._watch_poll_after_id: Optional[str] = None
        # Save / session state
        self._run_root: Path = Path.cwd() / DEFAULT_RUN_ROOT_REL
        self._current_session_dir: Optional[Path] = None
        self._current_session_count: int = 0
        # Last-used SaveOptions (sticky between saves)
        last_so = self.settings.get("last_save_options")
        self._last_save_options = SaveOptions.from_dict(last_so)

        # ── Build placeholder regions (real widgets land here in later steps)
        self._build_layout()

        # Save settings on close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self) -> None:
        """Top-down layout: Locked Banner | Preview Row | Canvas | Action Bar | Status Bar."""
        # Tier 1: locked banner (model + calibration + s_c)
        self.banner = LockedBanner(
            self,
            on_browse=self._browse_model,
            on_load=self._load_model,
            on_s_c_change=self._set_s_c,
        )
        self.banner.pack(fill="x", padx=8, pady=(8, 4))
        # Initial state — no model loaded yet, but populate s_c from settings
        self.banner.update_state(
            engine=None,
            model_path=self.settings.get("model_path", ""),
            s_c_setting=float(self.settings.get("s_c", 0.0) or 0.0),
        )

        # Tier 1.5: input mode tabs (.cine input  |  Precropped PNG)
        # The .cine tab embeds the watch folder + cine file picker.
        # The PNG tab has folder + auto + current PNG name display.
        self.input_tabs = InputModeTabs(
            self,
            on_mode_change=self._handle_input_mode_change,
            on_cine_browse=self._browse_cine_file,
            on_cine_open=self._open_selected_cine,
            on_cine_watch_browse=self._browse_watch_folder,
            on_cine_auto_toggle=self._toggle_cine_auto,
            on_cine_watch_change=lambda _p: self._refresh_watch_position(),
            on_png_browse=self._browse_png_folder,
            on_png_auto_toggle=self._toggle_png_auto,
            on_png_folder_change=lambda _p: self._refresh_png_position(),
        )
        self.input_tabs.pack(fill="x", padx=8, pady=(0, 2))
        self.input_tabs.set_cine_path(self.settings.get("last_cine", ""))
        self.input_tabs.set_watch_folder(self.settings.get("watch_folder", ""))
        self.input_tabs.set_png_folder(self.settings.get("png_folder", ""))
        self.input_tabs.set_mode(self.settings.get("input_mode", MODE_CINE),
                                 fire_callback=False)

        # ── Tabs (Notebook) ────────────────────────────────────────────
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=4)

        # Tab 1: Preprocessing — preview row + thumbs comparison
        self.tab_preprocess = ttk.Frame(self.notebook, padding=4)
        self.notebook.add(self.tab_preprocess, text="Preprocessing")
        self.preview = PreviewRow(
            self.tab_preprocess,
            on_override_change=self._set_flatten_mode,
            on_view_full_size=self._jump_to_processed_full_size)
        self.preview.pack(fill="both", expand=True, padx=4, pady=4)
        self.preview.set_override_setting(
            self.settings.get("flatten_mode", "auto"))

        # Tab 2: Frame view — view toggle + canvas + frame slider
        self.tab_frame_view = ttk.Frame(self.notebook, padding=4)
        self.notebook.add(self.tab_frame_view, text="Frame view")
        self.view_toggle = ViewModeToggle(
            self.tab_frame_view,
            on_change=self._set_view_mode,
            initial=MODE_OVERLAY,
        )
        self.view_toggle.pack(fill="x", padx=4, pady=(2, 0))
        self.canvas = FrameCanvas(self.tab_frame_view)
        self.canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.frame_slider = FrameSlider(
            self.tab_frame_view, on_change=self._set_frame_idx)
        self.frame_slider.pack(fill="x", padx=4, pady=(0, 2))
        # Until a cine is open, all view modes (except none) are disabled
        for m in (MODE_RAW, MODE_OVERLAY, MODE_CROP, MODE_PROCESSED):
            self.view_toggle.set_enabled(m, False)
        self.frame_slider.set_enabled(False)

        # ── Persistent strips (visible regardless of which tab) ─────────
        # Result panel
        self.result_panel = ResultPanel(self)
        self.result_panel.pack(fill="x", padx=8, pady=(2, 2))

        # Active mode strip
        self.active_mode_strip = ActiveModeStrip(self)
        self.active_mode_strip.pack(fill="x", padx=8, pady=(0, 2))

        # Session indicator
        self.session_indicator = SessionIndicator(
            self, on_view_folder=self._open_session_folder)
        self.session_indicator.pack(fill="x", padx=8, pady=(0, 2))

        # Action bar
        self.action_bar = ActionBar(
            self,
            on_open=self._open_cine_dialog,
            on_process=self._process_current,
            on_save=self._save_current,
            on_batch=self._batch_folder,
            on_prev=lambda: self._step_input(-1),
            on_next=lambda: self._step_input(+1),
            on_settings=self._open_settings,
        )
        self.action_bar.pack(fill="x", padx=8, pady=4)
        # Initial state — defer to the consolidated refresh helper

        # Status bar
        self.status = StatusBar(self)
        self.status.pack(fill="x", padx=8, pady=(2, 8))
        self.status.set_status("V2 ready — browse for a model checkpoint to begin.")
        self.status.set_banner("warn", "No model loaded yet.")
        self.status_var = self.status._status_var

        # Initial active-mode strip state
        self._refresh_active_mode_strip()

    # ── Model load flow ──────────────────────────────────────────────
    def _browse_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Select checkpoint (.pth)",
            filetypes=[("PyTorch checkpoints", "*.pth"), ("All files", "*.*")],
        )
        if path:
            self.settings["model_path"] = path
            self.banner.update_state(
                engine=self.engine,
                model_path=path,
                s_c_setting=float(self.settings.get("s_c", 0.0) or 0.0),
            )

    def _load_model(self) -> None:
        model_path = self.settings.get("model_path", "")
        if not model_path or not Path(model_path).is_file():
            messagebox.showwarning("No model", "Browse for a checkpoint file first.")
            return
        self.status_var.set("Loading model…")
        self.update_idletasks()
        try:
            # Build a settings dict the engine understands. V1 expects rho /
            # sigma_0 / s_calib to be present (they get overridden from the
            # checkpoint anyway when calibration_model is baked in).
            engine_settings = dict(self.settings)
            # Provide harmless defaults for engine fields V2 doesn't track
            engine_settings.setdefault("rho", 1.0)
            engine_settings.setdefault("sigma_0", 0.0)
            engine_settings.setdefault("s_calib", float(engine_settings.get("s_c", 1.0)))
            engine_settings.setdefault("rho_std", 0.0)
            engine_settings.setdefault("sigma_0_std", 0.0)
            self.engine = InferenceEngine(engine_settings)
            msg = self.engine.load_model()
            # Engine has now (possibly) populated rho/sigma_0/s_calib from
            # the checkpoint. Sync those back into our settings so they
            # persist and remain consistent.
            for key in ("rho", "sigma_0", "s_calib", "rho_std", "sigma_0_std"):
                if key in self.engine.settings:
                    self.settings[key] = self.engine.settings[key]
            # Update UI
            self.banner.update_state(
                engine=self.engine,
                model_path=model_path,
                s_c_setting=float(self.settings.get("s_c", 0.0) or 0.0),
            )
            self.status.set_status(msg.split("\n")[0])
            self.status.set_banner("ok", "Model loaded — open a .cine to begin.")
            self._refresh_action_bar()
            self._refresh_active_mode_strip()
            # Restore last cine if there was one
            last = self.settings.get("last_cine", "")
            if last and Path(last).is_file():
                # Defer so the window mapping completes first
                self.after(50, lambda: self._open_cine(Path(last)))
        except Exception as e:
            self.engine = None
            self.banner.update_state(
                engine=None,
                model_path=model_path,
                s_c_setting=float(self.settings.get("s_c", 0.0) or 0.0),
            )
            messagebox.showerror("Model Load Error", str(e))
            self.status_var.set(f"Model load failed: {e}")

    def _set_s_c(self, value: float) -> None:
        """Called by LockedBanner when user commits a new s_c value."""
        self.settings["s_c"] = float(value)
        if self.engine is not None:
            self.engine.settings["s_c"] = float(value)
        save_settings(self.settings)

    def _set_flatten_mode(self, mode: str) -> None:
        """Called by PreviewRow when user changes the override dropdown.

        Sticky-for-session: persists in settings until user switches again
        or selects 'auto' to resume per-cine heuristic.
        """
        self.settings["flatten_mode"] = mode
        if self.engine is not None:
            self.engine.settings["flatten_mode"] = mode
        save_settings(self.settings)
        if self.current_crop is not None and self.engine is not None:
            self.auto_decision = self.preview.update_with_crop(
                self.current_crop, self.engine, mode)
        self._refresh_active_mode_strip()

    def _refresh_active_mode_strip(self) -> None:
        """Update the active-mode strip and Process button label.

        Reads the current effective mode from auto_decision (if a crop
        is loaded) or settings (the user's override choice).
        """
        # No engine → not ready
        if self.engine is None:
            self.active_mode_strip.clear()
            self.action_bar.set_process_label("Process current")
            return
        setting = self.settings.get("flatten_mode", "auto")
        # Effective mode comes from auto_decision when available; otherwise
        # echo the setting directly
        if self.auto_decision is not None:
            effective_mode = self.auto_decision.mode
            source = ("user override" if setting != "auto" else "auto")
        else:
            effective_mode = setting if setting != "auto" else "(awaiting crop)"
            source = "user override" if setting != "auto" else "auto"
        self.active_mode_strip.set_mode(effective_mode, source=source)
        self.action_bar.set_process_label(
            process_button_label(effective_mode))

    # ── Cine open / process flow ─────────────────────────────────────
    def _open_cine_dialog(self) -> None:
        path = filedialog.askopenfilename(
            title="Open .cine file",
            filetypes=[("Cine recordings", "*.cine"), ("All files", "*.*")],
        )
        if path:
            self.settings["last_cine"] = path
            save_settings(self.settings)
            self._open_cine(Path(path))

    def _open_cine(self, cine_path: Path) -> None:
        """Load a cine, show its first frame, run extract+preprocess for preview."""
        if self.engine is None:
            messagebox.showwarning("No model", "Load a model first.")
            return
        self.status.set_status(f"Opening {cine_path.name}…")
        self.update_idletasks()
        try:
            cine_obj, frame_idx, geo, frame_gray = self.engine.select_best_frame(cine_path)
            self._cine_obj = cine_obj
            self.canvas.set_image(frame_gray, reset_view=True)
            crop = self.engine.extract_crop(frame_gray, geo)
            self.current_crop = crop
            self.auto_decision = self.preview.update_with_crop(
                crop, self.engine, self.settings.get("flatten_mode", "auto"))
            self.preview.update_with_results(None)
            self.status.set_status(
                f"Loaded {cine_path.name}  |  best frame {frame_idx}  |  "
                f"auto-pick: {self.auto_decision.mode}")
            self.status.set_banner("ok", "Ready to process.")
            self.settings["last_cine"] = str(cine_path)
            save_settings(self.settings)
            self._refresh_watch_position()
            self._refresh_active_mode_strip()
            self.result_panel.clear()
            # Cache best-frame info for Tab 2's frame view
            self.current_best_frame_idx = frame_idx
            self.current_geo = geo
            self.current_frame_gray = frame_gray
            # Configure Tab 2 frame slider + enable all view modes
            try:
                first, last = cine_obj.range
                self.frame_slider.set_range(first, last, best=frame_idx)
            except Exception:
                pass
            for m in (MODE_RAW, MODE_OVERLAY, MODE_CROP, MODE_PROCESSED):
                self.view_toggle.set_enabled(m, True)
            self.view_toggle.set_mode(MODE_OVERLAY, fire_callback=False)
            self.current_view_mode = MODE_OVERLAY
            self.frame_slider.set_enabled(True)
            self._render_current_view()
            self.current_results = None
            self._refresh_action_bar()
            self._refresh_session_state()
        except Exception as e:
            self.status.set_status(f"Open failed: {e}")
            self.status.set_banner("err", f"Could not open {cine_path.name}: {e}")
            messagebox.showerror("Open Error", str(e))

    def _process_current(self) -> None:
        """Run inference on the loaded cine in a background thread."""
        if self.engine is None or self.current_crop is None:
            messagebox.showwarning("Not ready",
                                     "Load a model and open a cine first.")
            return
        self.status.set_status("Processing…")
        self.status.start_progress()
        self.update_idletasks()
        # Background thread to keep UI responsive
        import threading
        def worker():
            try:
                # Use the auto-decided mode the preview already chose
                mode = (self.auto_decision.mode if self.auto_decision
                         else self.settings.get("flatten_mode", "auto"))
                # Push that mode into engine settings just for this call
                prev_mode = self.engine.settings.get("flatten_mode")
                self.engine.settings["flatten_mode"] = mode
                try:
                    norm_img, tensor_input = self.engine.preprocess_crop(
                        self.current_crop)
                    native_size = max(self.current_crop.shape[0],
                                        self.current_crop.shape[1])
                    results = self.engine.run_inference(
                        tensor_input, native_size)
                finally:
                    if prev_mode is not None:
                        self.engine.settings["flatten_mode"] = prev_mode
                self.after(0, self._on_process_done, results)
            except Exception as e:
                self.after(0, self._on_process_error, str(e))
        threading.Thread(target=worker, daemon=True).start()

    def _on_process_done(self, results: Dict[str, Any]) -> None:
        self.current_results = results
        self.preview.update_with_results(results)
        self.result_panel.update(results)
        self.status.set_status(
            f"Done. z = {results.get('defocus_mm', 0):.3f} mm  "
            f"[{results.get('bounds_flag', 'IN_RANGE')}]")
        self.status.stop_progress()
        # Enable Save now that there's a result
        self._refresh_action_bar()
        self._refresh_session_state()

    def _on_process_error(self, msg: str) -> None:
        self.status.set_status(f"Process failed: {msg}")
        self.status.stop_progress()
        self.status.set_banner("err", f"Inference failed: {msg}")
        messagebox.showerror("Inference Error", msg)

    def _batch_folder(self) -> None:
        """Pick folder, ask save options, then run batch in a thread.
        Mode-aware: globs .cine in cine mode, .png in PNG mode."""
        if self.engine is None:
            messagebox.showwarning("No model", "Load a model first.")
            return
        png_mode = self._is_png_mode()
        title = ("Select folder of cropped PNGs" if png_mode
                  else "Select folder of .cine files")
        folder = filedialog.askdirectory(title=title)
        if not folder:
            return
        folder_path = Path(folder)
        if png_mode:
            cine_files = sorted(
                list(folder_path.glob("*.png"))
                + list(folder_path.glob("*.PNG")),
                key=lambda p: p.name.lower())
            if not cine_files:
                messagebox.showinfo("Empty",
                                       "No PNG files in that folder.")
                return
        else:
            cine_files = sorted(folder_path.glob("*.cine"))
            if not cine_files:
                messagebox.showinfo("Empty",
                                       "No .cine files in that folder.")
                return
        # Pre-flight: ask what to save per cine
        target = batch_folder_for(folder_path, self._run_root)
        dlg = SaveDialog(
            self, defaults=self._last_save_options,
            target_folder=target, context="batch",
            input_mode=self.input_tabs.get_mode())
        if dlg.result is None:
            return
        opts = dlg.result
        self._last_save_options = opts
        self.settings["last_save_options"] = opts.to_dict()
        save_settings(self.settings)
        # Initialise batch run folder + write metadata
        try:
            batch_dir = init_batch_run(
                folder_path=folder_path, engine=self.engine,
                settings=self.settings, save_options=opts,
                run_root=self._run_root)
        except Exception as e:
            messagebox.showerror("Batch init failed", str(e))
            return
        self.status.set_status(
            f"Batch processing {len(cine_files)} cines into {batch_dir.name}…")
        self.status.start_progress()
        self.update_idletasks()
        import threading
        def worker():
            results = []
            errors = 0
            for i, cine in enumerate(cine_files):
                try:
                    self.after(0, self._on_batch_progress,
                               f"[{i+1}/{len(cine_files)}] {cine.name}",
                               i / max(1, len(cine_files)))
                    res = self._batch_process_one(
                        cine_path=cine, batch_dir=batch_dir,
                        save_options=opts)
                    if res is not None:
                        results.append(res)
                except Exception as e:
                    errors += 1
                    print(f"  ERROR on {cine.name}: {e}")
            self.after(0, self._on_batch_done, results, batch_dir, errors)
        threading.Thread(target=worker, daemon=True).start()

    def _batch_process_one(
        self, cine_path: Path, batch_dir: Path,
        save_options: SaveOptions,
    ) -> Optional[Dict[str, Any]]:
        """Process one cine OR png inside a batch loop; write outputs
        into batch_dir/per_cine/<stem>/. Returns the CSV row dict.

        Mode-aware: in PNG mode, treats the input as already-cropped
        and skips the cine select_best_frame + extract_crop steps.
        """
        import datetime as _dt
        when = _dt.datetime.now()
        if self._is_png_mode():
            # PNG mode: load the PNG directly as the crop
            crop = cv2.imread(str(cine_path), cv2.IMREAD_GRAYSCALE)
            if crop is None:
                raise RuntimeError(f"could not load {cine_path.name}")
            # Run preprocess + inference manually (no cine pipeline)
            mode = self.settings.get("flatten_mode", "auto")
            if mode == "auto":
                from Inference.auto_preprocess import resolve_mode as _rm
                effective = _rm("auto", crop).mode
            else:
                effective = mode
            prev_mode = self.engine.settings.get("flatten_mode")
            self.engine.settings["flatten_mode"] = effective
            try:
                norm_img, tensor = self.engine.preprocess_crop(crop)
                native_size = max(crop.shape[0], crop.shape[1])
                inf = self.engine.run_inference(tensor, native_size)
            finally:
                if prev_mode is not None:
                    self.engine.settings["flatten_mode"] = prev_mode
            # Wrap into a dict shaped like process_cine's output
            results = {**inf, "crop": crop, "norm_img": norm_img}
        else:
            # Engine's high-level pipeline gives us crop + norm_img + results
            results = self.engine.process_cine(cine_path)
        crop = results.get("crop")
        # Auto-decide for this cine using the same heuristic
        if self.settings.get("flatten_mode", "auto") == "auto" and crop is not None:
            decision = resolve_mode("auto", crop)
        else:
            decision = resolve_mode(
                self.settings.get("flatten_mode", "auto"), crop)
        # Per-cine subfolder
        per_dir = batch_dir / "per_cine" / cine_path.stem
        per_dir.mkdir(parents=True, exist_ok=True)
        # Save images per options
        if save_options.raw_frame and "frame_gray" in results:
            save_image(per_dir / "raw_frame.png", results["frame_gray"])
        if save_options.overlay and "frame_gray" in results:
            ovl = draw_geometry_overlay(
                results["frame_gray"].astype(np.uint8)
                if results["frame_gray"].dtype != np.uint8
                else results["frame_gray"],
                geo=results.get("geometry"),
                crop_size=int(self.settings.get("crop_size", 299)),
                frame_idx=results.get("frame_idx"),
                best_idx=results.get("frame_idx"),
            )
            save_image(per_dir / "overlay.png", ovl)
        if save_options.crop and crop is not None:
            save_image(per_dir / "crop.png", crop)
        if save_options.processed and "norm_img" in results:
            save_image(per_dir / "processed.png", results["norm_img"])
        # Append CSV row
        row = build_csv_row(
            when=when, cine_path=cine_path, results=results,
            auto_decision=decision, save_options=save_options,
            engine=self.engine, settings=self.settings,
        )
        append_csv_row(batch_dir / "results.csv", row, CSV_COLUMNS)
        return row

    def _on_batch_progress(self, msg: str, frac: float) -> None:
        self.status.set_status(msg)
        self.status.update_progress(frac)

    def _on_batch_done(self, results: list, batch_dir: Path,
                        errors: int) -> None:
        # Write the bounds-flag distribution plot for batch summary
        try:
            write_bounds_flag_plot(batch_dir)
        except Exception:
            pass
        msg = (f"Batch done — {len(results)} cines processed"
                + (f", {errors} errors" if errors else "") + ".")
        self.status.set_status(f"{msg}  Folder: {batch_dir.name}")
        self.status.set_banner("ok", f"Wrote: {batch_dir.name}")
        self.status.stop_progress()

    def _open_settings(self) -> None:
        SettingsDialogV2(
            self, self.settings, engine=self.engine,
            on_save=self._on_settings_saved)

    def _on_settings_saved(self, new_settings: Dict[str, Any]) -> None:
        self.settings.update(new_settings)
        if self.engine is not None:
            self.engine.settings.update(new_settings)
        save_settings(self.settings)
        # Refresh banner s_c display in case it changed
        self.banner.update_state(
            engine=self.engine,
            model_path=self.settings.get("model_path", ""),
            s_c_setting=float(self.settings.get("s_c", 0.0) or 0.0),
        )
        # Refresh preview if mode changed
        self.preview.set_override_setting(
            self.settings.get("flatten_mode", "auto"))
        if self.current_crop is not None and self.engine is not None:
            self.auto_decision = self.preview.update_with_crop(
                self.current_crop, self.engine,
                self.settings.get("flatten_mode", "auto"))
        self.status.set_status("Settings saved.")

    # ── Save: single cine ────────────────────────────────────────────
    def _save_current(self) -> None:
        """Open the SaveDialog, then write to the session folder."""
        if (self.current_results is None or self.current_crop is None
                or self.engine is None):
            messagebox.showwarning(
                "Nothing to save",
                "Click Process first — there's no current result.")
            return
        cine_path = self._current_input_path()
        if cine_path is None:
            messagebox.showerror(
                "Save error",
                "Current input path missing. Open a .cine or PNG first.")
            return
        target = session_folder_for(cine_path, self._run_root)
        dlg = SaveDialog(
            self, defaults=self._last_save_options,
            target_folder=target, context="single",
            cine_name=cine_path.name,
            input_mode=self.input_tabs.get_mode())
        if dlg.result is None:
            return
        opts = dlg.result
        # Persist as defaults for next time
        self._last_save_options = opts
        self.settings["last_save_options"] = opts.to_dict()
        save_settings(self.settings)
        # Render the saveable views from current state
        raw_frame = (self.current_frame_gray.astype(np.uint8)
                     if self.current_frame_gray is not None
                     and self.current_frame_gray.dtype != np.uint8
                     else self.current_frame_gray)
        overlay_img = None
        if opts.overlay and raw_frame is not None:
            overlay_img = draw_geometry_overlay(
                raw_frame, geo=self.current_geo,
                crop_size=int(self.settings.get("crop_size", 299)),
                frame_idx=self.current_best_frame_idx,
                best_idx=self.current_best_frame_idx,
            )
        processed_img = None
        if opts.processed and self.current_crop is not None:
            mode = (self.auto_decision.mode if self.auto_decision is not None
                    else self.settings.get("flatten_mode", "auto"))
            if mode == "auto":
                mode = resolve_mode("auto", self.current_crop).mode
            prev_mode = self.engine.settings.get("flatten_mode")
            self.engine.settings["flatten_mode"] = mode
            try:
                processed_img, _ = self.engine.preprocess_crop(self.current_crop)
            except Exception:
                processed_img = None
            finally:
                if prev_mode is not None:
                    self.engine.settings["flatten_mode"] = prev_mode
        try:
            written = save_single_cine(
                cine_path=cine_path,
                results=self.current_results,
                auto_decision=self.auto_decision,
                engine=self.engine, settings=self.settings,
                save_options=opts, run_root=self._run_root,
                raw_frame=raw_frame, overlay_frame=overlay_img,
                crop=self.current_crop, processed=processed_img,
            )
            self._refresh_session_state()
            self.status.set_status(f"Saved to: {written.name}")
            self.status.set_banner("ok",
                f"Saved into session: "
                f"{self._current_session_dir.name if self._current_session_dir else '(?)'}")
        except Exception as e:
            self.status.set_status(f"Save failed: {e}")
            messagebox.showerror("Save error", str(e))

    def _refresh_session_state(self) -> None:
        """Recompute and display the active session for the current input."""
        cine_path = self._current_input_path()
        if cine_path is None:
            self.session_indicator.set_state(None, 0)
            return
        target = session_folder_for(cine_path, self._run_root)
        if target.is_dir():
            csv_path = target / "results.csv"
            if csv_path.is_file():
                # Subtract 1 for the header line
                count = max(0, sum(1 for _ in open(csv_path)) - 1)
            else:
                count = 0
            self._current_session_dir = target
            self._current_session_count = count
            self.session_indicator.set_state(target, count)
        else:
            self._current_session_dir = None
            self._current_session_count = 0
            self.session_indicator.set_state(None, 0)

    def _open_session_folder(self) -> None:
        if self._current_session_dir is None:
            return
        try:
            import os
            os.startfile(str(self._current_session_dir))
        except Exception as e:
            self.status.set_status(f"Could not open folder: {e}")

    # ── Watch folder polling + Prev/Next ─────────────────────────────
    _WATCH_POLL_INTERVAL_MS = 10_000

    def _browse_watch_folder(self) -> None:
        path = filedialog.askdirectory(title="Select watch folder")
        if not path:
            return
        self.input_tabs.set_watch_folder(path)
        self.settings["watch_folder"] = path
        save_settings(self.settings)
        self._refresh_watch_position()

    def _toggle_cine_auto(self, enabled: bool) -> None:
        if enabled:
            folder = self.input_tabs.get_watch_folder()
            if not folder or not Path(folder).is_dir():
                messagebox.showwarning("Watch folder",
                                          "Set a valid watch folder first.")
                self.input_tabs.set_cine_auto(False)
                return
            latest = InferenceEngine.find_latest_cine(Path(folder))
            if latest is None:
                messagebox.showinfo("Watch folder", "No .cine files found.")
                self.input_tabs.set_cine_auto(False)
                return
            try:
                self._last_auto_mtime = latest.stat().st_mtime
            except OSError:
                self._last_auto_mtime = None
            if self.engine is not None:
                self._open_cine(latest)
            self.status.set_status(
                f"Auto-detected: {latest.name} (polling every "
                f"{self._WATCH_POLL_INTERVAL_MS // 1000}s)")
            self._schedule_watch_poll()
        else:
            self._last_auto_mtime = None
            # Next tick will see auto disabled and exit cleanly

    def _schedule_watch_poll(self) -> None:
        if self._watch_poll_after_id is not None:
            try:
                self.after_cancel(self._watch_poll_after_id)
            except Exception:
                pass
        self._watch_poll_after_id = self.after(
            self._WATCH_POLL_INTERVAL_MS, self._poll_watch_folder)

    def _poll_watch_folder(self) -> None:
        self._watch_poll_after_id = None
        if not self.input_tabs.is_cine_auto_enabled():
            return
        folder = self.input_tabs.get_watch_folder()
        if not folder or not Path(folder).is_dir():
            self.input_tabs.set_cine_auto(False)
            self.status.set_status("Watch folder gone — auto stopped.")
            return
        latest = InferenceEngine.find_latest_cine(Path(folder))
        if latest is None:
            self._schedule_watch_poll()
            return
        try:
            mtime = latest.stat().st_mtime
        except OSError:
            self._schedule_watch_poll()
            return
        if self._last_auto_mtime is None or mtime > self._last_auto_mtime:
            self._last_auto_mtime = mtime
            self.status.set_status(f"Auto-loaded: {latest.name}")
            if self.engine is not None:
                self._open_cine(latest)
        self._schedule_watch_poll()

    def _list_watch_cines(self) -> list:
        folder = self.input_tabs.get_watch_folder()
        if not folder or not Path(folder).is_dir():
            return []
        return sorted(Path(folder).glob("*.cine"),
                       key=lambda p: p.name.lower())

    def _refresh_watch_position(self) -> None:
        """Update only the position label. Button state is the
        responsibility of ``_refresh_action_bar`` below."""
        cines = self._list_watch_cines()
        if not cines:
            self.action_bar.set_position_text("")
        else:
            cur = self.settings.get("last_cine", "")
            try:
                idx = next(i for i, p in enumerate(cines) if _samepath(str(p), cur))
            except StopIteration:
                idx = -1
            self.action_bar.set_position_text(
                f"{idx + 1} / {len(cines)}" if idx >= 0
                else f"– / {len(cines)}")
        self._refresh_action_bar()

    def _refresh_action_bar(self) -> None:
        """Single source of truth for action-bar button enable/disable.

        Computes the FULL state (model loaded, input loaded, result
        available, prev/next enabled) and pushes one update_state call.
        Mode-aware: in .cine mode reads watch folder; in PNG mode reads
        PNG folder. Call this anywhere state changes — never call
        ``self.action_bar.update_state(...)`` directly with partial args.
        """
        if self._is_png_mode():
            items = self._list_png_folder()
            cur = self.settings.get("last_png", "")
        else:
            items = self._list_watch_cines()
            cur = self.settings.get("last_cine", "")
        prev_enabled = False
        next_enabled = False
        if items:
            try:
                idx = next(i for i, p in enumerate(items) if _samepath(str(p), cur))
            except StopIteration:
                idx = -1
            prev_enabled = (idx > 0 or idx < 0)
            next_enabled = (0 <= idx < len(items) - 1 or idx < 0)
        self.action_bar.update_state(
            model_loaded=self.engine is not None,
            cine_loaded=self.current_crop is not None,
            result_available=self.current_results is not None,
            prev_enabled=prev_enabled,
            next_enabled=next_enabled,
        )

    def _step_cine(self, direction: int) -> None:
        cines = self._list_watch_cines()
        if not cines or self.engine is None:
            return
        cur = self.settings.get("last_cine", "")
        try:
            idx = next(i for i, p in enumerate(cines) if _samepath(str(p), cur))
        except StopIteration:
            idx = 0 if direction > 0 else len(cines) - 1
            target = cines[idx]
        else:
            new_idx = idx + direction
            if new_idx < 0 or new_idx >= len(cines):
                return
            target = cines[new_idx]
        self.settings["last_cine"] = str(target)
        save_settings(self.settings)
        self._open_cine(target)
        self._refresh_watch_position()

    def _step_input(self, direction: int) -> None:
        """Mode-aware Prev/Next: dispatches to .cine or PNG stepping."""
        if self._is_png_mode():
            self._step_png(direction)
        else:
            self._step_cine(direction)

    def _current_input_path(self) -> Optional[Path]:
        """Mode-aware: path of the currently-loaded cine OR png."""
        key = "last_png" if self._is_png_mode() else "last_cine"
        s = self.settings.get(key, "")
        if not s:
            return None
        p = Path(s)
        return p if p.is_file() else None

    def _refresh_position_label(self) -> None:
        """Mode-aware position label refresh."""
        if self._is_png_mode():
            self._refresh_png_position()
        else:
            self._refresh_watch_position()

    # ── Input mode dispatch ─────────────────────────────────────────
    def _handle_input_mode_change(self, mode: str) -> None:
        """Called by InputModeTabs when the user clicks a mode tab."""
        self.settings["input_mode"] = mode
        save_settings(self.settings)
        # Wipe transient state so the wrong-mode crop doesn't linger
        self.current_crop = None
        self.current_results = None
        self.auto_decision = None
        self.current_frame_gray = None
        self.current_geo = None
        self.current_best_frame_idx = None
        self._cine_obj = None
        self.canvas.clear()
        self.preview.clear()
        self.result_panel.clear()
        # Mode-specific Tab 2 view modes
        self._refresh_view_mode_availability()
        # Refresh action bar (for prev/next + result_available)
        self._refresh_action_bar()
        # Refresh position label for the new mode
        if mode == MODE_CINE:
            self._refresh_watch_position()
        else:
            self._refresh_png_position()
        self.status.set_status(f"Switched to {mode} input mode.")

    def _is_png_mode(self) -> bool:
        return self.input_tabs.get_mode() == MODE_PNG

    # ── PNG-mode controls ────────────────────────────────────────────
    def _browse_cine_file(self) -> None:
        """Browse for a single .cine to load (.cine input tab)."""
        path = filedialog.askopenfilename(
            title="Open .cine file",
            filetypes=[("Cine recordings", "*.cine"), ("All files", "*.*")])
        if not path:
            return
        self.input_tabs.set_cine_path(path)
        self.settings["last_cine"] = path
        save_settings(self.settings)

    def _open_selected_cine(self) -> None:
        """Open whichever .cine is currently in the path field."""
        path_str = self.input_tabs.get_cine_path()
        if not path_str:
            messagebox.showwarning("No cine", "Browse for a .cine first.")
            return
        p = Path(path_str)
        if not p.is_file():
            messagebox.showerror("File not found", str(p))
            return
        self._open_cine(p)

    def _browse_png_folder(self) -> None:
        path = filedialog.askdirectory(title="Select folder of cropped PNGs")
        if not path:
            return
        self.input_tabs.set_png_folder(path)
        self.settings["png_folder"] = path
        save_settings(self.settings)
        self._refresh_png_position()
        # Auto-load the first PNG so the user sees something immediately
        pngs = self._list_png_folder()
        if pngs and self.engine is not None:
            self._open_png(pngs[0])

    def _toggle_png_auto(self, enabled: bool) -> None:
        if enabled:
            folder = self.input_tabs.get_png_folder()
            if not folder or not Path(folder).is_dir():
                messagebox.showwarning("PNG folder",
                                          "Set a valid PNG folder first.")
                self.input_tabs.set_png_auto(False)
                return
            pngs = self._list_png_folder()
            if not pngs:
                messagebox.showinfo("PNG folder", "No PNG files found.")
                self.input_tabs.set_png_auto(False)
                return
            latest = max(pngs, key=lambda p: p.stat().st_mtime)
            try:
                self._last_auto_mtime = latest.stat().st_mtime
            except OSError:
                self._last_auto_mtime = None
            if self.engine is not None:
                self._open_png(latest)
            self.status.set_status(
                f"Auto-detected PNG: {latest.name} (polling every "
                f"{self._WATCH_POLL_INTERVAL_MS // 1000}s)")
            self._schedule_png_poll()
        else:
            self._last_auto_mtime = None

    def _schedule_png_poll(self) -> None:
        if self._watch_poll_after_id is not None:
            try:
                self.after_cancel(self._watch_poll_after_id)
            except Exception:
                pass
        self._watch_poll_after_id = self.after(
            self._WATCH_POLL_INTERVAL_MS, self._poll_png_folder)

    def _poll_png_folder(self) -> None:
        self._watch_poll_after_id = None
        if not self.input_tabs.is_png_auto_enabled():
            return
        folder = self.input_tabs.get_png_folder()
        if not folder or not Path(folder).is_dir():
            self.input_tabs.set_png_auto(False)
            self.status.set_status("PNG folder gone — auto stopped.")
            return
        pngs = self._list_png_folder()
        if not pngs:
            self._schedule_png_poll()
            return
        latest = max(pngs, key=lambda p: p.stat().st_mtime)
        try:
            mtime = latest.stat().st_mtime
        except OSError:
            self._schedule_png_poll()
            return
        if self._last_auto_mtime is None or mtime > self._last_auto_mtime:
            self._last_auto_mtime = mtime
            self.status.set_status(f"Auto-loaded PNG: {latest.name}")
            if self.engine is not None:
                self._open_png(latest)
        self._schedule_png_poll()

    def _list_png_folder(self) -> list:
        folder = self.input_tabs.get_png_folder()
        if not folder or not Path(folder).is_dir():
            return []
        return sorted(
            list(Path(folder).glob("*.png")) + list(Path(folder).glob("*.PNG")),
            key=lambda p: p.name.lower())

    def _refresh_png_position(self) -> None:
        """Update position label for PNG mode + active button state."""
        pngs = self._list_png_folder()
        cur = self.settings.get("last_png", "")
        try:
            idx = next(i for i, p in enumerate(pngs) if _samepath(str(p), cur))
        except StopIteration:
            idx = -1
        if pngs:
            self.action_bar.set_position_text(
                f"{idx + 1} / {len(pngs)}" if idx >= 0
                else f"– / {len(pngs)}")
        else:
            self.action_bar.set_position_text("")
        self._refresh_action_bar()

    def _open_png(self, png_path: Path) -> None:
        """Load a PNG as the current crop and refresh preview."""
        if self.engine is None:
            messagebox.showwarning("No model", "Load a model first.")
            return
        self.status.set_status(f"Loading PNG {png_path.name}…")
        self.update_idletasks()
        try:
            img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError("Could not read image (cv2 returned None)")
            # In PNG mode, the PNG IS the crop
            self.current_crop = img
            self.current_frame_gray = img        # for Raw view in Tab 2
            self.current_geo = None
            self.current_best_frame_idx = None
            self._cine_obj = None
            self.input_tabs.set_current_png_label(png_path.name)
            # Run preview (auto-decide mode + render thumbnails)
            self.auto_decision = self.preview.update_with_crop(
                img, self.engine, self.settings.get("flatten_mode", "auto"))
            self.preview.update_with_results(None)
            # Tab 2: only Crop / Processed / Raw views are meaningful
            self.canvas.set_image(img, reset_view=True)
            self._refresh_view_mode_availability()
            self.view_toggle.set_mode(MODE_PROCESSED, fire_callback=True)
            # Persist + bookkeeping
            self.settings["last_png"] = str(png_path)
            save_settings(self.settings)
            self.current_results = None
            self._refresh_action_bar()
            self._refresh_session_state()
            self._refresh_active_mode_strip()
            self.status.set_status(
                f"Loaded {png_path.name}  |  auto-pick: "
                f"{self.auto_decision.mode}")
            self.status.set_banner("ok", "Ready to process.")
        except Exception as e:
            self.status.set_status(f"PNG load failed: {e}")
            self.status.set_banner("err", f"Could not load {png_path.name}: {e}")
            messagebox.showerror("Load error", str(e))

    def _step_png(self, direction: int) -> None:
        pngs = self._list_png_folder()
        if not pngs or self.engine is None:
            return
        cur = self.settings.get("last_png", "")
        try:
            idx = next(i for i, p in enumerate(pngs) if _samepath(str(p), cur))
        except StopIteration:
            idx = 0 if direction > 0 else len(pngs) - 1
            target = pngs[idx]
        else:
            new_idx = idx + direction
            if new_idx < 0 or new_idx >= len(pngs):
                return
            target = pngs[new_idx]
        self._open_png(target)
        self._refresh_png_position()

    # ── Mode-aware Tab 2 view-mode availability ──────────────────────
    def _refresh_view_mode_availability(self) -> None:
        """Adapt the view toggle for the current input mode.

        PNG mode disables Overlay (no source frame to annotate) and the
        slider (no multi-frame data).
        """
        if self._is_png_mode():
            for m in (MODE_RAW, MODE_CROP, MODE_PROCESSED):
                self.view_toggle.set_enabled(m, self.current_crop is not None)
            self.view_toggle.set_enabled(MODE_OVERLAY, False)
            self.frame_slider.set_enabled(False)
        else:
            for m in (MODE_RAW, MODE_OVERLAY, MODE_CROP, MODE_PROCESSED):
                self.view_toggle.set_enabled(m, self.current_crop is not None)
            self.frame_slider.set_enabled(self._cine_obj is not None)

    def _jump_to_processed_full_size(self) -> None:
        """Switch to Tab 2 and select the Processed view mode.

        Called by PreviewRow's 'View full size' link button.
        """
        try:
            self.notebook.select(self.tab_frame_view)
        except Exception:
            pass
        self.view_toggle.set_mode(MODE_PROCESSED, fire_callback=True)

    # ── Tab 2 frame view: view mode + slider ─────────────────────────
    def _set_view_mode(self, mode: str) -> None:
        """Called by ViewModeToggle when the user clicks a view button."""
        self.current_view_mode = mode
        # Slider is only meaningful for raw / overlay (frame-dependent)
        self.frame_slider.set_enabled(
            mode in (MODE_RAW, MODE_OVERLAY) and self._cine_obj is not None)
        self._render_current_view()

    def _set_frame_idx(self, idx: int) -> None:
        """Called by FrameSlider when the user scrubs.

        Loads the requested frame from the cine using V1's image_utils
        helper. For overlay mode, also recomputes geometry per frame so
        the overlay updates as the user scrubs (matches V1 behaviour).
        """
        if self._cine_obj is None:
            return
        try:
            self.current_frame_gray = load_frame_gray(self._cine_obj, idx)
        except Exception as e:
            self.status.set_status(f"Frame load error: {e}")
            return
        # In overlay mode, refresh per-frame geometry
        if self.current_view_mode == MODE_OVERLAY:
            try:
                self.current_geo = analyze_frame_geometric(
                    self._cine_obj, idx)
            except Exception:
                pass  # keep last geo
        self._render_current_view()

    def _render_current_view(self) -> None:
        """Render whatever view_mode is selected into the canvas."""
        mode = self.current_view_mode
        if mode == MODE_RAW:
            if self.current_frame_gray is not None:
                self.canvas.set_image(self.current_frame_gray, reset_view=False)
        elif mode == MODE_OVERLAY:
            if self.current_frame_gray is not None:
                # Use the engine's crop_size setting for the box
                cs = int(self.settings.get("crop_size", 299))
                idx_now = self.frame_slider.get_value()
                annotated = draw_geometry_overlay(
                    self.current_frame_gray.astype(np.uint8)
                    if self.current_frame_gray.dtype != np.uint8
                    else self.current_frame_gray,
                    geo=self.current_geo,
                    crop_size=cs,
                    frame_idx=idx_now,
                    best_idx=self.current_best_frame_idx,
                )
                self.canvas.set_image(annotated, reset_view=False)
        elif mode == MODE_CROP:
            if self.current_crop is not None:
                self.canvas.set_image(self.current_crop, reset_view=True)
        elif mode == MODE_PROCESSED:
            if self.current_crop is not None and self.engine is not None:
                # Compute the processed image using the active mode
                effective_mode = (self.auto_decision.mode
                                    if self.auto_decision is not None
                                    else self.settings.get("flatten_mode", "auto"))
                if effective_mode == "auto":
                    # Fall back to whatever auto would pick now
                    effective_mode = resolve_mode(
                        "auto", self.current_crop).mode
                prev_mode = self.engine.settings.get("flatten_mode")
                self.engine.settings["flatten_mode"] = effective_mode
                try:
                    norm_img, _ = self.engine.preprocess_crop(self.current_crop)
                    self.canvas.set_image(norm_img, reset_view=True)
                except Exception as e:
                    self.status.set_status(
                        f"Processed view failed: {e}")
                finally:
                    if prev_mode is not None:
                        self.engine.settings["flatten_mode"] = prev_mode

    def _on_close(self) -> None:
        save_settings(self.settings)
        self.destroy()


def main() -> None:
    app = InferenceApp()
    app.mainloop()


if __name__ == "__main__":
    main()
