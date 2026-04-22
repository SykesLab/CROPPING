"""
Defocus Estimation Inference GUI

Standalone tkinter application for lab use in high-speed shadowgraphy.
Loads a .cine recording, provides a frame slider to browse all frames,
and runs the full inference pipeline on the best (or selected) frame.

Usage:
    python inference_gui.py
"""

import csv
import json
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------


def _check_deps():
    missing = []
    for pkg, imp in [("numpy", "numpy"), ("opencv-python", "cv2"),
                     ("torch", "torch"), ("pyyaml", "yaml"),
                     ("Pillow", "PIL")]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("Missing dependencies: " + ", ".join(missing))
        print("Install with:  pip install " + " ".join(missing))
        sys.exit(1)


_check_deps()

from PIL import Image, ImageTk  # noqa: E402

from inference_engine import (  # noqa: E402
    DEFAULT_CROP_SIZE,
    DEFAULT_FEATHER_PX,
    DEFAULT_RHO,
    DEFAULT_S_C,
    DEFAULT_S_CALIB,
    DEFAULT_SIGMA_0,
    InferenceEngine,
    boundary_normalise,
)

try:
    from physics import validate_inference_config  # noqa: E402
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from physics import validate_inference_config  # noqa: E402

# ---------------------------------------------------------------------------
# Settings persistence
# ---------------------------------------------------------------------------

SETTINGS_FILE = Path(__file__).resolve().parent / "inference_settings.json"

DEFAULT_SETTINGS: Dict[str, Any] = {
    "model_path": "",
    "watch_folder": "",
    "auto_detect": False,
    "rho": DEFAULT_RHO,
    "sigma_0": DEFAULT_SIGMA_0,
    "s_calib": DEFAULT_S_CALIB,
    "s_c": DEFAULT_S_C,
    "feather_px": DEFAULT_FEATHER_PX,
    "crop_size": DEFAULT_CROP_SIZE,
    "device": "cpu",
    "rho_std": 0.0,
    "sigma_0_std": 0.0,
}


def load_settings() -> Dict[str, Any]:
    settings = dict(DEFAULT_SETTINGS)
    if SETTINGS_FILE.is_file():
        try:
            with open(SETTINGS_FILE, "r") as f:
                saved = json.load(f)
            settings.update(saved)
        except Exception as e:
            import logging
            logging.warning(f"Settings load failed ({e}). Using defaults. Calibration constants may be incorrect.")
    return settings


def save_settings(settings: Dict[str, Any]) -> None:
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helper: numpy → PhotoImage
# ---------------------------------------------------------------------------

def _np_to_photo(img: np.ndarray, max_side: int = 500):
    """Convert a grayscale or float32 image to a tkinter-compatible PhotoImage."""
    if img is None:
        return None

    if img.dtype == np.float32 or img.dtype == np.float64:
        disp = np.clip(img * 255, 0, 255).astype(np.uint8)
    else:
        disp = img.astype(np.uint8)

    h, w = disp.shape[:2]
    scale = min(max_side / max(h, 1), max_side / max(w, 1), 1.0)
    new_w, new_h = max(int(w * scale), 1), max(int(h * scale), 1)
    disp = cv2.resize(disp, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Handle grayscale vs colour
    if len(disp.shape) == 2:
        pil_img = Image.fromarray(disp, mode="L")
    else:
        pil_img = Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))

    return ImageTk.PhotoImage(pil_img)


def _draw_overlay(frame_gray: np.ndarray, geo: Optional[Dict[str, Any]],
                  crop_size: int, safety: int = 3,
                  show_crop_box: bool = True,
                  frame_idx: Optional[int] = None,
                  best_idx: Optional[int] = None) -> np.ndarray:
    """
    Draw preprocessing-style overlay on frame: geometry lines, and
    optionally a crop box (only on the best frame).
    Returns a BGR colour image.
    """
    rgb = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    h, w = frame_gray.shape

    if geo is None:
        # No geometry detected — just show the raw frame with a note
        cv2.putText(rgb, "No droplet detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        return rgb

    y_top = geo.get("y_top")
    y_bottom = geo.get("y_bottom")
    cx = geo.get("cx")
    y_sphere = geo.get("y_bottom_sphere")

    # Draw horizontal guide lines
    if y_top is not None:
        cv2.line(rgb, (0, int(y_top)), (w, int(y_top)), (0, 255, 0), 1)
    if y_bottom is not None:
        cv2.line(rgb, (0, int(y_bottom)), (w, int(y_bottom)), (0, 165, 255), 1)
    if y_sphere is not None:
        cv2.line(rgb, (0, int(y_sphere)), (w, int(y_sphere)), (0, 0, 255), 2)

    # Top margin: top of image to top of droplet
    if y_top is not None:
        top_margin_px = int(y_top)
        x_ann0 = w - 120
        cv2.line(rgb, (x_ann0, 0), (x_ann0, int(y_top)), (0, 255, 0), 1)
        cv2.putText(rgb, f"{top_margin_px} px", (x_ann0 - 55, int(y_top // 2) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Distance annotation between top and bottom of droplet
    if y_top is not None and y_bottom is not None:
        dist_px = int(abs(y_bottom - y_top))
        mid_y = int(0.5 * (y_top + y_bottom))
        x_ann = w - 80
        cv2.line(rgb, (x_ann, int(y_top)), (x_ann, int(y_bottom)), (255, 255, 255), 1)
        cv2.putText(rgb, f"{dist_px} px", (x_ann - 55, mid_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Gap to sphere
    if y_bottom is not None and y_sphere is not None:
        gap_px = int(abs(y_sphere - y_bottom))
        mid_gap = int(0.5 * (y_bottom + y_sphere))
        x_ann2 = w - 40
        cv2.line(rgb, (x_ann2, int(y_bottom)), (x_ann2, int(y_sphere)), (100, 100, 255), 1)
        cv2.putText(rgb, f"{gap_px} px", (x_ann2 - 55, mid_gap + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)

    # Crop box — only on the best frame
    is_best = (frame_idx is not None and best_idx is not None and frame_idx == best_idx)
    if show_crop_box and is_best and y_top is not None and y_bottom is not None and cx is not None:
        cy = int(0.5 * (y_top + y_bottom))
        half = crop_size // 2
        x0 = max(0, int(cx) - half)
        y0 = max(0, cy - half)
        x1 = min(w, x0 + crop_size)
        y1 = min(h, y0 + crop_size)

        if y_sphere is not None and y1 > (y_sphere - safety):
            shift = y1 - int(y_sphere - safety)
            y0 = max(0, y0 - shift)
            y1 = y0 + crop_size

        cv2.rectangle(rgb, (x0, y0), (x1, y1), (255, 255, 0), 2)
        cv2.putText(rgb, f"BEST — Crop {crop_size}x{crop_size}",
                    (x0, max(y0 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Frame info top-right
    if frame_idx is not None:
        label = f"Frame {frame_idx}"
        if is_best:
            label += " (BEST)"
        cv2.putText(rgb, label, (w - 180, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Legend in top-left
    legend_y = 20
    items = [((0, 255, 0), "Droplet top"),
             ((0, 165, 255), "Droplet bottom"),
             ((0, 0, 255), "Sphere")]
    if is_best:
        items.append(((255, 255, 0), "Crop region"))
    for colour, label in items:
        cv2.line(rgb, (10, legend_y), (30, legend_y), colour, 2)
        cv2.putText(rgb, label, (35, legend_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
        legend_y += 18

    return rgb


# ---------------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------------

class InferenceApp(tk.Tk):
    """Top-level tkinter window."""

    def __init__(self):
        super().__init__()
        self.title("Defocus Inference Tool")
        self.minsize(1000, 750)
        self.resizable(True, True)

        self.settings = load_settings()
        self.engine = InferenceEngine(self.settings)
        self._model_loaded = False
        self._photo_refs: list = []

        # Cine state
        self._cine_obj = None
        self._cine_range = (0, 0)  # (first_frame, last_frame)
        self._current_frame_idx: int = 0

        # Processing results
        self._results: Optional[Dict[str, Any]] = None
        self._best_frame_idx: Optional[int] = None
        self._best_geo: Optional[Dict[str, Any]] = None

        # View mode: "slider" = raw frame, "overlay" = annotated, "crop" = just the crop
        self._view_mode = "slider"

        self._build_ui()
        self._apply_settings_to_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # First-launch warning if no saved settings exist
        if not SETTINGS_FILE.is_file():
            self.after(500, lambda: messagebox.showinfo(
                "Calibration Required",
                "Default calibration values are loaded. These are examples "
                "from a specific lab setup.\n\n"
                "Run calibration for your camera before trusting results.\n\n"
                "Update values in Settings > Calibration.",
            ))

    # ── UI construction ────────────────────────────────────────────────

    def _build_ui(self):
        # --- Top: file selection ---
        file_frame = ttk.LabelFrame(self, text="Input", padding=8)
        file_frame.pack(fill="x", padx=10, pady=(10, 4))

        # Model row
        ttk.Label(file_frame, text="Model:").grid(row=0, column=0, sticky="w")
        self.var_model_path = tk.StringVar(value=self.settings.get("model_path", ""))
        ttk.Entry(file_frame, textvariable=self.var_model_path, width=60).grid(
            row=0, column=1, sticky="ew", padx=4
        )
        model_btn_frame = ttk.Frame(file_frame)
        model_btn_frame.grid(row=0, column=2, padx=4)
        ttk.Button(model_btn_frame, text="Browse...", command=self._browse_model).pack(side="left")
        self.btn_load_model = ttk.Button(model_btn_frame, text="Load", command=self._load_model)
        self.btn_load_model.pack(side="left", padx=(4, 0))

        # Cine row
        ttk.Label(file_frame, text=".cine file:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.var_cine_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.var_cine_path, width=60).grid(
            row=1, column=1, sticky="ew", padx=4, pady=(4, 0)
        )
        cine_btn_frame = ttk.Frame(file_frame)
        cine_btn_frame.grid(row=1, column=2, padx=4, pady=(4, 0))
        ttk.Button(cine_btn_frame, text="Browse...", command=self._browse_cine).pack(side="left")
        self.btn_open_cine = ttk.Button(cine_btn_frame, text="Open", command=self._open_cine)
        self.btn_open_cine.pack(side="left", padx=(4, 0))
        ttk.Button(cine_btn_frame, text="Batch Folder...", command=self._batch_folder).pack(
            side="left", padx=(4, 0)
        )

        # Watch folder
        ttk.Label(file_frame, text="Watch folder:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.var_watch_folder = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.var_watch_folder, width=60).grid(
            row=2, column=1, sticky="ew", padx=4, pady=(4, 0)
        )
        watch_btn_frame = ttk.Frame(file_frame)
        watch_btn_frame.grid(row=2, column=2, padx=4, pady=(4, 0))
        ttk.Button(watch_btn_frame, text="Browse...", command=self._browse_watch).pack(side="left")
        self.var_auto_detect = tk.BooleanVar()
        ttk.Checkbutton(watch_btn_frame, text="Auto", variable=self.var_auto_detect,
                        command=self._on_auto_toggle).pack(side="left", padx=(4, 0))

        file_frame.columnconfigure(1, weight=1)

        # --- Middle: zoomable/pannable image display ---
        img_frame = ttk.Frame(self)
        img_frame.pack(fill="both", expand=True, padx=0, pady=0)

        self.canvas = tk.Canvas(img_frame, bg="#2b2b2b", highlightthickness=0,
                                borderwidth=0, relief="flat")
        self.canvas.pack(fill="both", expand=True)
        self._canvas_img_id = None
        self._current_full_image: Optional[np.ndarray] = None  # full-res image for zoom
        self._zoom_level: float = 1.0
        self._pan_offset = [0, 0]  # [x, y] offset in image coords
        self._drag_start = None

        # Bind zoom (scroll) and pan (drag)
        self.canvas.bind("<MouseWheel>", self._on_scroll_zoom)        # Windows
        self.canvas.bind("<Button-4>", self._on_scroll_zoom)          # Linux up
        self.canvas.bind("<Button-5>", self._on_scroll_zoom)          # Linux down
        self.canvas.bind("<ButtonPress-1>", self._on_pan_start)
        self.canvas.bind("<B1-Motion>", self._on_pan_drag)
        self.canvas.bind("<Double-Button-1>", self._on_zoom_reset)    # Double-click resets

        # --- Frame slider ---
        slider_frame = ttk.Frame(self)
        slider_frame.pack(fill="x", padx=10, pady=(0, 4))

        self.var_frame_idx = tk.IntVar(value=0)
        self.frame_slider = ttk.Scale(
            slider_frame, from_=0, to=0, orient="horizontal",
            variable=self.var_frame_idx, command=self._on_slider_change,
        )
        self.frame_slider.pack(side="left", fill="x", expand=True)
        self.frame_slider.config(state="disabled")

        self.var_frame_label = tk.StringVar(value="Frame: ---")
        ttk.Label(slider_frame, textvariable=self.var_frame_label, width=20,
                  anchor="e").pack(side="right", padx=(8, 0))

        # --- Results ---
        result_frame = ttk.LabelFrame(self, text="Result", padding=8)
        result_frame.pack(fill="x", padx=10, pady=4)

        self.var_defocus = tk.StringVar(value="---")
        ttk.Label(
            result_frame, textvariable=self.var_defocus,
            font=("Segoe UI", 28, "bold"), foreground="#1a73e8",
        ).pack()

        detail_frame = ttk.Frame(result_frame)
        detail_frame.pack(pady=(4, 0))
        self.var_sigma_model = tk.StringVar(value="")
        self.var_sigma_native = tk.StringVar(value="")
        self.var_pred_norm = tk.StringVar(value="")
        for var in (self.var_pred_norm, self.var_sigma_model, self.var_sigma_native):
            ttk.Label(detail_frame, textvariable=var, font=("Consolas", 9)).pack(
                side="left", padx=12
            )

        # --- Action buttons ---
        action_frame = ttk.Frame(self)
        action_frame.pack(fill="x", padx=10, pady=4)

        self.btn_process = ttk.Button(action_frame, text="Process",
                                      command=self._process, state="disabled")
        self.btn_process.pack(side="left", padx=4)

        self.btn_view_overlay = ttk.Button(action_frame, text="Show Overlay",
                                           command=self._show_overlay, state="disabled")
        self.btn_view_overlay.pack(side="left", padx=4)

        self.btn_view_crop = ttk.Button(action_frame, text="Show Crop",
                                        command=self._show_crop, state="disabled")
        self.btn_view_crop.pack(side="left", padx=4)

        self.btn_view_normalised = ttk.Button(action_frame, text="Show Normalised",
                                              command=self._show_normalised, state="disabled")
        self.btn_view_normalised.pack(side="left", padx=4)

        self.btn_view_raw = ttk.Button(action_frame, text="Show Raw Frame",
                                       command=self._show_raw, state="disabled")
        self.btn_view_raw.pack(side="left", padx=4)

        ttk.Button(action_frame, text="Settings...", command=self._open_settings).pack(
            side="right", padx=4
        )

        # --- Status bar ---
        status_frame = ttk.Frame(self)
        status_frame.pack(fill="x", padx=10, pady=(0, 8))
        self.var_status = tk.StringVar(value="Ready. Load a model and open a .cine to begin.")
        ttk.Label(status_frame, textvariable=self.var_status, anchor="w").pack(
            side="left", fill="x", expand=True
        )
        self.progress = ttk.Progressbar(status_frame, length=180, mode="determinate")
        self.progress.pack(side="right", padx=(8, 0))

    # ── Settings ↔ UI ──────────────────────────────────────────────────

    def _apply_settings_to_ui(self):
        self.var_cine_path.set(self.settings.get("last_cine", ""))
        self.var_watch_folder.set(self.settings.get("watch_folder", ""))
        self.var_auto_detect.set(self.settings.get("auto_detect", False))

    def _gather_settings_from_ui(self):
        self.settings["last_cine"] = self.var_cine_path.get()
        self.settings["watch_folder"] = self.var_watch_folder.get()
        self.settings["auto_detect"] = self.var_auto_detect.get()
        self.settings["model_path"] = self.var_model_path.get()

    # ── Browse helpers ─────────────────────────────────────────────────

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Select model checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt *.pth"), ("All files", "*.*")],
        )
        if path:
            self.var_model_path.set(path)

    def _browse_cine(self):
        path = filedialog.askopenfilename(
            title="Select .cine file",
            filetypes=[("CINE files", "*.cine"), ("All files", "*.*")],
        )
        if path:
            self.var_cine_path.set(path)
            self.var_auto_detect.set(False)

    def _browse_watch(self):
        path = filedialog.askdirectory(title="Select watch folder")
        if path:
            self.var_watch_folder.set(path)

    def _on_auto_toggle(self):
        if self.var_auto_detect.get():
            folder = self.var_watch_folder.get().strip()
            if not folder or not Path(folder).is_dir():
                messagebox.showwarning("Watch Folder", "Set a valid watch folder first.")
                self.var_auto_detect.set(False)
                return
            latest = InferenceEngine.find_latest_cine(Path(folder))
            if latest:
                self.var_cine_path.set(str(latest))
                self.var_status.set(f"Auto-detected: {latest.name}")
            else:
                messagebox.showinfo("Watch Folder", "No .cine files found.")
                self.var_auto_detect.set(False)

    # ── Load model ─────────────────────────────────────────────────────

    def _load_model(self):
        model_path = self.var_model_path.get().strip()
        if not model_path or not Path(model_path).is_file():
            messagebox.showwarning("No model", "Browse for a checkpoint file first.")
            return
        self.settings["model_path"] = model_path

        self.var_status.set("Loading model...")
        self.update_idletasks()

        try:
            self.engine = InferenceEngine(self.settings)
            msg = self.engine.load_model()
            self._model_loaded = True
            self._update_button_states()
            self.var_status.set(msg)
        except Exception as e:
            self._model_loaded = False
            self._update_button_states()
            messagebox.showerror("Model Load Error", str(e))
            self.var_status.set("Model load failed.")

    # ── Open cine + slider ─────────────────────────────────────────────

    def _open_cine(self):
        cine_path_str = self.var_cine_path.get().strip()
        if not cine_path_str:
            messagebox.showwarning("No file", "Browse for a .cine file first.")
            return
        cine_path = Path(cine_path_str)
        if not cine_path.is_file():
            messagebox.showerror("File not found", str(cine_path))
            return

        self.var_status.set("Opening .cine file...")
        self.update_idletasks()

        try:
            from cine_io import safe_load_cine  # already on sys.path via engine
            self._cine_obj = safe_load_cine(cine_path)
            if self._cine_obj is None:
                raise RuntimeError(f"Could not open {cine_path.name}")

            first, last = self._cine_obj.range
            self._cine_range = (first, last)

            # Configure slider
            self.frame_slider.config(from_=first, to=last, state="normal")
            self.var_frame_idx.set(first)

            # Reset results
            self._results = None
            self._best_frame_idx = None
            self._best_geo = None
            self._view_mode = "slider"

            self._update_button_states()
            self._display_frame(first)
            self.var_status.set(
                f"Opened {cine_path.name}  |  frames {first} to {last} "
                f"({last - first + 1} total)"
            )

        except Exception as e:
            self._cine_obj = None
            messagebox.showerror("Open Error", str(e))
            self.var_status.set("Failed to open .cine")

    def _on_slider_change(self, val):
        """Called when the frame slider moves."""
        if self._cine_obj is None:
            return
        idx = int(float(val))
        self._current_frame_idx = idx

        is_best = (self._best_frame_idx is not None and idx == self._best_frame_idx)
        label = f"Frame: {idx}"
        if is_best:
            label += " (BEST)"
        self.var_frame_label.set(label)

        if self._view_mode == "slider":
            self._display_frame(idx)
        elif self._view_mode == "overlay":
            self._display_frame_with_overlay(idx)

    def _display_frame(self, idx: int):
        """Load and display a single frame from the cine."""
        if self._cine_obj is None:
            return
        try:
            from image_utils import load_frame_gray
            frame = load_frame_gray(self._cine_obj, idx)
            self._show_image(frame)
            self.var_frame_label.set(f"Frame: {idx}")
        except Exception as e:
            self.var_status.set(f"Frame load error: {e}")

    def _display_frame_with_overlay(self, idx: int):
        """Load a frame, compute its geometry on the fly, and draw the overlay."""
        if self._cine_obj is None:
            return
        try:
            from image_utils import load_frame_gray
            from geom_analysis import analyze_frame_geometric
            frame = load_frame_gray(self._cine_obj, idx)

            # Compute geometry for this frame
            geo = analyze_frame_geometric(self._cine_obj, idx)

            crop_size = int(self.settings.get("crop_size", DEFAULT_CROP_SIZE))
            overlay = _draw_overlay(
                frame, geo, crop_size,
                show_crop_box=True,
                frame_idx=idx,
                best_idx=self._best_frame_idx,
            )
            self._show_image(overlay)
        except Exception as e:
            self.var_status.set(f"Overlay error: {e}")

    def _show_image(self, img: np.ndarray, reset_view: bool = False):
        """Display an image on the canvas. Stores full-res for zoom."""
        self._current_full_image = img
        if reset_view:
            self._zoom_level = 1.0
            self._pan_offset = [0, 0]
        self._render_canvas()

    def _render_canvas(self):
        """Render the current image to the canvas at the current zoom/pan."""
        img = self._current_full_image
        if img is None:
            return

        self._photo_refs.clear()
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            # Canvas not yet mapped — schedule a retry
            self.after(50, self._render_canvas)
            return

        # Convert to uint8 display
        if img.dtype in (np.float32, np.float64):
            disp = np.clip(img * 255, 0, 255).astype(np.uint8)
        else:
            disp = img.astype(np.uint8)

        img_h, img_w = disp.shape[:2]

        # Base scale: fit image to canvas
        base_scale = min(canvas_w / img_w, canvas_h / img_h, 1.0)
        scale = base_scale * self._zoom_level

        new_w = max(int(img_w * scale), 1)
        new_h = max(int(img_h * scale), 1)

        # Apply pan offset (in scaled pixels)
        ox, oy = self._pan_offset

        # Crop the source image region that's visible
        # (for large zoom levels, only render what's on screen)
        resized = cv2.resize(disp, (new_w, new_h),
                             interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)

        # Convert to PIL
        if len(resized.shape) == 2:
            pil_img = Image.fromarray(resized, mode="L")
        else:
            pil_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

        photo = ImageTk.PhotoImage(pil_img)
        self._photo_refs.append(photo)

        # Place on canvas centred + pan offset
        cx = canvas_w // 2 + int(ox)
        cy = canvas_h // 2 + int(oy)

        self.canvas.delete("all")
        self._canvas_img_id = self.canvas.create_image(cx, cy, image=photo, anchor="center")

    def _on_scroll_zoom(self, event):
        """Zoom in/out with scroll wheel, centred on mouse position."""
        if self._current_full_image is None:
            return

        # Determine scroll direction
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            factor = 1.15
        elif event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            factor = 1 / 1.15
        else:
            return

        old_zoom = self._zoom_level
        self._zoom_level = max(1.0, min(20.0, self._zoom_level * factor))

        # Adjust pan to zoom towards mouse position
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        mx = event.x - canvas_w // 2 - self._pan_offset[0]
        my = event.y - canvas_h // 2 - self._pan_offset[1]
        zoom_ratio = self._zoom_level / old_zoom
        self._pan_offset[0] -= mx * (zoom_ratio - 1)
        self._pan_offset[1] -= my * (zoom_ratio - 1)

        self._render_canvas()

    def _on_pan_start(self, event):
        """Start panning."""
        self._drag_start = (event.x, event.y)

    def _on_pan_drag(self, event):
        """Pan the image by dragging."""
        if self._drag_start is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._pan_offset[0] += dx
        self._pan_offset[1] += dy
        self._drag_start = (event.x, event.y)
        self._render_canvas()

    def _on_zoom_reset(self, _event):
        """Double-click to reset zoom and pan."""
        self._zoom_level = 1.0
        self._pan_offset = [0, 0]
        self._render_canvas()

    # ── Process ────────────────────────────────────────────────────────

    def _process(self):
        if not self._model_loaded:
            messagebox.showwarning("Not ready", "Load a model first.")
            return
        if self._cine_obj is None:
            messagebox.showwarning("Not ready", "Open a .cine file first.")
            return

        # Pre-flight sanity check
        issues = validate_inference_config(
            rho=float(self.settings.get("rho", DEFAULT_RHO)),
            sigma_0=float(self.settings.get("sigma_0", DEFAULT_SIGMA_0)),
            s_calib=float(self.settings.get("s_calib", DEFAULT_S_CALIB)),
            s_c=float(self.settings.get("s_c", DEFAULT_S_C)),
            max_blur=self.engine.max_blur if hasattr(self.engine, "max_blur") else 20.0,
            model_size=self.engine.model_size if hasattr(self.engine, "model_size") else 256,
            crop_size=int(self.settings.get("crop_size", DEFAULT_CROP_SIZE)),
        )
        if issues:
            proceed = messagebox.askyesno(
                "Sanity Check",
                "Issues detected:\n\n" + "\n".join(f"- {i}" for i in issues)
                + "\n\nProceed anyway?",
            )
            if not proceed:
                return

        self.btn_process.config(state="disabled")
        self.var_defocus.set("...")
        self.var_sigma_model.set("")
        self.var_sigma_native.set("")
        self.var_pred_norm.set("")
        self.progress["value"] = 0

        cine_path = Path(self.var_cine_path.get().strip())
        thread = threading.Thread(target=self._run_pipeline, args=(cine_path,), daemon=True)
        thread.start()

    def _run_pipeline(self, cine_path: Path):
        """Background worker."""
        try:
            def progress_cb(msg, frac):
                self.after(0, self._update_progress, msg, frac)

            results = self.engine.process_cine(cine_path, progress_cb=progress_cb)
            self.after(0, self._on_pipeline_done, results)
        except Exception as e:
            self.after(0, self._pipeline_error, str(e))

    def _update_progress(self, msg: str, frac: float):
        self.var_status.set(msg)
        self.progress["value"] = frac * 100

    def _pipeline_error(self, msg: str):
        self.btn_process.config(state="normal")
        self.var_status.set("Error.")
        self.progress["value"] = 0
        messagebox.showerror("Pipeline Error", msg)

    def _on_pipeline_done(self, results: Dict[str, Any]):
        """Called on main thread when pipeline completes."""
        self._results = results
        self._best_frame_idx = results.get("frame_idx")
        self._best_geo = results.get("geometry")

        # Jump slider to best frame
        if self._best_frame_idx is not None:
            self.var_frame_idx.set(self._best_frame_idx)
            self.var_frame_label.set(f"Frame: {self._best_frame_idx} (best)")

        # Show overlay view automatically on the best frame
        self._view_mode = "overlay"
        self._display_frame_with_overlay(self._best_frame_idx)

        # Update results display
        defocus = results.get("defocus_mm", 0.0)
        flags = []
        if results.get("saturated"):
            flags.append("SATURATED")
        if results.get("clamped"):
            flags.append("CLAMPED")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        unc = results.get("defocus_uncertainty_mm", 0.0)
        unc_str = f" \u00b1 {unc:.3f}" if unc > 0 else ""
        self.var_defocus.set(f"{defocus:.3f}{unc_str} mm{flag_str}")
        self.var_pred_norm.set(f"pred_norm={results.get('pred_norm', 0):.4f}")
        self.var_sigma_model.set(f"\u03c3_model={results.get('sigma_model', 0):.3f} px")
        self.var_sigma_native.set(f"\u03c3_native={results.get('sigma_native', 0):.3f} px")

        cine_name = results.get("cine_name", "")
        self.var_status.set(
            f"Done  |  {cine_name}  |  frame {self._best_frame_idx}  |  "
            f"defocus = {defocus:.3f} mm{flag_str}"
        )

        self._update_button_states()

    # ── View mode buttons ──────────────────────────────────────────────

    def _show_overlay(self):
        """Switch to overlay mode — slider now shows geometry on every frame."""
        if self._results is None:
            return
        self._view_mode = "overlay"
        # Jump to best frame and render it
        if self._best_frame_idx is not None:
            self.var_frame_idx.set(self._best_frame_idx)
        self._display_frame_with_overlay(int(self.var_frame_idx.get()))

    def _show_crop(self):
        """Show just the extracted crop."""
        if self._results is None:
            return
        self._view_mode = "crop"
        crop = self._results.get("crop")
        if crop is not None:
            self._show_image(crop)

    def _show_normalised(self):
        """Show the boundary-normalised crop."""
        if self._results is None:
            return
        self._view_mode = "normalised"
        norm = self._results.get("norm_img")
        if norm is not None:
            self._show_image(norm)

    def _show_raw(self):
        """Switch back to raw frame slider mode."""
        self._view_mode = "slider"
        self._display_frame(int(self.var_frame_idx.get()))

    # ── Button state management ────────────────────────────────────────

    def _update_button_states(self):
        has_cine = self._cine_obj is not None
        has_model = self._model_loaded
        has_results = self._results is not None

        self.btn_process.config(state="normal" if (has_cine and has_model) else "disabled")
        self.btn_view_overlay.config(state="normal" if has_results else "disabled")
        self.btn_view_crop.config(state="normal" if has_results else "disabled")
        self.btn_view_normalised.config(state="normal" if has_results else "disabled")
        self.btn_view_raw.config(state="normal" if has_cine else "disabled")

    # ── Settings dialog ────────────────────────────────────────────────

    def _open_settings(self):
        SettingsDialog(self, self.settings, on_save=self._on_settings_saved)

    def _on_settings_saved(self, new_settings: Dict[str, Any]):
        self.settings.update(new_settings)
        save_settings(self.settings)
        self._model_loaded = False
        self._update_button_states()
        self.var_status.set("Settings saved. Reload model to apply changes.")

    # ── Batch folder processing ───────────────────────────────────────

    def _batch_folder(self):
        if not self._model_loaded:
            messagebox.showwarning("Not ready", "Load a model first.")
            return

        folder = filedialog.askdirectory(title="Select folder containing .cine files")
        if not folder:
            return
        folder_path = Path(folder)

        cine_files = sorted(folder_path.glob("*.cine"), key=lambda p: p.name)
        if not cine_files:
            messagebox.showinfo("No files", "No .cine files found in the selected folder.")
            return

        # Pre-flight sanity check
        issues = validate_inference_config(
            rho=float(self.settings.get("rho", DEFAULT_RHO)),
            sigma_0=float(self.settings.get("sigma_0", DEFAULT_SIGMA_0)),
            s_calib=float(self.settings.get("s_calib", DEFAULT_S_CALIB)),
            s_c=float(self.settings.get("s_c", DEFAULT_S_C)),
            max_blur=self.engine.max_blur if hasattr(self.engine, "max_blur") else 20.0,
            model_size=self.engine.model_size if hasattr(self.engine, "model_size") else 256,
            crop_size=int(self.settings.get("crop_size", DEFAULT_CROP_SIZE)),
        )

        confirm_msg = f"Found {len(cine_files)} .cine file(s) in:\n{folder_path}"
        if issues:
            confirm_msg += "\n\nWarnings:\n" + "\n".join(f"- {i}" for i in issues)
        confirm_msg += "\n\nProcess all?"

        if not messagebox.askyesno("Batch Processing", confirm_msg):
            return

        self.btn_process.config(state="disabled")
        self.progress["value"] = 0
        self.var_defocus.set("Batch...")

        thread = threading.Thread(
            target=self._run_batch, args=(folder_path,), daemon=True
        )
        thread.start()

    def _run_batch(self, folder_path: Path):
        try:
            def progress_cb(msg, idx, total):
                frac = idx / total if total > 0 else 0
                self.after(0, self._update_progress, msg, frac)

            results = self.engine.process_folder(folder_path, progress_cb=progress_cb)
            self.after(0, self._on_batch_done, results)
        except Exception as e:
            self.after(0, self._pipeline_error, str(e))

    def _on_batch_done(self, results: List[Dict[str, Any]]):
        self.progress["value"] = 100

        successes = [r for r in results if "error" not in r]
        failures = [r for r in results if "error" in r]

        if not successes:
            messagebox.showerror("Batch Failed", "All files failed to process.")
            self.var_defocus.set("---")
            self.var_status.set("Batch failed.")
            self.btn_process.config(state="normal")
            return

        # Prompt user for CSV save location
        csv_path = filedialog.asksaveasfilename(
            title="Save batch results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if csv_path:
            fieldnames = ["cine_name", "frame_idx", "pred_norm",
                          "sigma_model", "sigma_native", "defocus_mm",
                          "defocus_uncertainty_mm",
                          "saturated", "clamped",
                          "training_mode", "model_path", "rho", "sigma_0",
                          "s_calib", "s_c", "feather_px", "crop_size"]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(successes)

        mean_defocus = sum(r["defocus_mm"] for r in successes) / len(successes)
        summary = (
            f"Batch complete: {len(successes)} processed"
            f"{f', {len(failures)} failed' if failures else ''}"
            f"  |  mean defocus = {mean_defocus:.3f} mm"
        )
        self.var_defocus.set(f"{mean_defocus:.3f} mm (mean)")
        self.var_status.set(summary)
        self.btn_process.config(state="normal")

        if failures:
            fail_names = "\n".join(r["cine_name"] for r in failures)
            messagebox.showwarning(
                "Batch Warnings",
                f"{len(failures)} file(s) failed:\n\n{fail_names}",
            )

    # ── Cleanup ────────────────────────────────────────────────────────

    def _on_close(self):
        self._gather_settings_from_ui()
        save_settings(self.settings)
        self.destroy()


# ---------------------------------------------------------------------------
# Settings Dialog
# ---------------------------------------------------------------------------

class SettingsDialog(tk.Toplevel):
    """Modal dialog for editing calibration and pipeline settings."""

    def __init__(self, parent, settings: Dict[str, Any], on_save=None):
        super().__init__(parent)
        self.title("Settings")
        self.transient(parent)
        self.grab_set()
        self.resizable(False, False)

        self.settings = dict(settings)
        self.on_save = on_save
        self._entries: Dict[str, tk.Variable] = {}

        self._build()
        self.wait_window(self)

    def _build(self):
        # Calibration
        f_cal = ttk.LabelFrame(self, text="Calibration", padding=6)
        f_cal.pack(fill="x", padx=10, pady=(10, 4))
        ttk.Label(f_cal, text="(These must match your camera \u2014 run Calibration GUI first)",
                  font=("TkDefaultFont", 8), foreground="gray").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=(0, 0), pady=(0, 4))
        self._add_float_row(f_cal, "rho", "\u03c1 (px/mm):", 1)
        self._add_float_row(f_cal, "sigma_0", "\u03c3\u2080 (px):", 2)
        self._add_float_row(f_cal, "s_calib", "s_calib (px/mm):", 3)
        self._add_float_row(f_cal, "s_c", "s_c (px/mm):", 4)
        ttk.Label(f_cal, text="Uncertainty (from LOO-CV, optional \u2014 0 = disabled):",
                  font=("TkDefaultFont", 8), foreground="gray").grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self._add_float_row(f_cal, "rho_std", "\u03c1 std (px/mm):", 6)
        self._add_float_row(f_cal, "sigma_0_std", "\u03c3\u2080 std (px):", 7)

        # Pipeline
        f_pipe = ttk.LabelFrame(self, text="Pipeline", padding=6)
        f_pipe.pack(fill="x", padx=10, pady=4)
        self._add_int_row(f_pipe, "feather_px", "Feather width (px):", 0)
        ttk.Label(f_pipe, text="(Gaussian boundary fade width, typical 20\u201360)",
                  font=("TkDefaultFont", 8), foreground="gray").grid(
            row=1, column=0, columnspan=2, sticky="w", padx=(20, 0))
        self._add_int_row(f_pipe, "crop_size", "Crop size (px):", 2)
        self._add_combo_row(f_pipe, "device", "Device:", 3, ["cpu", "cuda"])

        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=10, pady=(8, 10))
        ttk.Button(btn_frame, text="Save", command=self._save).pack(side="right", padx=4)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(btn_frame, text="Defaults", command=self._reset_defaults).pack(side="left")

    def _add_float_row(self, parent, key, label, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6))
        var = tk.StringVar(value=str(self.settings.get(key, "")))
        ttk.Entry(parent, textvariable=var, width=18).grid(row=row, column=1, sticky="ew", pady=2)
        self._entries[key] = var
        parent.columnconfigure(1, weight=1)

    def _add_int_row(self, parent, key, label, row):
        self._add_float_row(parent, key, label, row)

    def _add_combo_row(self, parent, key, label, row, values):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6))
        var = tk.StringVar(value=str(self.settings.get(key, values[0])))
        ttk.Combobox(parent, textvariable=var, values=values, state="readonly",
                     width=16).grid(row=row, column=1, sticky="w", pady=2)
        self._entries[key] = var
        parent.columnconfigure(1, weight=1)

    def _save(self):
        new = {}
        float_keys = {"rho", "sigma_0", "s_calib", "s_c", "rho_std", "sigma_0_std"}
        int_keys = {"feather_px", "crop_size"}

        for key, var in self._entries.items():
            raw = var.get().strip()
            if key in float_keys:
                try:
                    new[key] = float(raw)
                except ValueError:
                    messagebox.showerror("Invalid", f"'{key}' must be a number.")
                    return
            elif key in int_keys:
                try:
                    new[key] = int(raw)
                except ValueError:
                    messagebox.showerror("Invalid", f"'{key}' must be an integer.")
                    return
            else:
                new[key] = raw

        # Bounds validation
        checks = [
            ("rho", new.get("rho", 1), lambda v: v > 0, "rho must be positive."),
            ("sigma_0", new.get("sigma_0", 0), lambda v: v >= 0, "sigma_0 cannot be negative."),
            ("s_calib", new.get("s_calib", 1), lambda v: v > 0, "s_calib must be positive."),
            ("s_c", new.get("s_c", 1), lambda v: v > 0, "s_c must be positive."),
            ("feather_px", new.get("feather_px", 0), lambda v: v >= 0, "Feather width cannot be negative."),
            ("crop_size", new.get("crop_size", 256), lambda v: v >= 32, "Crop size must be at least 32 px."),
        ]
        for key, val, check, msg in checks:
            if key in new and not check(val):
                messagebox.showerror("Invalid", msg)
                return
        if new.get("feather_px", 0) > new.get("crop_size", 256):
            messagebox.showerror("Invalid", "Feather width cannot exceed crop size.")
            return

        self.settings.update(new)
        if self.on_save:
            self.on_save(self.settings)
        self.destroy()

    def _reset_defaults(self):
        if not messagebox.askyesno("Reset", "Reset all settings to defaults?"):
            return
        for key, val in DEFAULT_SETTINGS.items():
            if key in self._entries:
                self._entries[key].set(str(val))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = InferenceApp()
    app.mainloop()
