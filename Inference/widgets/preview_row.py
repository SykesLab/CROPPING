"""PreviewRow — three-thumbnail row showing original / processed / overlay.

Centrepiece of the V2 UX: the user sees the actual model input alongside
the source crop and a sphere-detection overlay, with the auto-decision
rationale displayed prominently. An override dropdown lets the user lock
a specific flatten mode for the session.

Layout:
  ┌─ Auto: <rationale text>                                      ┐
  │  Mode: [▼ Auto-detect ▼]   [advanced…]                       │
  │  ┌──────────┐    ┌──────────┐    ┌──────────┐                │
  │  │ Original │ →  │Processed │    │ Overlay  │                │
  │  │  crop    │    │ (model   │    │ (sphere  │                │
  │  │          │    │  input)  │    │  detect) │                │
  │  └──────────┘    └──────────┘    └──────────┘                │
  │  σ predicted: 1.34 px  →  z = 1.28 mm  [IN_RANGE]            │
  └──────────────────────────────────────────────────────────────┘

Public API:
  - PreviewRow(parent, on_override_change=cb)
  - .update_with_crop(crop, engine, override_setting) — render thumbs
  - .update_with_results(results_dict) — render result line
  - .clear() — blank everything
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from Calibration.sphere_processing import (
    _detect_sphere_contour,
    _detect_sphere_contour_otsu,
)
from Inference.auto_preprocess import (
    AutoDecision,
    MODE_AUTO,
    MODE_BOUNDARY,
    MODE_CALIBRATION,
    MODE_NONE,
    MODE_SIMPLE,
    resolve_mode,
)


# Thumbnail dimensions
_THUMB_SIZE = 180

# Descriptive labels for the dropdown (with technical name in tooltip)
MODE_LABELS = [
    ("Auto-detect", MODE_AUTO),
    ("Calibration sphere mode", MODE_CALIBRATION),
    ("Tight droplet mode", MODE_BOUNDARY),
    ("Simple flatten", MODE_SIMPLE),
    ("No flatten", MODE_NONE),
]
_LABEL_TO_MODE = {label: mode for label, mode in MODE_LABELS}
_MODE_TO_LABEL = {mode: label for label, mode in MODE_LABELS}

# Bounds-flag colour mapping
_FLAG_COLOURS = {
    "IN_RANGE": "#1a5d1a",
    "BELOW_FLOOR": "#a04000",
    "SATURATED": "#a01515",
}
_FLAG_BG = {
    "IN_RANGE": "#e8f4e8",
    "BELOW_FLOOR": "#fff4e0",
    "SATURATED": "#fde0e0",
}

_COLOUR_MUTED = "#707070"
_COLOUR_NEUTRAL = "#404040"
_COLOUR_ACCENT = "#1a5d1a"


class PreviewRow(ttk.LabelFrame):
    """Three-thumbnail preview + auto-decision text + override dropdown."""

    def __init__(
        self,
        parent: tk.Misc,
        on_override_change: Optional[Callable[[str], None]] = None,
        on_view_full_size: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(parent, text="Preprocessing preview",
                         padding=(8, 6))
        self._on_override_change = on_override_change or (lambda m: None)
        self._on_view_full_size = on_view_full_size or (lambda: None)

        self.var_rationale = tk.StringVar(value="(open a cine to see preview)")
        self.var_mode_label = tk.StringVar(value=MODE_LABELS[0][0])
        self.var_result_text = tk.StringVar(value="")

        self._photo_refs: list = []  # keep PIL references alive
        self._last_decision: Optional[AutoDecision] = None

        self._build()

    # ── Build UI ──────────────────────────────────────────────────────
    def _build(self) -> None:
        # Row 0: rationale text
        self.lbl_rationale = ttk.Label(
            self, textvariable=self.var_rationale,
            foreground=_COLOUR_NEUTRAL,
            font=("TkDefaultFont", 9), wraplength=900, justify="left")
        self.lbl_rationale.pack(fill="x", anchor="w")

        # Row 1: mode dropdown
        mode_row = ttk.Frame(self)
        mode_row.pack(fill="x", pady=(2, 6))
        ttk.Label(mode_row, text="Flatten mode:",
                  foreground=_COLOUR_MUTED,
                  font=("TkDefaultFont", 8)).pack(side="left", padx=(0, 4))
        self.combo_mode = ttk.Combobox(
            mode_row, textvariable=self.var_mode_label,
            values=[label for label, _ in MODE_LABELS],
            state="readonly", width=24)
        self.combo_mode.pack(side="left")
        self.combo_mode.bind("<<ComboboxSelected>>", self._handle_mode_change)

        # Row 2: three thumbnails. The "processed" one gets a thicker
        # green border to make it obvious that's what's locked in for
        # inference (the model input).
        thumb_row = ttk.Frame(self)
        thumb_row.pack(fill="x", pady=(2, 0))
        self._thumb_canvases = {}
        self._thumb_titles = {}
        for i, (key, title) in enumerate([
                ("original", "Original crop"),
                ("processed", "✓ Pre-processed (model input)"),
                ("overlay", "Sphere detect overlay"),
        ]):
            cell = ttk.Frame(thumb_row)
            cell.pack(side="left", padx=(0, 8))
            title_var = tk.StringVar(value=title)
            title_fg = _COLOUR_ACCENT if key == "processed" else _COLOUR_MUTED
            title_font = (("TkDefaultFont", 8, "bold")
                          if key == "processed"
                          else ("TkDefaultFont", 8))
            ttk.Label(cell, textvariable=title_var,
                      foreground=title_fg,
                      font=title_font).pack(anchor="w")
            border_thickness = 3 if key == "processed" else 1
            border_colour = _COLOUR_ACCENT if key == "processed" else "#888"
            cv = tk.Canvas(cell, width=_THUMB_SIZE, height=_THUMB_SIZE,
                            bg="#404040",
                            highlightthickness=border_thickness,
                            highlightbackground=border_colour)
            cv.pack()
            self._thumb_canvases[key] = cv
            self._thumb_titles[key] = title_var
            # Arrow between original and processed
            if i == 0:
                ttk.Label(thumb_row, text="→",
                          font=("TkDefaultFont", 14, "bold"),
                          foreground=_COLOUR_MUTED).pack(side="left", padx=4)
        # "View full size" jump-to-tab-2 link
        link_cell = ttk.Frame(thumb_row)
        link_cell.pack(side="left", padx=(8, 0), anchor="s", pady=(0, 4))
        self.btn_view_full = ttk.Button(
            link_cell, text="View full size →",
            command=self._on_view_full_size, width=18)
        self.btn_view_full.pack()

        # Row 3: result line
        self.lbl_result = tk.Label(
            self, textvariable=self.var_result_text,
            background="#f0f0f0", foreground=_COLOUR_NEUTRAL,
            font=("Consolas", 11, "bold"), padx=8, pady=4, anchor="w")
        self.lbl_result.pack(fill="x", pady=(8, 0))

    # ── Public API ────────────────────────────────────────────────────
    def set_override_setting(self, mode_setting: str) -> None:
        """Sync the dropdown to a setting value (without firing callback).

        Used at startup or when settings get loaded from JSON.
        """
        label = _MODE_TO_LABEL.get(mode_setting, MODE_LABELS[0][0])
        if self.var_mode_label.get() != label:
            self.var_mode_label.set(label)

    def update_with_crop(
        self,
        crop: np.ndarray,
        engine: Any,
        override_setting: str,
    ) -> AutoDecision:
        """Run preprocess for the chosen mode and update all three thumbs.

        Returns the AutoDecision so the caller can inspect/reuse it.
        """
        # Decide which mode to use
        decision = resolve_mode(override_setting, crop)
        self._last_decision = decision

        # Update rationale label
        self.var_rationale.set(f"→ {decision.rationale}")

        # Render thumbnails
        self._set_thumb("original", crop)

        processed = self._compute_processed(crop, decision.mode, engine)
        self._set_thumb("processed", processed)

        overlay = self._compute_overlay(crop, decision.info,
                                          mode=decision.mode)
        self._set_thumb("overlay", overlay)

        return decision

    def update_with_results(self, results: Optional[dict]) -> None:
        """Render the σ / z / bounds_flag result line."""
        if not results:
            self.var_result_text.set("")
            self.lbl_result.configure(background="#f0f0f0",
                                        foreground=_COLOUR_NEUTRAL)
            return
        sigma = results.get("sigma_native", 0.0)
        z = results.get("defocus_mm", 0.0)
        flag = str(results.get("bounds_flag", "IN_RANGE"))
        unc = results.get("defocus_uncertainty_mm", 0.0) or 0.0
        unc_str = f" ± {unc:.3f} mm" if unc > 0 else ""
        self.var_result_text.set(
            f"σ = {sigma:.3f} px   →   z = {z:.3f} mm{unc_str}   [{flag}]"
        )
        self.lbl_result.configure(
            background=_FLAG_BG.get(flag, "#f0f0f0"),
            foreground=_FLAG_COLOURS.get(flag, _COLOUR_NEUTRAL),
        )

    def clear(self) -> None:
        """Blank all displays."""
        self.var_rationale.set("(open a cine to see preview)")
        self.var_result_text.set("")
        for canvas in self._thumb_canvases.values():
            canvas.delete("all")
        self._photo_refs.clear()

    # ── Internal: thumbnail rendering ─────────────────────────────────
    def _set_thumb(self, key: str, img: Optional[np.ndarray]) -> None:
        canvas = self._thumb_canvases[key]
        canvas.delete("all")
        if img is None:
            canvas.create_text(
                _THUMB_SIZE // 2, _THUMB_SIZE // 2,
                text="(none)", fill="#888")
            return
        # Convert to display uint8
        if img.dtype in (np.float32, np.float64):
            disp = np.clip(img * 255, 0, 255).astype(np.uint8)
        else:
            disp = img.astype(np.uint8)
        # Resize to thumb dim
        h, w = disp.shape[:2]
        scale = min(_THUMB_SIZE / w, _THUMB_SIZE / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        thumb = cv2.resize(
            disp, (new_w, new_h),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        if thumb.ndim == 2:
            pil_img = Image.fromarray(thumb, mode="L")
        else:
            pil_img = Image.fromarray(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(pil_img)
        self._photo_refs.append(photo)
        canvas.create_image(_THUMB_SIZE // 2, _THUMB_SIZE // 2,
                            image=photo, anchor="center")

    def _compute_processed(
        self,
        crop: np.ndarray,
        mode: str,
        engine: Any,
    ) -> Optional[np.ndarray]:
        """Run engine.preprocess_crop with a temporary mode override and
        return the processed image (in [0,1] float32 as engine returns it).

        Doesn't run the model — just the preprocessing step. Cheap.
        """
        if engine is None:
            return None
        # Save & restore the engine's flatten_mode setting
        prev_mode = engine.settings.get("flatten_mode")
        prev_inner = engine.settings.get("inner_margin_px")
        try:
            if mode == MODE_NONE:
                # No flatten — convert input to [0,1] float and return
                if crop.dtype != np.uint8:
                    return crop.astype(np.float32)
                return crop.astype(np.float32) / 255.0
            engine.settings["flatten_mode"] = mode
            engine.settings.setdefault("inner_margin_px", 20)
            norm_img, _tensor = engine.preprocess_crop(crop)
            return norm_img
        except Exception as e:
            # Render an error thumb instead of crashing
            err_img = np.zeros((100, 100), dtype=np.uint8)
            return err_img
        finally:
            if prev_mode is not None:
                engine.settings["flatten_mode"] = prev_mode
            if prev_inner is not None:
                engine.settings["inner_margin_px"] = prev_inner

    def _compute_overlay(
        self,
        crop: np.ndarray,
        info: dict,
        mode: str = MODE_CALIBRATION,
    ) -> Optional[np.ndarray]:
        """Render an overlay thumb showing what the *chosen mode's* sphere
        detector actually found. Each flatten mode uses a different
        detector internally; we re-run that same detector here so the
        overlay is honest about what the preprocessing saw.

        Per-mode detector:
          - calibration (cfg1)        → _detect_sphere_contour_otsu (Otsu+contour)
          - simple (cfg3)             → _detect_sphere_contour (Canny+contour)
          - boundary_normalise (cfg4) → cv2.threshold(OTSU) (binary mask edge)
          - none                      → no detection
        """
        if crop is None:
            return None
        if crop.dtype != np.uint8:
            disp = np.clip(crop * 255, 0, 255).astype(np.uint8) if crop.max() <= 1.5 \
                   else crop.astype(np.uint8)
        else:
            disp = crop.copy()
        bgr = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

        # ── Per-mode detection ──
        contour = None
        detector_name = ""
        try:
            if mode == MODE_CALIBRATION:
                detector_name = "Otsu + contour (cfg1)"
                result = _detect_sphere_contour_otsu(disp)
                if result is not None:
                    contour = result[0]
            elif mode == MODE_SIMPLE:
                detector_name = "Canny + contour (cfg3)"
                result = _detect_sphere_contour(disp)
                if result is not None:
                    contour = result[0]
            elif mode == MODE_BOUNDARY:
                detector_name = "Otsu threshold (cfg4)"
                # Reproduce boundary_normalise's first step: Otsu + fill
                _, binary = cv2.threshold(
                    disp, 0, 255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # cfg4 takes the dark mask (droplet is dark on light bg)
                dark_mask = (binary == 0).astype(np.uint8)
                fill_contours, _ = cv2.findContours(
                    dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(dark_mask, fill_contours, -1, 1, cv2.FILLED)
                # Draw boundary of the filled mask
                edge_contours, _ = cv2.findContours(
                    dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if edge_contours:
                    contour = max(edge_contours, key=cv2.contourArea)
            elif mode == MODE_NONE:
                detector_name = "(none — no flatten)"
            else:
                detector_name = f"(unknown mode '{mode}')"
        except Exception as e:
            detector_name = f"(detector error: {type(e).__name__})"

        # ── Draw ──
        if contour is not None and len(contour) >= 5:
            cv2.drawContours(bgr, [contour], -1, (0, 255, 0), 2)
            # Compute centre via moments + draw cross
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.drawMarker(bgr, (cx, cy), (0, 255, 0),
                                 markerType=cv2.MARKER_CROSS,
                                 markerSize=10, thickness=1)
        elif mode == MODE_NONE:
            cv2.putText(bgr, "no flatten", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        else:
            cv2.putText(bgr, "detector failed", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Stamp the detector name at the bottom so user knows what they're seeing
        if detector_name:
            h = bgr.shape[0]
            cv2.putText(bgr, detector_name, (4, h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 0), 1, cv2.LINE_AA)

        return bgr

    # ── Event handlers ────────────────────────────────────────────────
    def _handle_mode_change(self, _event=None) -> None:
        label = self.var_mode_label.get()
        mode = _LABEL_TO_MODE.get(label, MODE_AUTO)
        self._on_override_change(mode)
