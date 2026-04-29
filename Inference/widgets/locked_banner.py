"""LockedBanner — always-visible top bar showing model + calibration + s_c.

The "honesty layer" of V2: at all times the user can see exactly what
model is loaded, what calibration is driving inference, and what
deployment camera scale they have set.

Three rows:
  1. MODEL          [filename]                              [Browse] [Load]
  2. CALIBRATION    method · ρ=… · σ_floor=… · sha=…       [from checkpoint]
  3. DEPLOYMENT     s_c [editable] px/mm                    ⚠ verify

Public API:
  - LockedBanner(parent, on_browse=cb, on_load=cb, on_s_c_change=cb)
  - .update_state(engine, model_path)  — refreshes display from engine state
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Optional


# Colour palette — kept consistent with status bar later
_COLOUR_OK = "#1a5d1a"          # dark green
_COLOUR_WARN = "#a04000"        # orange
_COLOUR_ERR = "#a01515"         # red
_COLOUR_NEUTRAL = "#404040"     # dark gray
_COLOUR_MUTED = "#707070"       # mid gray for secondary text
_COLOUR_BADGE_BG = "#e8f4e8"    # very pale green
_COLOUR_BADGE_FG = "#1a5d1a"    # dark green


class LockedBanner(ttk.Frame):
    """Top-of-window status banner. Reads from engine state, never mutates it."""

    def __init__(
        self,
        parent: tk.Misc,
        on_browse: Optional[Callable[[], None]] = None,
        on_load: Optional[Callable[[], None]] = None,
        on_s_c_change: Optional[Callable[[float], None]] = None,
    ) -> None:
        super().__init__(parent, padding=(8, 6))
        self._on_browse = on_browse or (lambda: None)
        self._on_load = on_load or (lambda: None)
        self._on_s_c_change = on_s_c_change or (lambda v: None)

        # State held in tk vars
        self.var_model_filename = tk.StringVar(value="(no model loaded)")
        self.var_model_status = tk.StringVar(value="✗")
        self.var_calibration_text = tk.StringVar(value="—")
        self.var_calibration_source = tk.StringVar(value="(load a checkpoint to populate)")
        self.var_s_c = tk.StringVar(value="")
        self.var_s_c_hint = tk.StringVar(value="")

        # Cached for s_c validation
        self._s_calib_for_compare: Optional[float] = None

        self._build()

    # ── Build UI ──────────────────────────────────────────────────────
    def _build(self) -> None:
        # Row 0: MODEL
        row0 = ttk.Frame(self)
        row0.pack(fill="x", pady=(0, 4))
        ttk.Label(row0, text="MODEL", width=11, anchor="w",
                  foreground=_COLOUR_MUTED,
                  font=("TkDefaultFont", 8, "bold")).pack(side="left")
        self.lbl_model_status = ttk.Label(
            row0, textvariable=self.var_model_status,
            foreground=_COLOUR_ERR, font=("TkDefaultFont", 11, "bold"),
            width=2)
        self.lbl_model_status.pack(side="left", padx=(0, 6))
        ttk.Label(row0, textvariable=self.var_model_filename,
                  foreground=_COLOUR_NEUTRAL).pack(side="left")
        ttk.Button(row0, text="Browse…", command=self._on_browse).pack(
            side="right", padx=(4, 0))
        ttk.Button(row0, text="Load", command=self._on_load).pack(
            side="right", padx=(4, 0))

        # Row 1: CALIBRATION (read-only, locked from checkpoint)
        row1 = ttk.Frame(self)
        row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="CALIBRATION", width=11, anchor="w",
                  foreground=_COLOUR_MUTED,
                  font=("TkDefaultFont", 8, "bold")).pack(side="left")
        ttk.Label(row1, textvariable=self.var_calibration_text,
                  foreground=_COLOUR_NEUTRAL,
                  font=("Consolas", 9)).pack(side="left", padx=(0, 8))
        # Source badge — shows where the calibration came from
        self.lbl_calibration_source = tk.Label(
            row1, textvariable=self.var_calibration_source,
            background=_COLOUR_BADGE_BG, foreground=_COLOUR_BADGE_FG,
            font=("TkDefaultFont", 8), padx=6, pady=1, borderwidth=0)
        self.lbl_calibration_source.pack(side="left")

        # Row 2: DEPLOYMENT CAMERA SCALE (the one editable required field)
        row2 = ttk.Frame(self)
        row2.pack(fill="x", pady=(4, 0))
        ttk.Label(row2, text="DEPLOYMENT", width=11, anchor="w",
                  foreground=_COLOUR_MUTED,
                  font=("TkDefaultFont", 8, "bold")).pack(side="left")
        ttk.Label(row2, text="s_c =",
                  foreground=_COLOUR_NEUTRAL).pack(side="left", padx=(0, 4))
        self.entry_s_c = ttk.Entry(row2, textvariable=self.var_s_c, width=12)
        self.entry_s_c.pack(side="left")
        self.entry_s_c.bind("<FocusOut>", self._handle_s_c_commit)
        self.entry_s_c.bind("<Return>", self._handle_s_c_commit)
        ttk.Label(row2, text="px/mm  (deployment camera)",
                  foreground=_COLOUR_MUTED,
                  font=("TkDefaultFont", 8)).pack(side="left", padx=(4, 8))
        self.lbl_s_c_hint = ttk.Label(
            row2, textvariable=self.var_s_c_hint,
            font=("TkDefaultFont", 8, "italic"))
        self.lbl_s_c_hint.pack(side="left")

    # ── Public API ────────────────────────────────────────────────────
    def update_state(
        self,
        engine: Any,
        model_path: str,
        s_c_setting: Optional[float] = None,
    ) -> None:
        """Refresh display from current engine state and settings.

        Pass ``engine=None`` to render the no-model state.
        ``s_c_setting`` is the current s_c value from settings dict; we
        use it to populate the entry (without firing on_change).
        """
        # MODEL row
        if model_path:
            from pathlib import Path as _P
            self.var_model_filename.set(_P(model_path).name)
        else:
            self.var_model_filename.set("(no model loaded)")
        if engine is not None and getattr(engine, 'model', None) is not None:
            self.var_model_status.set("✓")
            self.lbl_model_status.configure(foreground=_COLOUR_OK)
        else:
            self.var_model_status.set("✗")
            self.lbl_model_status.configure(foreground=_COLOUR_ERR)

        # CALIBRATION row — read directly from engine.calibration_model
        cm = getattr(engine, 'calibration_model', None) if engine is not None else None
        if cm is not None:
            method = cm.method
            rho = cm.rho_px_per_mm
            aux_label = "σ_floor" if cm.method in ("quadrature", "hybrid") else "σ_0"
            aux_val = (cm.sigma_floor_calib_px if cm.method in ("quadrature", "hybrid")
                       else cm.sigma_0_calib_px) or 0.0
            sha_prefix = cm.sha256()[:12] if hasattr(cm, 'sha256') else "?"
            self.var_calibration_text.set(
                f"{method}   ρ={rho:.4f}   {aux_label}={aux_val:.4f}   sha={sha_prefix}…"
            )
            self.var_calibration_source.set(" ✓ from checkpoint ")
            self.lbl_calibration_source.configure(
                background=_COLOUR_BADGE_BG, foreground=_COLOUR_BADGE_FG)
            self._s_calib_for_compare = cm.s_calib_px_per_mm
        elif engine is not None:
            self.var_calibration_text.set("(legacy checkpoint — no CalibrationModel block)")
            self.var_calibration_source.set(" ⚠ legacy fallback ")
            self.lbl_calibration_source.configure(
                background="#fff4e0", foreground=_COLOUR_WARN)
            self._s_calib_for_compare = None
        else:
            self.var_calibration_text.set("—")
            self.var_calibration_source.set("(load a checkpoint to populate)")
            self.lbl_calibration_source.configure(
                background="#f0f0f0", foreground=_COLOUR_MUTED)
            self._s_calib_for_compare = None

        # DEPLOYMENT row — only update entry if it differs from setting
        if s_c_setting is not None:
            try:
                current = float(self.var_s_c.get())
            except (ValueError, tk.TclError):
                current = None
            if current is None or abs(current - float(s_c_setting)) > 1e-6:
                self.var_s_c.set(f"{float(s_c_setting):.4f}".rstrip('0').rstrip('.'))
        self._refresh_s_c_hint()

    # ── s_c validation hint ───────────────────────────────────────────
    def _refresh_s_c_hint(self) -> None:
        try:
            v = float(self.var_s_c.get())
        except (ValueError, tk.TclError):
            self.var_s_c_hint.set("⚠ enter a positive number")
            self.lbl_s_c_hint.configure(foreground=_COLOUR_ERR)
            return
        if v <= 0:
            self.var_s_c_hint.set("⚠ must be positive")
            self.lbl_s_c_hint.configure(foreground=_COLOUR_ERR)
            return
        if self._s_calib_for_compare is not None and self._s_calib_for_compare > 0:
            ratio = v / self._s_calib_for_compare
            if ratio < 0.5 or ratio > 2.0:
                self.var_s_c_hint.set(
                    f"⚠ {ratio:.2f}× the calibration scale "
                    f"({self._s_calib_for_compare:.2f}) — check your camera"
                )
                self.lbl_s_c_hint.configure(foreground=_COLOUR_WARN)
                return
            if abs(ratio - 1.0) < 1e-3:
                self.var_s_c_hint.set("✓ matches calibration camera")
                self.lbl_s_c_hint.configure(foreground=_COLOUR_OK)
                return
            self.var_s_c_hint.set(f"({ratio:.2f}× the calibration scale)")
            self.lbl_s_c_hint.configure(foreground=_COLOUR_MUTED)
            return
        # No checkpoint loaded → just say OK
        self.var_s_c_hint.set("")

    def _handle_s_c_commit(self, _event=None) -> None:
        """Called on Enter or focus-out — validates and notifies App."""
        try:
            v = float(self.var_s_c.get())
        except (ValueError, tk.TclError):
            return
        if v <= 0:
            return
        self._on_s_c_change(v)
        self._refresh_s_c_hint()
