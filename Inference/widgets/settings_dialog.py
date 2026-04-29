"""SettingsDialogV2 — three-section reorganised settings.

Splits the historically-undifferentiated parameter list into three
clearly-labelled sections:

  1. From Checkpoint — read-only display of the calibration block.
     Shows the user exactly what's locked-in from the loaded model.
  2. Required — just `s_c` (deployment camera scale). The only field
     the user genuinely needs to set per deployment.
  3. Advanced — preprocessing tweaks (flatten_mode, inner_margin_px,
     feather_px, crop_size, device).

Public API:
  - SettingsDialogV2(parent, settings, engine, on_save=cb)
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable, Dict, Optional

# Reuse the descriptive labels from preview_row to stay consistent
from Inference.widgets.preview_row import (
    MODE_LABELS as _FLATTEN_MODE_LABELS,
    _LABEL_TO_MODE as _FLATTEN_LABEL_TO_MODE,
    _MODE_TO_LABEL as _FLATTEN_MODE_TO_LABEL,
)


_COLOUR_OK = "#1a5d1a"
_COLOUR_WARN = "#a04000"
_COLOUR_ERR = "#a01515"
_COLOUR_MUTED = "#707070"
_COLOUR_NEUTRAL = "#404040"
_COLOUR_LOCKED_BG = "#e8f4e8"
_COLOUR_REQUIRED_BG = "#fffae0"


class SettingsDialogV2(tk.Toplevel):
    """Modal dialog with three clearly-separated sections."""

    def __init__(
        self,
        parent: tk.Misc,
        settings: Dict[str, Any],
        engine: Optional[Any] = None,
        on_save: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        super().__init__(parent)
        self.title("Settings — V2")
        self.transient(parent)
        self.grab_set()
        self.resizable(False, False)

        self._settings = dict(settings)
        self._engine = engine
        self._on_save = on_save
        self._has_calibration_model = (
            engine is not None
            and getattr(engine, 'calibration_model', None) is not None
        )
        self._entries: Dict[str, tk.StringVar] = {}
        self._build()
        self.wait_window(self)

    # ── Build UI ──────────────────────────────────────────────────────
    def _build(self) -> None:
        # Section 1: From Checkpoint
        self._build_locked_section()
        # Section 2: Required
        self._build_required_section()
        # Section 3: Advanced
        self._build_advanced_section()
        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=10, pady=(8, 10))
        ttk.Button(btn_frame, text="Save", command=self._save).pack(
            side="right", padx=4)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(
            side="right")

    def _build_locked_section(self) -> None:
        f = ttk.LabelFrame(self,
                            text="1. From Checkpoint  (locked, source of truth)",
                            padding=8)
        f.pack(fill="x", padx=10, pady=(10, 4))
        if self._has_calibration_model:
            cm = self._engine.calibration_model
            sha = cm.sha256()[:16] if hasattr(cm, 'sha256') else "?"
            header = (f"✓ Loaded — method: {cm.method}  "
                       f"sha256: {sha}…")
            ttk.Label(f, text=header,
                      foreground=_COLOUR_OK,
                      font=("TkDefaultFont", 9, "bold"),
                      background=_COLOUR_LOCKED_BG).pack(
                anchor="w", pady=(0, 4))
            # Show all the values that came from the checkpoint
            rows = [
                ("rho_px_per_mm", f"{cm.rho_px_per_mm:.6f}"),
            ]
            if cm.method in ("quadrature", "hybrid"):
                rows.append(("sigma_floor_calib_px",
                              f"{(cm.sigma_floor_calib_px or 0):.6f}"))
            else:
                rows.append(("sigma_0_calib_px",
                              f"{(cm.sigma_0_calib_px or 0):.6f}"))
            rows.extend([
                ("s_calib_px_per_mm",
                  f"{(cm.s_calib_px_per_mm or 0):.6f}"),
                ("sigma_min_trusted_calib_px",
                  f"{cm.sigma_min_trusted_calib_px:.4f}"),
                ("sigma_max_trusted_calib_px",
                  f"{cm.sigma_max_trusted_calib_px:.4f}"),
            ])
            if getattr(cm, 'sigma_max_model_observed_px', None):
                rows.append(("sigma_max_model_observed_px",
                              f"{cm.sigma_max_model_observed_px:.4f}"))
            loo = getattr(cm, 'loo_cv', None) or {}
            if loo:
                rows.append(("rho_std (LOO)",
                              f"{loo.get('rho_std', 0):.6f}"))
                aux_name = loo.get('aux_param_name', 'aux')
                rows.append((f"{aux_name}_std (LOO)",
                              f"{loo.get('aux_param_std', 0):.6f}"))
                rows.append(("loo_mae_mm",
                              f"{loo.get('loo_mae', 0):.4f}"))
            grid = ttk.Frame(f)
            grid.pack(fill="x", pady=(2, 0))
            for i, (label, value) in enumerate(rows):
                ttk.Label(grid, text=label + ":",
                          foreground=_COLOUR_MUTED,
                          font=("TkDefaultFont", 8)).grid(
                    row=i, column=0, sticky="w", padx=(0, 8))
                ttk.Label(grid, text=value,
                          foreground=_COLOUR_NEUTRAL,
                          font=("Consolas", 9)).grid(
                    row=i, column=1, sticky="w")
            ttk.Label(f,
                      text=("These values come from the loaded checkpoint and "
                             "are used directly by the inversion math."),
                      foreground=_COLOUR_MUTED,
                      font=("TkDefaultFont", 8, "italic")).pack(
                anchor="w", pady=(6, 0))
        else:
            ttk.Label(f,
                      text=("⚠ No calibration model in checkpoint "
                             "(or no checkpoint loaded). Inversion will use "
                             "the legacy linear fallback."),
                      foreground=_COLOUR_WARN,
                      font=("TkDefaultFont", 9)).pack(anchor="w")

    def _build_required_section(self) -> None:
        f = ttk.LabelFrame(self,
                            text="2. Required  (deployment-specific)",
                            padding=8)
        f.pack(fill="x", padx=10, pady=4)
        ttk.Label(f,
                  text=("Deployment camera pixel scale (s_c). Must match "
                         "the camera you run inference on."),
                  foreground=_COLOUR_MUTED,
                  font=("TkDefaultFont", 8)).pack(anchor="w", pady=(0, 4))
        row = ttk.Frame(f)
        row.pack(fill="x")
        ttk.Label(row, text="s_c =",
                  foreground=_COLOUR_NEUTRAL,
                  font=("TkDefaultFont", 10, "bold")).pack(
            side="left", padx=(0, 4))
        var = tk.StringVar(
            value=str(self._settings.get("s_c", "")))
        self._entries["s_c"] = var
        entry = ttk.Entry(row, textvariable=var, width=18,
                          font=("Consolas", 10))
        entry.pack(side="left")
        ttk.Label(row, text="px/mm",
                  foreground=_COLOUR_MUTED,
                  font=("TkDefaultFont", 9)).pack(side="left", padx=(4, 0))

    def _build_advanced_section(self) -> None:
        f = ttk.LabelFrame(self,
                            text="3. Advanced  (preprocessing tweaks)",
                            padding=8)
        f.pack(fill="x", padx=10, pady=4)
        ttk.Label(f,
                  text=("Flatten mode picks how each crop is normalised "
                         "before the model. Auto-detect uses sphere fill ratio."),
                  foreground=_COLOUR_MUTED,
                  font=("TkDefaultFont", 8)).pack(anchor="w", pady=(0, 4))
        # flatten_mode dropdown
        mode_row = ttk.Frame(f)
        mode_row.pack(fill="x", pady=2)
        ttk.Label(mode_row, text="Flatten mode:", width=18, anchor="w").grid(
            row=0, column=0, sticky="w")
        cur_mode = self._settings.get("flatten_mode", "auto")
        cur_label = _FLATTEN_MODE_TO_LABEL.get(cur_mode,
                                                  _FLATTEN_MODE_LABELS[0][0])
        var_mode = tk.StringVar(value=cur_label)
        self._entries["flatten_mode"] = var_mode
        ttk.Combobox(mode_row, textvariable=var_mode,
                     values=[label for label, _ in _FLATTEN_MODE_LABELS],
                     state="readonly", width=24).grid(
            row=0, column=1, sticky="w", padx=(4, 0))
        # inner_margin_px
        self._add_int_entry(f, "inner_margin_px",
                             "Inner margin (cfg1 only):", 1)
        # feather_px
        self._add_int_entry(f, "feather_px",
                             "Feather width (cfg4 only):", 2)
        # crop_size
        self._add_int_entry(f, "crop_size", "Crop size:", 3)
        # device
        device_row = ttk.Frame(f)
        device_row.pack(fill="x", pady=2)
        ttk.Label(device_row, text="Device:", width=18, anchor="w").grid(
            row=0, column=0, sticky="w")
        var_device = tk.StringVar(
            value=str(self._settings.get("device", "cpu")))
        self._entries["device"] = var_device
        ttk.Combobox(device_row, textvariable=var_device,
                     values=["cpu", "cuda"], state="readonly",
                     width=10).grid(row=0, column=1, sticky="w", padx=(4, 0))

    def _add_int_entry(self, parent, key: str, label: str, row_idx: int) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=22, anchor="w").grid(
            row=0, column=0, sticky="w")
        var = tk.StringVar(value=str(self._settings.get(key, "")))
        self._entries[key] = var
        ttk.Entry(row, textvariable=var, width=10).grid(
            row=0, column=1, sticky="w", padx=(4, 0))

    # ── Save ──────────────────────────────────────────────────────────
    def _save(self) -> None:
        new: Dict[str, Any] = {}
        # s_c (float, must be > 0)
        try:
            s_c = float(self._entries["s_c"].get())
            if s_c <= 0:
                raise ValueError("must be positive")
            new["s_c"] = s_c
        except ValueError as e:
            messagebox.showerror("Invalid s_c", f"s_c must be a positive number ({e}).")
            return
        # flatten_mode (label → mode)
        label = self._entries["flatten_mode"].get()
        new["flatten_mode"] = _FLATTEN_LABEL_TO_MODE.get(label, "auto")
        # int fields
        for key in ("inner_margin_px", "feather_px", "crop_size"):
            raw = self._entries[key].get().strip()
            if not raw:
                continue
            try:
                new[key] = int(raw)
            except ValueError:
                messagebox.showerror(
                    f"Invalid {key}",
                    f"{key} must be an integer (got '{raw}').")
                return
        # device
        new["device"] = self._entries["device"].get()

        if self._on_save is not None:
            self._on_save(new)
        self.destroy()
