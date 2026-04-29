"""SaveDialog — checkbox dialog for choosing what to save.

Used in two contexts:
  - Single-cine Save (after Process completes): user clicks Save, dialog
    pops up, returns selected SaveOptions + the target folder.
  - Batch pre-flight: user clicks Batch folder, picks a folder, dialog
    asks what to capture per-cine before the long-running batch starts.

Both contexts share the same checkbox layout. The dialog header text
adapts to the context.

Public API:
  - SaveDialog(parent, defaults: SaveOptions, target_folder: Path,
                context: 'single' | 'batch', cine_name: str = '')
  - .result: SaveOptions | None  (None on cancel)
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Optional

from Inference.run_io import SaveOptions


_COLOUR_MUTED = "#707070"
_COLOUR_NEUTRAL = "#404040"
_COLOUR_ACCENT = "#1a5d1a"


class SaveDialog(tk.Toplevel):

    def __init__(
        self,
        parent: tk.Misc,
        defaults: SaveOptions,
        target_folder: Path,
        context: str = "single",         # 'single' or 'batch'
        cine_name: str = "",
        input_mode: str = "cine",        # 'cine' | 'png'
    ) -> None:
        super().__init__(parent)
        self.title("Save options")
        self.transient(parent)
        self.grab_set()
        self.resizable(False, False)

        self.result: Optional[SaveOptions] = None
        self._defaults = defaults
        self._target_folder = target_folder
        self._context = context
        self._cine_name = cine_name
        self._input_mode = input_mode

        self._vars = {
            "raw_frame": tk.BooleanVar(value=defaults.raw_frame),
            "overlay":   tk.BooleanVar(value=defaults.overlay),
            "crop":      tk.BooleanVar(value=defaults.crop),
            "processed": tk.BooleanVar(value=defaults.processed),
            "csv":       tk.BooleanVar(value=defaults.results_csv),
        }

        self._build()
        self.wait_window(self)

    def _build(self) -> None:
        # Header
        if self._context == "batch":
            header = (f"Batch processing — what to capture per cine?\n"
                       f"(CSV is always written; image options apply to "
                       f"every cine in the folder)")
            csv_locked = True
        else:
            header = f"Single cine: {self._cine_name or '(unknown)'}"
            csv_locked = False

        ttk.Label(self, text=header,
                  foreground=_COLOUR_NEUTRAL, justify="left",
                  font=("TkDefaultFont", 9, "bold"),
                  wraplength=420).pack(
            anchor="w", padx=12, pady=(10, 6))

        # Checkboxes
        cb_frame = ttk.LabelFrame(self, text="Save which views?", padding=10)
        cb_frame.pack(fill="x", padx=12, pady=4)
        # In PNG mode, raw_frame and overlay don't apply (the PNG IS
        # the crop — there's no source frame and no geometry overlay).
        png_mode = (self._input_mode == "png")
        rows = [
            ("raw_frame", "Raw best frame",
             "Full cine frame at the chosen best frame (no annotations)"),
            ("overlay",   "Raw frame + overlay",
             "Full frame with V1-style geometry annotations on top"),
            ("crop",      "Extracted crop",
             "Crop region (square, centred on droplet)"),
            ("processed", "Processed (model input)",
             "The exact image fed to the model — what the model sees"),
            ("csv",       "Result + metadata (CSV/YAML/TXT)",
             "Results table row + reproducibility yaml + summary text"),
        ]
        if png_mode:
            # Drop the cine-only options — they have no source data
            rows = [r for r in rows
                    if r[0] not in ("raw_frame", "overlay")]
            # And force the corresponding vars off so they don't get
            # picked up in result if previously persisted as True
            self._vars["raw_frame"].set(False)
            self._vars["overlay"].set(False)
        for key, label, tooltip in rows:
            row = ttk.Frame(cb_frame)
            row.pack(fill="x", pady=1)
            cb = ttk.Checkbutton(
                row, text=label, variable=self._vars[key])
            if csv_locked and key == "csv":
                cb.state(["disabled", "selected"])
            cb.pack(side="left")
            ttk.Label(row, text=f"   — {tooltip}",
                      foreground=_COLOUR_MUTED,
                      font=("TkDefaultFont", 8)).pack(side="left", padx=(4, 0))

        # Target folder display
        target_frame = ttk.LabelFrame(
            self, text="Output folder", padding=8)
        target_frame.pack(fill="x", padx=12, pady=(8, 4))
        ttk.Label(target_frame,
                  text=str(self._target_folder),
                  foreground=_COLOUR_NEUTRAL,
                  font=("Consolas", 9), wraplength=420).pack(
            anchor="w")
        ttk.Label(
            target_frame,
            text=("(Auto-named — same source folder + same date "
                   "appends to the existing session)"
                   if self._context == "single"
                   else "(Timestamped batch folder — fresh per click)"),
            foreground=_COLOUR_MUTED,
            font=("TkDefaultFont", 8, "italic")).pack(
            anchor="w", pady=(2, 0))

        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=12, pady=(8, 12))
        ttk.Button(btn_frame, text="Save",
                   command=self._on_save).pack(side="right", padx=4)
        ttk.Button(btn_frame, text="Cancel",
                   command=self.destroy).pack(side="right")

    def _on_save(self) -> None:
        self.result = SaveOptions(
            raw_frame=self._vars["raw_frame"].get(),
            overlay=self._vars["overlay"].get(),
            crop=self._vars["crop"].get(),
            processed=self._vars["processed"].get(),
            results_csv=self._vars["csv"].get(),
        )
        self.destroy()
