"""ActionBar — primary action buttons with tooltips.

Layout:
  [Open .cine…]  [Process current]  [Batch folder…]  ◀ Prev   Next ▶  N/total   [Settings]

Each button has a hover tooltip explaining what it does. Watch-folder
state (cine count, current position) is rendered inline.

Public API:
  - ActionBar(parent, on_open=cb, on_process=cb, on_batch=cb,
                on_prev=cb, on_next=cb, on_settings=cb)
  - .update_state(model_loaded, cine_loaded, watch_position) — enable/disable
  - .set_position_text(text) — update the "N / total" label
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from Inference.widgets.tooltips import Tooltip


class ActionBar(ttk.Frame):
    """Primary action buttons row."""

    def __init__(
        self,
        parent: tk.Misc,
        on_open: Optional[Callable[[], None]] = None,
        on_process: Optional[Callable[[], None]] = None,
        on_save: Optional[Callable[[], None]] = None,
        on_batch: Optional[Callable[[], None]] = None,
        on_prev: Optional[Callable[[], None]] = None,
        on_next: Optional[Callable[[], None]] = None,
        on_settings: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(parent, padding=(8, 4))
        self._on_open = on_open or (lambda: None)
        self._on_process = on_process or (lambda: None)
        self._on_save = on_save or (lambda: None)
        self._on_batch = on_batch or (lambda: None)
        self._on_prev = on_prev or (lambda: None)
        self._on_next = on_next or (lambda: None)
        self._on_settings = on_settings or (lambda: None)

        self.var_position = tk.StringVar(value="")

        self._build()

    def _build(self) -> None:
        # Open .cine
        self.btn_open = ttk.Button(self, text="Open .cine…",
                                     command=self._on_open)
        self.btn_open.pack(side="left", padx=(0, 4))
        Tooltip(self.btn_open,
                "Browse for a .cine file and load it for viewing. "
                "The file is shown in the canvas with the frame slider — "
                "no inference runs until you click Process.")

        # Process
        self.btn_process = ttk.Button(self, text="Process current",
                                        command=self._on_process)
        self.btn_process.pack(side="left", padx=4)
        Tooltip(self.btn_process,
                "Run the full pipeline on the loaded cine: pick best "
                "frame, crop, auto-preprocess, model forward pass, "
                "invert to defocus mm.")

        # Save (after Process)
        self.btn_save = ttk.Button(self, text="Save…",
                                     command=self._on_save)
        self.btn_save.pack(side="left", padx=4)
        Tooltip(self.btn_save,
                "Save the current Process result to disk. Opens a "
                "checkbox dialog: which views to write (raw / overlay "
                "/ crop / processed) + the always-on results CSV. "
                "Goes into the session folder for the cine's source "
                "folder + today's date.")

        # Batch
        self.btn_batch = ttk.Button(self, text="Batch folder…",
                                      command=self._on_batch)
        self.btn_batch.pack(side="left", padx=4)
        Tooltip(self.btn_batch,
                "Process every .cine in a folder sequentially and "
                "write a CSV summary. No GUI display — pure batch.")

        # Separator
        ttk.Separator(self, orient="vertical").pack(
            side="left", fill="y", padx=8)

        # Prev / Next (watch folder navigation)
        self.btn_prev = ttk.Button(self, text="◀ Prev", width=8,
                                     command=self._on_prev)
        self.btn_prev.pack(side="left", padx=2)
        Tooltip(self.btn_prev,
                "Step to the previous .cine in the watch folder "
                "(alphabetical order).")
        self.btn_next = ttk.Button(self, text="Next ▶", width=8,
                                     command=self._on_next)
        self.btn_next.pack(side="left", padx=2)
        Tooltip(self.btn_next,
                "Step to the next .cine in the watch folder "
                "(alphabetical order).")
        ttk.Label(self, textvariable=self.var_position,
                  foreground="#707070",
                  font=("TkDefaultFont", 8)).pack(
            side="left", padx=(6, 0))

        # Settings (right-aligned)
        self.btn_settings = ttk.Button(self, text="Settings…",
                                         command=self._on_settings)
        self.btn_settings.pack(side="right", padx=(4, 0))
        Tooltip(self.btn_settings,
                "Open the settings dialog: read-only checkpoint info, "
                "deployment camera scale (s_c), preprocessing tweaks.")

    # ── Public API ────────────────────────────────────────────────────
    def update_state(
        self,
        model_loaded: bool,
        cine_loaded: bool,
        prev_enabled: bool = False,
        next_enabled: bool = False,
        result_available: bool = False,
    ) -> None:
        """Enable/disable buttons based on app state."""
        self.btn_process.state(
            ["!disabled" if (model_loaded and cine_loaded) else "disabled"])
        self.btn_save.state(
            ["!disabled" if result_available else "disabled"])
        self.btn_batch.state(
            ["!disabled" if model_loaded else "disabled"])
        self.btn_prev.state(["!disabled" if prev_enabled else "disabled"])
        self.btn_next.state(["!disabled" if next_enabled else "disabled"])

    def set_position_text(self, text: str) -> None:
        self.var_position.set(text)

    def set_process_label(self, label: str) -> None:
        """Update the Process button text — called when active mode changes."""
        self.btn_process.configure(text=label)
