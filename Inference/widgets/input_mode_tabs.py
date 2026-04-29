"""InputModeTabs — tab strip selecting between .cine input and Precropped PNG input.

Sits between the locked banner and the main Notebook. Each tab contains
its own input controls. The active tab determines what Process / Batch
/ Prev / Next operate on.

Layout:
  ┌─[.cine input]──[Precropped PNG]─────────────────────────┐
  │                                                          │
  │  per-mode controls land here (see below)                 │
  │                                                          │
  └──────────────────────────────────────────────────────────┘

`.cine input` panel:
    .cine file:    [path]                 [Browse] [Open]
    Watch folder:  [path]                 [Browse] [☐ Auto]

`Precropped PNG` panel:
    PNG folder:    [path]                 [Browse] [☐ Auto]
    Current PNG:   sphere0959v_crop.png   (set by Prev/Next or Auto)

Public API:
  - InputModeTabs(parent, callbacks=…)
  - .set_mode(mode)             # 'cine' | 'png', without firing change
  - .get_mode() -> str
  - .set_cine_path(p) / .get_cine_path()
  - .set_watch_folder(p) / .get_watch_folder()
  - .set_png_folder(p) / .get_png_folder()
  - .set_current_png_label(name)
  - .is_auto_enabled() -> bool       # which checkbox depends on mode

Callbacks (all optional):
    on_mode_change(mode)
    on_cine_browse() / on_cine_open()
    on_cine_watch_browse() / on_cine_auto_toggle(enabled)
    on_png_browse() / on_png_auto_toggle(enabled)
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from Inference.widgets.tooltips import Tooltip


_COLOUR_MUTED = "#707070"


MODE_CINE = "cine"
MODE_PNG = "png"
ALL_INPUT_MODES = (MODE_CINE, MODE_PNG)


class InputModeTabs(ttk.Frame):

    def __init__(
        self,
        parent: tk.Misc,
        on_mode_change: Optional[Callable[[str], None]] = None,
        on_cine_browse: Optional[Callable[[], None]] = None,
        on_cine_open: Optional[Callable[[], None]] = None,
        on_cine_watch_browse: Optional[Callable[[], None]] = None,
        on_cine_auto_toggle: Optional[Callable[[bool], None]] = None,
        on_cine_watch_change: Optional[Callable[[str], None]] = None,
        on_png_browse: Optional[Callable[[], None]] = None,
        on_png_auto_toggle: Optional[Callable[[bool], None]] = None,
        on_png_folder_change: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__(parent, padding=(8, 2))
        self._on_mode_change = on_mode_change or (lambda m: None)
        self._on_cine_browse = on_cine_browse or (lambda: None)
        self._on_cine_open = on_cine_open or (lambda: None)
        self._on_cine_watch_browse = on_cine_watch_browse or (lambda: None)
        self._on_cine_auto_toggle = on_cine_auto_toggle or (lambda e: None)
        self._on_cine_watch_change = on_cine_watch_change or (lambda p: None)
        self._on_png_browse = on_png_browse or (lambda: None)
        self._on_png_auto_toggle = on_png_auto_toggle or (lambda e: None)
        self._on_png_folder_change = on_png_folder_change or (lambda p: None)

        # State variables (the App reads/writes these via the public API)
        self.var_cine_path = tk.StringVar(value="")
        self.var_watch_folder = tk.StringVar(value="")
        self.var_cine_auto = tk.BooleanVar(value=False)
        self.var_png_folder = tk.StringVar(value="")
        self.var_png_auto = tk.BooleanVar(value=False)
        self.var_current_png = tk.StringVar(value="(none)")

        self._suppress_tab_change = False
        self._build()

    # ── Build UI ──────────────────────────────────────────────────────
    def _build(self) -> None:
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="x", expand=False)
        self._tab_cine = ttk.Frame(self.notebook, padding=(8, 6))
        self._tab_png = ttk.Frame(self.notebook, padding=(8, 6))
        self.notebook.add(self._tab_cine, text=".cine input")
        self.notebook.add(self._tab_png, text="Precropped PNG")
        self.notebook.bind("<<NotebookTabChanged>>", self._handle_tab_change)
        self._build_cine_tab()
        self._build_png_tab()

    def _build_cine_tab(self) -> None:
        f = self._tab_cine
        # Row 0: .cine file + Browse + Open
        ttk.Label(f, text=".cine file:", width=14, anchor="w",
                  foreground=_COLOUR_MUTED,
                  font=("TkDefaultFont", 8, "bold")).grid(
            row=0, column=0, sticky="w", pady=2)
        e = ttk.Entry(f, textvariable=self.var_cine_path)
        e.grid(row=0, column=1, sticky="ew", padx=(0, 4), pady=2)
        btn_browse = ttk.Button(f, text="Browse…", width=10,
                                  command=self._on_cine_browse)
        btn_browse.grid(row=0, column=2, padx=(0, 4), pady=2)
        Tooltip(btn_browse, "Browse for a single .cine file to open.")
        btn_open = ttk.Button(f, text="Open", width=8,
                                command=self._on_cine_open)
        btn_open.grid(row=0, column=3, pady=2)
        Tooltip(btn_open,
                "Load the selected .cine: pick best frame, extract crop, "
                "auto-pick preprocessing. No model run yet — click Process for that.")

        # Row 1: Watch folder + Browse + Auto checkbox
        ttk.Label(f, text="Watch folder:", width=14, anchor="w",
                  foreground=_COLOUR_MUTED,
                  font=("TkDefaultFont", 8, "bold")).grid(
            row=1, column=0, sticky="w", pady=2)
        e2 = ttk.Entry(f, textvariable=self.var_watch_folder)
        e2.grid(row=1, column=1, sticky="ew", padx=(0, 4), pady=2)
        e2.bind("<FocusOut>",
                  lambda _e: self._on_cine_watch_change(
                      self.var_watch_folder.get()))
        e2.bind("<Return>",
                  lambda _e: self._on_cine_watch_change(
                      self.var_watch_folder.get()))
        btn_w = ttk.Button(f, text="Browse…", width=10,
                             command=self._on_cine_watch_browse)
        btn_w.grid(row=1, column=2, padx=(0, 4), pady=2)
        Tooltip(btn_w, "Pick a folder containing .cine files for "
                       "Prev/Next stepping and Auto polling.")
        cb = ttk.Checkbutton(
            f, text="Auto", variable=self.var_cine_auto,
            command=lambda: self._on_cine_auto_toggle(
                self.var_cine_auto.get()))
        cb.grid(row=1, column=3, sticky="w", pady=2)
        Tooltip(cb, "When ticked, polls the watch folder every 10s "
                    "and auto-loads the most recently modified .cine.")
        f.columnconfigure(1, weight=1)

    def _build_png_tab(self) -> None:
        f = self._tab_png
        # Row 0: PNG folder + Browse + Auto
        ttk.Label(f, text="PNG folder:", width=14, anchor="w",
                  foreground=_COLOUR_MUTED,
                  font=("TkDefaultFont", 8, "bold")).grid(
            row=0, column=0, sticky="w", pady=2)
        e = ttk.Entry(f, textvariable=self.var_png_folder)
        e.grid(row=0, column=1, sticky="ew", padx=(0, 4), pady=2)
        e.bind("<FocusOut>",
                lambda _e: self._on_png_folder_change(
                    self.var_png_folder.get()))
        e.bind("<Return>",
                lambda _e: self._on_png_folder_change(
                    self.var_png_folder.get()))
        btn_browse = ttk.Button(f, text="Browse…", width=10,
                                  command=self._on_png_browse)
        btn_browse.grid(row=0, column=2, padx=(0, 4), pady=2)
        Tooltip(btn_browse, "Pick a folder containing pre-cropped PNG "
                            "images. Each PNG is treated as a square "
                            "model-ready input.")
        cb = ttk.Checkbutton(
            f, text="Auto", variable=self.var_png_auto,
            command=lambda: self._on_png_auto_toggle(
                self.var_png_auto.get()))
        cb.grid(row=0, column=3, sticky="w", pady=2)
        Tooltip(cb, "When ticked, polls the folder every 10s and "
                    "auto-loads the newest PNG.")
        # Row 1: current PNG name (read-only display)
        ttk.Label(f, text="Current PNG:", width=14, anchor="w",
                  foreground=_COLOUR_MUTED,
                  font=("TkDefaultFont", 8, "bold")).grid(
            row=1, column=0, sticky="w", pady=2)
        ttk.Label(f, textvariable=self.var_current_png,
                  foreground="#404040",
                  font=("Consolas", 9)).grid(
            row=1, column=1, columnspan=3, sticky="w", padx=(0, 4), pady=2)
        f.columnconfigure(1, weight=1)

    # ── Event handlers ────────────────────────────────────────────────
    def _handle_tab_change(self, _event=None) -> None:
        if self._suppress_tab_change:
            return
        self._on_mode_change(self.get_mode())

    # ── Public API ────────────────────────────────────────────────────
    def get_mode(self) -> str:
        idx = self.notebook.index(self.notebook.select())
        return MODE_CINE if idx == 0 else MODE_PNG

    def set_mode(self, mode: str, fire_callback: bool = False) -> None:
        # Always suppress the event-driven callback so we don't double-fire.
        # If fire_callback=True, we call the callback explicitly afterwards
        # — more deterministic than relying on Notebook's event for
        # programmatic select() (which is unreliable on some Tk builds).
        target = self._tab_cine if mode == MODE_CINE else self._tab_png
        self._suppress_tab_change = True
        try:
            self.notebook.select(target)
        finally:
            self._suppress_tab_change = False
        if fire_callback:
            self._on_mode_change(mode)

    # cine controls
    def set_cine_path(self, p: str) -> None: self.var_cine_path.set(p or "")
    def get_cine_path(self) -> str: return self.var_cine_path.get().strip()

    def set_watch_folder(self, p: str) -> None:
        self.var_watch_folder.set(p or "")
    def get_watch_folder(self) -> str:
        return self.var_watch_folder.get().strip()

    def is_cine_auto_enabled(self) -> bool:
        return bool(self.var_cine_auto.get())
    def set_cine_auto(self, enabled: bool) -> None:
        self.var_cine_auto.set(bool(enabled))

    # png controls
    def set_png_folder(self, p: str) -> None:
        self.var_png_folder.set(p or "")
    def get_png_folder(self) -> str:
        return self.var_png_folder.get().strip()

    def is_png_auto_enabled(self) -> bool:
        return bool(self.var_png_auto.get())
    def set_png_auto(self, enabled: bool) -> None:
        self.var_png_auto.set(bool(enabled))

    def set_current_png_label(self, name: str) -> None:
        self.var_current_png.set(name or "(none)")

    # convenience: which Auto checkbox is active in the current mode
    def is_auto_enabled(self) -> bool:
        return (self.is_cine_auto_enabled() if self.get_mode() == MODE_CINE
                else self.is_png_auto_enabled())
