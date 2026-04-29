"""WatchFolderBar — folder field + Browse + Auto checkbox.

Sits just below the locked banner. The actual polling loop lives in the
main App (so cancellation + state are unified there).

Public API:
  - WatchFolderBar(parent, on_browse=, on_auto_toggle=, on_folder_change=)
  - .set_folder(path: str)
  - .get_folder() -> str
  - .is_auto_enabled() -> bool
  - .set_auto(enabled: bool)  # without firing the toggle callback
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from Inference.widgets.tooltips import Tooltip


_COLOUR_MUTED = "#707070"


class WatchFolderBar(ttk.Frame):
    """Watch folder input + Auto-poll checkbox."""

    def __init__(
        self,
        parent: tk.Misc,
        on_browse: Optional[Callable[[], None]] = None,
        on_auto_toggle: Optional[Callable[[bool], None]] = None,
        on_folder_change: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__(parent, padding=(8, 2))
        self._on_browse = on_browse or (lambda: None)
        self._on_auto_toggle = on_auto_toggle or (lambda enabled: None)
        self._on_folder_change = on_folder_change or (lambda path: None)

        self.var_folder = tk.StringVar(value="")
        self.var_auto = tk.BooleanVar(value=False)

        self._build()

    def _build(self) -> None:
        ttk.Label(self, text="Watch folder:", width=14, anchor="w",
                  foreground=_COLOUR_MUTED,
                  font=("TkDefaultFont", 8, "bold")).pack(side="left")
        entry = ttk.Entry(self, textvariable=self.var_folder)
        entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        entry.bind("<FocusOut>", lambda _e: self._on_folder_change(
            self.var_folder.get()))
        entry.bind("<Return>", lambda _e: self._on_folder_change(
            self.var_folder.get()))

        btn_browse = ttk.Button(self, text="Browse…", command=self._on_browse,
                                  width=10)
        btn_browse.pack(side="left", padx=(0, 4))
        Tooltip(btn_browse,
                "Pick a folder to watch. When Auto is checked, the GUI "
                "loads the most-recently-modified .cine in this folder, "
                "and re-checks every 10 seconds for new files.")

        cb_auto = ttk.Checkbutton(
            self, text="Auto-load latest", variable=self.var_auto,
            command=lambda: self._on_auto_toggle(self.var_auto.get()))
        cb_auto.pack(side="left")
        Tooltip(cb_auto,
                "When checked, the GUI polls the watch folder every 10 "
                "seconds. If a newer .cine appears, it loads it "
                "automatically (no inference unless you click Process).")

    # ── Public API ────────────────────────────────────────────────────
    def set_folder(self, path: str) -> None:
        self.var_folder.set(path)

    def get_folder(self) -> str:
        return self.var_folder.get().strip()

    def is_auto_enabled(self) -> bool:
        return bool(self.var_auto.get())

    def set_auto(self, enabled: bool) -> None:
        """Set the Auto checkbox without firing the toggle callback."""
        self.var_auto.set(bool(enabled))
