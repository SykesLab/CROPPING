"""Tooltip — lightweight hover tooltip helper for Tk widgets.

Standard pattern: bind <Enter> to show, <Leave>/<ButtonPress> to hide.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class Tooltip:
    """Attach a hover tooltip to any Tk widget.

    Usage:
        Tooltip(my_button, "Process the loaded cine through the model")
    """

    _SHOW_DELAY_MS = 500
    _BG = "#ffffe0"
    _FG = "#000000"

    def __init__(self, widget: tk.Misc, text: str,
                  wraplength: int = 320) -> None:
        self.widget = widget
        self.text = text
        self.wraplength = wraplength
        self._tip: tk.Toplevel | None = None
        self._show_id: str | None = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)
        widget.bind("<ButtonPress>", self._hide)

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self._show_id = self.widget.after(self._SHOW_DELAY_MS, self._show)

    def _cancel(self) -> None:
        if self._show_id is not None:
            try:
                self.widget.after_cancel(self._show_id)
            except Exception:
                pass
            self._show_id = None

    def _show(self) -> None:
        if self._tip is not None:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(self._tip, text=self.text,
                        background=self._BG, foreground=self._FG,
                        relief="solid", borderwidth=1,
                        padx=6, pady=3,
                        font=("TkDefaultFont", 9),
                        wraplength=self.wraplength,
                        justify="left")
        lbl.pack()

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None
