"""StatusBar — bottom-of-window status text + progress bar + validation banners.

Layout (top to bottom):
  ┌─ [optional validation banner row] ─────────────────────────────┐
  └─ status text … … …                       [████████ 65%]        ┘

The validation banner is only visible when there are issues. Three
severities (red / yellow / green) with prepended icon + text.

Public API:
  - StatusBar(parent)
  - .set_status(text) — updates the bottom text
  - .set_banner(severity, text) — show validation banner; severity in
    {'ok', 'warn', 'err', 'none'}; 'none' hides it
  - .start_progress() / .update_progress(0..1) / .stop_progress()
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Literal


_BG_OK = "#e8f4e8"
_FG_OK = "#1a5d1a"
_BG_WARN = "#fff4e0"
_FG_WARN = "#a04000"
_BG_ERR = "#fde0e0"
_FG_ERR = "#a01515"


class StatusBar(ttk.Frame):

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)

        # Validation banner (initially hidden)
        self._banner_frame = tk.Frame(self, height=24)
        self._banner_var = tk.StringVar(value="")
        self._banner_label = tk.Label(
            self._banner_frame, textvariable=self._banner_var,
            font=("TkDefaultFont", 9, "bold"),
            anchor="w", padx=8, pady=2)
        self._banner_label.pack(fill="x")
        # Don't pack the banner frame yet — only when there's a message

        # Bottom row: status + progress
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=8, pady=(2, 4))
        self._status_var = tk.StringVar(value="Ready.")
        ttk.Label(bottom, textvariable=self._status_var,
                  foreground="#404040").pack(side="left")
        self._progress = ttk.Progressbar(
            bottom, mode="determinate", length=200, maximum=1000)
        self._progress.pack(side="right")
        self._progress.pack_forget()  # hidden when idle
        self._progress_visible = False

    # ── Status text ───────────────────────────────────────────────────
    def set_status(self, text: str) -> None:
        self._status_var.set(text)

    # ── Validation banner ─────────────────────────────────────────────
    def set_banner(
        self,
        severity: Literal["ok", "warn", "err", "none"],
        text: str = "",
    ) -> None:
        if severity == "none" or not text:
            self._banner_frame.pack_forget()
            return
        if severity == "ok":
            bg, fg, icon = _BG_OK, _FG_OK, "✓"
        elif severity == "warn":
            bg, fg, icon = _BG_WARN, _FG_WARN, "⚠"
        else:
            bg, fg, icon = _BG_ERR, _FG_ERR, "✗"
        self._banner_var.set(f"{icon}  {text}")
        self._banner_label.configure(background=bg, foreground=fg)
        self._banner_frame.configure(background=bg)
        if not self._banner_frame.winfo_ismapped():
            self._banner_frame.pack(fill="x", before=self.winfo_children()[1])

    # ── Progress ──────────────────────────────────────────────────────
    def start_progress(self) -> None:
        if not self._progress_visible:
            self._progress.pack(side="right")
            self._progress_visible = True
        self._progress["value"] = 0

    def update_progress(self, fraction: float) -> None:
        """fraction in [0, 1]."""
        v = max(0, min(1000, int(round(fraction * 1000))))
        self._progress["value"] = v

    def stop_progress(self) -> None:
        self._progress["value"] = 0
        if self._progress_visible:
            self._progress.pack_forget()
            self._progress_visible = False
