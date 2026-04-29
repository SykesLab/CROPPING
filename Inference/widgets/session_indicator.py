"""SessionIndicator — small strip showing the current session at a glance.

Sits between the active-mode strip and the action bar. One line:

  "Session: 4mm-borosilicate (today)  ·  3 cines saved  ·  view folder ⇗"

Public API:
  - SessionIndicator(parent, on_view_folder=cb)
  - .set_state(folder: Path | None, count: int)
        folder=None → strip shows "no active session"
"""

from __future__ import annotations

import datetime
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable, Optional


_BG_ACTIVE = "#eef3ee"
_FG_ACTIVE = "#1a5d1a"
_BG_INACTIVE = "#f0f0f0"
_FG_INACTIVE = "#707070"


class SessionIndicator(ttk.Frame):

    def __init__(
        self,
        parent: tk.Misc,
        on_view_folder: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(parent, padding=(8, 0))
        self._on_view_folder = on_view_folder or (lambda: None)
        self._folder: Optional[Path] = None

        self.var_text = tk.StringVar(
            value="Session: (no active session — Save a cine to start one)")

        # Container coloured background
        self._bg_frame = tk.Frame(self, background=_BG_INACTIVE)
        self._bg_frame.pack(fill="x")
        self.lbl_text = tk.Label(
            self._bg_frame, textvariable=self.var_text,
            background=_BG_INACTIVE, foreground=_FG_INACTIVE,
            font=("TkDefaultFont", 9), padx=8, pady=2, anchor="w")
        self.lbl_text.pack(side="left")
        self.btn_view = tk.Label(
            self._bg_frame, text="view folder ⇗",
            background=_BG_INACTIVE, foreground=_FG_INACTIVE,
            font=("TkDefaultFont", 9, "underline"),
            padx=8, pady=2, cursor="hand2")
        self.btn_view.pack(side="right")
        self.btn_view.bind("<Button-1>", lambda _e: self._on_view_folder())

    # ── Public API ────────────────────────────────────────────────────
    def set_state(self, folder: Optional[Path], count: int) -> None:
        if folder is None or count <= 0:
            self.var_text.set(
                "Session: (no active session — Save a cine to start one)")
            self.lbl_text.configure(
                background=_BG_INACTIVE, foreground=_FG_INACTIVE)
            self._bg_frame.configure(background=_BG_INACTIVE)
            self.btn_view.configure(
                background=_BG_INACTIVE, foreground=_FG_INACTIVE)
            self._folder = None
            return
        # Active session
        # Folder name: "2026-04-29_4mm-borosilicate"
        # Show just the source-folder portion + a "today" hint if
        # today's date matches.
        name = folder.name
        date_part, _, source = name.partition("_")
        try:
            date_obj = datetime.date.fromisoformat(date_part)
            today_hint = ("today" if date_obj == datetime.date.today()
                          else date_obj.isoformat())
        except ValueError:
            today_hint = date_part
        suffix = "cine" if count == 1 else "cines"
        self.var_text.set(
            f"Session: {source} ({today_hint})  ·  "
            f"{count} {suffix} saved")
        self.lbl_text.configure(
            background=_BG_ACTIVE, foreground=_FG_ACTIVE)
        self._bg_frame.configure(background=_BG_ACTIVE)
        self.btn_view.configure(
            background=_BG_ACTIVE, foreground=_FG_ACTIVE)
        self._folder = folder

    def get_folder(self) -> Optional[Path]:
        return self._folder
