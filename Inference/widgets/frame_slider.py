"""FrameSlider — V1's frame scrubber + 'Frame: N / total (best)' label.

Lives above the canvas in Tab 2. Slider from frame_range[0] to
frame_range[1]; emits the current index on change.

Public API:
  - FrameSlider(parent, on_change=cb)
  - .set_range(first, last, best=None)
  - .set_value(idx)  # without firing the callback
  - .get_value() -> int
  - .set_enabled(bool)
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional


class FrameSlider(ttk.Frame):

    def __init__(
        self,
        parent: tk.Misc,
        on_change: Optional[Callable[[int], None]] = None,
    ) -> None:
        super().__init__(parent, padding=(4, 2))
        self._on_change = on_change or (lambda i: None)
        self._first = 0
        self._last = 0
        self._best: Optional[int] = None
        self._suppress = False

        self.var_idx = tk.IntVar(value=0)
        self.var_label = tk.StringVar(value="Frame: – / –")

        self.scale = ttk.Scale(
            self, from_=0, to=0, orient="horizontal",
            variable=self.var_idx, command=self._handle_change)
        self.scale.pack(side="left", fill="x", expand=True)

        ttk.Label(self, textvariable=self.var_label,
                  font=("Consolas", 9), foreground="#404040",
                  width=22, anchor="e").pack(side="left", padx=(8, 0))

    def _handle_change(self, _value=None) -> None:
        if self._suppress:
            return
        try:
            idx = int(round(float(self.var_idx.get())))
        except (ValueError, tk.TclError):
            return
        self._refresh_label(idx)
        self._on_change(idx)

    def _refresh_label(self, idx: int) -> None:
        if self._last == self._first:
            self.var_label.set("Frame: – / –")
            return
        n = self._last - self._first + 1
        offset = idx - self._first + 1
        suffix = " (best)" if (self._best is not None and idx == self._best) else ""
        self.var_label.set(f"Frame: {idx}    {offset}/{n}{suffix}")

    # ── Public API ────────────────────────────────────────────────────
    def set_range(self, first: int, last: int,
                   best: Optional[int] = None) -> None:
        self._first = int(first)
        self._last = int(last)
        self._best = int(best) if best is not None else None
        self.scale.configure(from_=self._first, to=self._last)
        # Pick a sensible initial position (best frame, else first)
        initial = self._best if self._best is not None else self._first
        self.set_value(initial)

    def set_value(self, idx: int, fire_callback: bool = False) -> None:
        self._suppress = not fire_callback
        try:
            self.var_idx.set(int(idx))
            self._refresh_label(int(idx))
        finally:
            self._suppress = False

    def get_value(self) -> int:
        try:
            return int(round(float(self.var_idx.get())))
        except (ValueError, tk.TclError):
            return self._first

    def set_enabled(self, enabled: bool) -> None:
        self.scale.state(["!disabled"] if enabled else ["disabled"])
