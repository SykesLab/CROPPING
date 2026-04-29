"""ViewModeToggle — radio-style toggle bar for the four Tab 2 view modes.

Layout (single row of buttons, stuck-down/raised state):
  [Raw frame] [Overlay] [Crop] [Processed (model input)]

Public API:
  - ViewModeToggle(parent, on_change=cb)
  - .set_mode(mode_str)
  - .set_enabled(mode_str, enabled)  # grey out modes not yet available
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Optional

from Inference.widgets.tooltips import Tooltip


# Mode constants — internal strings the App uses
MODE_RAW = "raw"
MODE_OVERLAY = "overlay"
MODE_CROP = "crop"
MODE_PROCESSED = "processed"

ALL_MODES = (MODE_RAW, MODE_OVERLAY, MODE_CROP, MODE_PROCESSED)

_LABELS = {
    MODE_RAW:       "Raw frame",
    MODE_OVERLAY:   "Overlay",
    MODE_CROP:      "Crop",
    MODE_PROCESSED: "Processed (model input)",
}

_TOOLTIPS = {
    MODE_RAW:       ("Raw cine frame at the slider's current position. "
                     "No annotations, no preprocessing."),
    MODE_OVERLAY:   ("Raw frame with V1's geometry annotations: droplet "
                     "top/bottom/sphere lines, crop box on the best frame, "
                     "frame number, colour legend."),
    MODE_CROP:      ("The extracted crop region (square, centred on the "
                     "droplet). Static — same regardless of slider."),
    MODE_PROCESSED: ("The exact image fed to the model after the chosen "
                     "flatten mode. THIS is what the model 'sees'."),
}


class ViewModeToggle(ttk.Frame):
    """Radio-style toggle bar of four view-mode buttons."""

    def __init__(
        self,
        parent: tk.Misc,
        on_change: Optional[Callable[[str], None]] = None,
        initial: str = MODE_OVERLAY,
    ) -> None:
        super().__init__(parent, padding=(4, 4))
        self._on_change = on_change or (lambda m: None)
        self._current = initial
        self._buttons: Dict[str, ttk.Button] = {}
        self._enabled: Dict[str, bool] = {m: True for m in ALL_MODES}

        ttk.Label(self, text="View:",
                  foreground="#707070",
                  font=("TkDefaultFont", 9, "bold")).pack(
            side="left", padx=(0, 6))

        for m in ALL_MODES:
            btn = ttk.Button(
                self, text=_LABELS[m], width=22 if m == MODE_PROCESSED else 12,
                command=lambda mm=m: self._handle_click(mm))
            btn.pack(side="left", padx=2)
            self._buttons[m] = btn
            Tooltip(btn, _TOOLTIPS[m])

        self._refresh_pressed_state()

    def _handle_click(self, mode: str) -> None:
        if not self._enabled.get(mode, True):
            return
        if self._current == mode:
            return
        self._current = mode
        self._refresh_pressed_state()
        self._on_change(mode)

    def _refresh_pressed_state(self) -> None:
        for m, btn in self._buttons.items():
            if not self._enabled.get(m, True):
                btn.state(["disabled"])
            elif m == self._current:
                btn.state(["pressed", "!disabled"])
            else:
                btn.state(["!pressed", "!disabled"])

    # ── Public API ────────────────────────────────────────────────────
    def set_mode(self, mode: str, fire_callback: bool = False) -> None:
        if mode not in ALL_MODES:
            return
        self._current = mode
        self._refresh_pressed_state()
        if fire_callback:
            self._on_change(mode)

    def get_mode(self) -> str:
        return self._current

    def set_enabled(self, mode: str, enabled: bool) -> None:
        self._enabled[mode] = enabled
        self._refresh_pressed_state()
