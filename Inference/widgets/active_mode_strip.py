"""ActiveModeStrip — small strip showing the currently-locked-in flatten mode.

Sits between the result panel and the action bar. One line of text:

  "Active mode: Tight droplet mode (cfg4 / boundary_normalise)"

Updates the moment the override dropdown changes or the auto-decide
output changes. Provides redundant clarity (the same info is in the
preview row's rationale + dropdown) so users don't have to look in
multiple places to know "what's going to the model".

Public API:
  - ActiveModeStrip(parent)
  - .set_mode(mode_str: str, source: str)
        mode_str   — concrete mode used (calibration / boundary_normalise / ...)
        source     — "auto" or "user override"
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


_MODE_DESCRIPTIVE = {
    "calibration":        "Calibration sphere mode",
    "boundary_normalise": "Tight droplet mode",
    "simple":             "Simple flatten",
    "none":               "No flatten",
    "auto":               "Auto-detect",
}

_MODE_TECHNICAL = {
    "calibration":        "cfg1 / calibration",
    "boundary_normalise": "cfg4 / boundary_normalise",
    "simple":             "cfg3 / simple",
    "none":               "no flatten",
    "auto":               "auto-detect",
}


class ActiveModeStrip(ttk.Frame):
    """One-line indicator: which preprocessing mode the next Process will use."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, padding=(8, 2))
        self.var_text = tk.StringVar(value="Active mode: (no model loaded)")
        self.lbl = tk.Label(
            self, textvariable=self.var_text,
            background="#eef3ee", foreground="#1a5d1a",
            font=("TkDefaultFont", 9),
            padx=8, pady=2, anchor="w")
        self.lbl.pack(fill="x")

    # ── Public API ────────────────────────────────────────────────────
    def set_mode(self, mode_str: str, source: str = "auto") -> None:
        """source is 'auto' or 'user override'."""
        descriptive = _MODE_DESCRIPTIVE.get(mode_str, mode_str)
        technical = _MODE_TECHNICAL.get(mode_str, mode_str)
        if source == "user override":
            text = (f"Active mode: {descriptive}  ({technical})  "
                     f"— locked by user override")
            bg, fg = "#fff4e0", "#a04000"
        else:
            text = (f"Active mode: {descriptive}  ({technical})  "
                     f"— picked by auto-detect")
            bg, fg = "#eef3ee", "#1a5d1a"
        self.var_text.set(text)
        self.lbl.configure(background=bg, foreground=fg)

    def clear(self) -> None:
        self.var_text.set("Active mode: (no model loaded)")
        self.lbl.configure(background="#f0f0f0", foreground="#707070")


# ── Helpers usable by ActionBar to format Process button label ───────
def process_button_label(mode_str: str) -> str:
    """Return e.g. 'Process current (Tight droplet mode)'."""
    descriptive = _MODE_DESCRIPTIVE.get(mode_str, mode_str)
    return f"Process current  ({descriptive})"
