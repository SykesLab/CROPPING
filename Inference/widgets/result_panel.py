"""ResultPanel — V1's big prominent defocus number, plus detail row.

Lives between the tabs and the action bar. Always visible. Updates the
moment a Process completes; cleared on cine open / model load.

Layout:
  ┌─ Result ──────────────────────────────────────────────────────┐
  │                                                                │
  │           2.244 ± 0.015 mm        [IN_RANGE]                  │
  │                                                                │
  │      pred_norm = 0.41   sigma_model = 2.45 px   sigma_native = 2.86 px
  └────────────────────────────────────────────────────────────────┘

Public API:
  - ResultPanel(parent)
  - .update(results: dict | None)
  - .clear()
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, Optional


_FLAG_FG = {
    "IN_RANGE": "#1a5d1a",
    "BELOW_FLOOR": "#a04000",
    "SATURATED": "#a01515",
}
_FLAG_BG = {
    "IN_RANGE": "#e8f4e8",
    "BELOW_FLOOR": "#fff4e0",
    "SATURATED": "#fde0e0",
}

_BIG_FONT = ("Segoe UI", 22, "bold")
_DETAIL_FONT = ("Consolas", 10)
_FLAG_FONT = ("Segoe UI", 11, "bold")


class ResultPanel(ttk.LabelFrame):
    """Persistent result display."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, text="Result", padding=(10, 4))

        self.var_main = tk.StringVar(value="(no result yet — click Process)")
        self.var_flag = tk.StringVar(value="")
        self.var_details = tk.StringVar(value="")

        # Main row: big number + flag badge
        main_row = ttk.Frame(self)
        main_row.pack(fill="x")
        self.lbl_main = ttk.Label(
            main_row, textvariable=self.var_main,
            font=_BIG_FONT, foreground="#1a73e8")
        self.lbl_main.pack(side="left")
        self.lbl_flag = tk.Label(
            main_row, textvariable=self.var_flag,
            font=_FLAG_FONT, padx=8, pady=2,
            background="#f0f0f0", foreground="#404040")
        self.lbl_flag.pack(side="left", padx=(12, 0))

        # Detail row
        ttk.Label(self, textvariable=self.var_details,
                  font=_DETAIL_FONT, foreground="#404040").pack(
            anchor="w", pady=(2, 0))

    # ── Public API ────────────────────────────────────────────────────
    def update(self, results: Optional[Dict[str, Any]]) -> None:
        if not results:
            self.clear()
            return
        z = results.get("defocus_mm", 0.0)
        unc = results.get("defocus_uncertainty_mm", 0.0) or 0.0
        flag = str(results.get("bounds_flag", "IN_RANGE"))
        unc_str = f" ± {unc:.3f}" if unc > 0 else ""
        self.var_main.set(f"{z:.3f}{unc_str} mm")

        if flag and flag != "IN_RANGE":
            self.var_flag.set(f"  {flag}  ")
            self.lbl_flag.configure(
                background=_FLAG_BG.get(flag, "#f0f0f0"),
                foreground=_FLAG_FG.get(flag, "#404040"),
            )
        else:
            self.var_flag.set(f"  {flag}  ")
            self.lbl_flag.configure(
                background=_FLAG_BG.get("IN_RANGE", "#e8f4e8"),
                foreground=_FLAG_FG.get("IN_RANGE", "#1a5d1a"),
            )

        sigma_native = results.get("sigma_native", 0.0)
        sigma_model = results.get("sigma_model", 0.0)
        pred_norm = results.get("pred_norm", 0.0)
        self.var_details.set(
            f"pred_norm = {pred_norm:.4f}   "
            f"sigma_model = {sigma_model:.3f} px   "
            f"sigma_native = {sigma_native:.3f} px"
        )

    def clear(self) -> None:
        self.var_main.set("(no result yet — click Process)")
        self.var_flag.set("")
        self.lbl_flag.configure(background="#f0f0f0", foreground="#404040")
        self.var_details.set("")
