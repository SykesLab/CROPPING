"""FrameCanvas — zoomable / pannable image display.

Lifts V1's zoom/pan implementation into a self-contained widget. The
caller passes images via ``set_image()``; the widget handles all
rendering, mouse-wheel zoom, click-drag pan, and double-click reset.

Public API:
  - FrameCanvas(parent, bg='#2b2b2b')
  - .set_image(np.ndarray, reset_view=False) — display this image
  - .clear() — blank the canvas
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk


# Zoom limits — match V1
_ZOOM_MIN = 1.0
_ZOOM_MAX = 20.0
_ZOOM_STEP = 1.15


class FrameCanvas(ttk.Frame):
    """Self-contained zoom/pan image viewer."""

    def __init__(self, parent: tk.Misc, bg: str = "#2b2b2b") -> None:
        super().__init__(parent)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0,
                                  borderwidth=0, relief="flat")
        self.canvas.pack(fill="both", expand=True)

        # State
        self._current_image: Optional[np.ndarray] = None
        self._zoom_level: float = 1.0
        self._pan_offset = [0, 0]
        self._drag_start: Optional[tuple[int, int]] = None
        self._photo_refs: list = []
        self._canvas_img_id: Optional[int] = None

        # Bindings
        self.canvas.bind("<MouseWheel>", self._on_scroll_zoom)        # Windows
        self.canvas.bind("<Button-4>", self._on_scroll_zoom)          # Linux up
        self.canvas.bind("<Button-5>", self._on_scroll_zoom)          # Linux down
        self.canvas.bind("<ButtonPress-1>", self._on_pan_start)
        self.canvas.bind("<B1-Motion>", self._on_pan_drag)
        self.canvas.bind("<Double-Button-1>", self._on_zoom_reset)
        self.canvas.bind("<Configure>", lambda _e: self._render())

    # ── Public API ─────────────────────────────────────────────────────
    def set_image(self, img: np.ndarray, reset_view: bool = False) -> None:
        """Display ``img`` (numpy array, grayscale or BGR). When
        ``reset_view`` is True, zoom/pan are reset to fit-to-canvas."""
        self._current_image = img
        if reset_view:
            self._zoom_level = 1.0
            self._pan_offset = [0, 0]
        self._render()

    def clear(self) -> None:
        """Blank the canvas."""
        self._current_image = None
        self.canvas.delete("all")
        self._photo_refs.clear()

    # ── Render ─────────────────────────────────────────────────────────
    def _render(self) -> None:
        img = self._current_image
        if img is None:
            return

        self._photo_refs.clear()
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            # Canvas not yet mapped — schedule a retry once it is
            self.after(50, self._render)
            return

        # Convert to uint8 display
        if img.dtype in (np.float32, np.float64):
            disp = np.clip(img * 255, 0, 255).astype(np.uint8)
        else:
            disp = img.astype(np.uint8)

        img_h, img_w = disp.shape[:2]

        # Base scale: fit image inside canvas; never enlarge above 1× by
        # default (zoom is the user's job)
        base_scale = min(canvas_w / img_w, canvas_h / img_h, 1.0)
        scale = base_scale * self._zoom_level

        new_w = max(int(img_w * scale), 1)
        new_h = max(int(img_h * scale), 1)

        resized = cv2.resize(
            disp, (new_w, new_h),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST,
        )

        if resized.ndim == 2:
            pil_img = Image.fromarray(resized, mode="L")
        else:
            pil_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(pil_img)
        self._photo_refs.append(photo)

        ox, oy = self._pan_offset
        cx = canvas_w // 2 + int(ox)
        cy = canvas_h // 2 + int(oy)

        self.canvas.delete("all")
        self._canvas_img_id = self.canvas.create_image(
            cx, cy, image=photo, anchor="center")

    # ── Zoom / pan handlers ────────────────────────────────────────────
    def _on_scroll_zoom(self, event) -> None:
        if self._current_image is None:
            return
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            factor = _ZOOM_STEP
        elif event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            factor = 1 / _ZOOM_STEP
        else:
            return
        old_zoom = self._zoom_level
        self._zoom_level = max(_ZOOM_MIN, min(_ZOOM_MAX,
                                                self._zoom_level * factor))
        # Zoom toward mouse position
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        mx = event.x - canvas_w // 2 - self._pan_offset[0]
        my = event.y - canvas_h // 2 - self._pan_offset[1]
        zoom_ratio = self._zoom_level / old_zoom
        self._pan_offset[0] -= mx * (zoom_ratio - 1)
        self._pan_offset[1] -= my * (zoom_ratio - 1)
        self._render()

    def _on_pan_start(self, event) -> None:
        self._drag_start = (event.x, event.y)

    def _on_pan_drag(self, event) -> None:
        if self._drag_start is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._pan_offset[0] += dx
        self._pan_offset[1] += dy
        self._drag_start = (event.x, event.y)
        self._render()

    def _on_zoom_reset(self, _event=None) -> None:
        """Double-click resets zoom + pan to fit-to-canvas."""
        self._zoom_level = 1.0
        self._pan_offset = [0, 0]
        self._render()
