"""Visualisation utilities for darkness curves and geometry overlays."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def save_darkness_plot(
    out_path: Union[str, Path],
    curve: np.ndarray,
    first: int,
    last: int,
    best: int,
    name: str,
) -> None:
    """Save darkness curve plot with best frame marker.

    Args:
        out_path: Output file path.
        curve: Darkness curve array.
        first: First frame index.
        last: Last frame index.
        best: Best frame index.
        name: Title suffix (typically filename).
    """
    x = np.arange(first, last + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(x, curve, label="Dark fraction")
    plt.axvline(best, color="r", linestyle="--", label=f"Best = {best}")
    plt.title(f"Darkness Curve — {name}")
    plt.xlabel("Frame")
    plt.ylabel("Dark fraction")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_geometric_overlay(
    out_path: Union[str, Path],
    geo: Dict[str, Any],
    best_idx: int,
    cnn_size: Optional[int] = None,
    safety: int = 3,
) -> None:
    """Save frame with geometry annotations overlaid.

    Args:
        out_path: Output file path.
        geo: Geometry dict containing frame, mask, and coordinates.
        best_idx: Frame index for title.
        cnn_size: Crop size to display (optional).
        safety: Safety margin for crop rectangle.
    """
    raw = geo["frame"]
    mask = geo["mask"]
    y_top = geo["y_top"]
    y_bottom = geo["y_bottom"]
    y_sphere = geo["y_bottom_sphere"]

    height, width = raw.shape

    # Create RGB overlay
    rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    rgb[mask] = (rgb[mask] * 0.6 + np.array([255, 0, 0]) * 0.4).astype(np.uint8)

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb, cmap="gray")
    ax = plt.gca()

    # Draw horizontal lines
    ax.axhline(0, color="blue", linewidth=1.5, label="Top of image")
    if y_top is not None:
        ax.axhline(y_top, color="green", linewidth=1.5, label="Top of droplet")
    if y_bottom is not None:
        ax.axhline(y_bottom, color="orange", linewidth=1.5, label="Bottom of droplet")
    if y_sphere is not None:
        ax.axhline(y_sphere, color="red", linewidth=1.5, label="Top of sphere")

    # Arrow position
    x_arrow = int(width * 0.92)

    # Top margin arrow
    if y_top is not None and y_top > 0:
        ax.annotate(
            "",
            xy=(x_arrow, y_top),
            xytext=(x_arrow, 0),
            arrowprops=dict(arrowstyle="<->", color="yellow", lw=1.2),
        )
        ax.text(
            x_arrow + 5,
            y_top / 2.0,
            f"{float(y_top):.1f}",
            color="yellow",
            fontsize=6,
            va="center",
        )

    # Bottom gap arrow
    if y_bottom is not None and y_sphere is not None and y_sphere > y_bottom:
        gap = float(y_sphere - y_bottom)
        ax.annotate(
            "",
            xy=(x_arrow, y_sphere),
            xytext=(x_arrow, y_bottom),
            arrowprops=dict(arrowstyle="<->", color="yellow", lw=1.2),
        )
        ax.text(
            x_arrow + 5,
            (y_sphere + y_bottom) / 2.0,
            f"{gap:.1f}",
            color="yellow",
            fontsize=6,
            va="center",
        )

    # Droplet diameter arrow
    if y_top is not None and y_bottom is not None and y_bottom > y_top:
        diameter = float(y_bottom - y_top)
        x_diam = int(width * 0.06)
        ax.annotate(
            "",
            xy=(x_diam, y_bottom),
            xytext=(x_diam, y_top),
            arrowprops=dict(arrowstyle="<->", color="cyan", lw=1.2),
        )
        ax.text(
            x_diam + 5,
            (y_top + y_bottom) / 2.0,
            f"{diameter:.1f}",
            color="cyan",
            fontsize=6,
            va="center",
        )

    # Crop rectangle
    if cnn_size is not None and y_top is not None and y_bottom is not None:
        cy = 0.5 * (y_top + y_bottom)
        half = cnn_size / 2

        y0 = max(0, int(cy - half))
        x0 = max(0, int(geo["cx"] - half))
        y1 = min(height, y0 + cnn_size)
        x1 = min(width, x0 + cnn_size)

        rect = plt.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=1.0,
            edgecolor="magenta",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)

        ax.text(
            x0 + 3,
            y0 + 10,
            f"{x1 - x0} × {y1 - y0}",
            color="magenta",
            fontsize=6,
            va="top",
        )

    # Legend
    legend_elements = [
        Line2D([0], [0], color="blue", lw=1.5, label="Top of image"),
        Line2D([0], [0], color="green", lw=1.5, label="Top of droplet"),
        Line2D([0], [0], color="orange", lw=1.5, label="Bottom of droplet"),
        Line2D([0], [0], color="red", lw=1.5, label="Top of sphere"),
        Line2D([0], [0], color="cyan", lw=1.5, label="Droplet diameter"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=6)

    ax.set_title(f"Best frame {best_idx}")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
