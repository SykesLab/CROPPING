# plotting_modular.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cv2


def save_darkness_plot(out_path, curve, first, last, best, name):
    """
    Save darkness curve plot with best frame marked.
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


def save_geometric_overlay(out_path, geo, best_idx):
    """
    Overlay geometry lines and arrows on raw grayscale frame and save.
    """
    raw = geo["frame"]
    mask = geo["mask"]

    y_top = geo["y_top"]
    y_bottom = geo["y_bottom"]
    y_sphere = geo["y_bottom_sphere"]

    H, W = raw.shape

    rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    rgb[mask] = (rgb[mask] * 0.6 + np.array([255, 0, 0]) * 0.4).astype(np.uint8)

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb, cmap="gray")
    ax = plt.gca()

    # Lines
    ax.axhline(0, color="blue", linewidth=1.5, label="Top of image")
    if y_top is not None:
        ax.axhline(y_top, color="green", linewidth=1.5, label="Top of droplet")
    if y_bottom is not None:
        ax.axhline(y_bottom, color="orange", linewidth=1.5, label="Bottom of droplet")
    if y_sphere is not None:
        ax.axhline(y_sphere, color="red", linewidth=1.5, label="Top of bottom sphere")

    # Arrow X-position for vertical measurements
    x_arrow = int(W * 0.92)

    # === Top margin: 0 → y_top ===
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

    # === Bottom gap: y_bottom → y_sphere ===
    if (
        y_bottom is not None
        and y_sphere is not None
        and y_sphere > y_bottom
    ):
        gap = float(y_sphere - y_bottom)
        ax.annotate(
            "",
            xy=(x_arrow, y_sphere),
            xytext=(x_arrow, y_bottom),
            arrowprops=dict(arrowstyle="<->", color="cyan", lw=1.2),
        )
        ax.text(
            x_arrow + 5,
            (y_sphere + y_bottom) / 2.0,
            f"{gap:.1f}",
            color="cyan",
            fontsize=6,
            va="center",
        )

    # === NEW: Droplet diameter arrow: y_top → y_bottom ===
    if (
        y_top is not None
        and y_bottom is not None
        and y_bottom > y_top
    ):
        diameter = float(y_bottom - y_top)
        x_diam = int(W * 0.06)  # left side vertical measurement position

        ax.annotate(
            "",
            xy=(x_diam, y_bottom),
            xytext=(x_diam, y_top),
            arrowprops=dict(arrowstyle="<->", color="magenta", lw=1.2),
        )
        ax.text(
            x_diam + 5,
            (y_top + y_bottom) / 2.0,
            f"{diameter:.1f}",
            color="magenta",
            fontsize=6,
            va="center",
        )

    # Legend
    legend_elements = [
        Line2D([0], [0], color="blue",   lw=1.5, label="Top of image"),
        Line2D([0], [0], color="green",  lw=1.5, label="Top of droplet"),
        Line2D([0], [0], color="orange", lw=1.5, label="Bottom of droplet"),
        Line2D([0], [0], color="red",    lw=1.5, label="Top of bottom sphere"),
        Line2D([0], [0], color="yellow", lw=1.5, label="Top margin"),
        Line2D([0], [0], color="cyan",   lw=1.5, label="Bottom gap"),
        Line2D([0], [0], color="magenta",lw=1.5, label="Droplet diameter"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=6)

    ax.set_title(f"Best frame {best_idx}")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

