# cine_iterator_geometric_4mm_every10.py
#
# Selects best frames using hybrid geometric+darkness logic,
# draws overlay with arrows, then autonomous droplet-centered crop.

from pathlib import Path
import csv
import re

import numpy as np
import matplotlib.pyplot as plt
import cv2

from pyphantom import Phantom, cine

from frame_selector_geometric_4mm import (
    analyze_cine_geometric as analyze_cine,
    choose_best_frame_geometric,
)

from frame_cropping_geometric import (
    analyze_frame_geometric,
    crop_autocenter_simple,
)

ph = Phantom()

OUTPUT_ROOT = Path(
    r"C:\Users\justi\Downloads\coursework\testing_4mm\centering\OUTPUT"
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

CNN_W = 320
CNN_H = 240


# -------------------------------------------------------
# Group by droplet ID
# -------------------------------------------------------
def group_cines_by_droplet(folder):
    folder = Path(folder)
    groups = {}

    for path in sorted(folder.rglob("*.cine")):
        m = re.search(r"(\d+)([gv])$", path.stem, re.IGNORECASE)
        if not m:
            continue
        droplet_id = m.group(1)
        cam = m.group(2).lower()

        groups.setdefault(droplet_id, {"g": None, "v": None})
        groups[droplet_id][cam] = path

    sorted_ids = sorted(groups.keys(), key=lambda x: int(x))
    return [(did, groups[did]) for did in sorted_ids]


# -------------------------------------------------------
# Save darkness curve
# -------------------------------------------------------
def save_darkness_plot(out_path, curve, first, last, best_frame, cine_name):
    x = np.arange(first, last + 1)
    plt.figure(figsize=(10, 4))
    plt.plot(x, curve)
    plt.axvline(best_frame, color="r", linestyle="--")
    plt.title(f"Darkness Curve\n{cine_name}")
    plt.xlabel("Frame")
    plt.ylabel("Dark fraction")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -------------------------------------------------------
# Save geometric overlay (WITH ARROWS)
# -------------------------------------------------------
def save_geometric_overlay(out_path, geo_info, best_frame_idx):
    raw = geo_info["frame"]
    mask = geo_info["mask"]
    y_top = geo_info["y_top"]
    y_bottom = geo_info["y_bottom"]
    y_sphere = geo_info["y_bottom_sphere"]

    H, W = raw.shape

    norm = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
    rgb[mask] = (rgb[mask] * 0.7 + np.array([255, 0, 0]) * 0.3).astype(np.uint8)

    plt.figure(figsize=(7, 6))
    plt.imshow(rgb)
    ax = plt.gca()

    ax.axhline(0, color="blue", linewidth=2)
    if y_top is not None:
        ax.axhline(y_top, color="green", linewidth=2)
    if y_bottom is not None:
        ax.axhline(y_bottom, color="orange", linewidth=2)
    if y_sphere is not None:
        ax.axhline(y_sphere, color="red", linewidth=2)

    # Draw arrows if geometry exists
    if y_top is not None and y_bottom is not None and y_sphere is not None:
        x_arrow = W * 0.92

        # 1. Top margin: image top -> droplet top
        top_margin = y_top - 0
        ax.annotate(
            "", xy=(x_arrow, y_top), xytext=(x_arrow, 0),
            arrowprops=dict(arrowstyle="<->", color="yellow", lw=2)
        )
        ax.text(x_arrow + 5, (y_top + 0) / 2,
                f"{top_margin:.1f}px", color="yellow", va="center")

        # 2. Bottom gap: droplet bottom -> sphere top
        bottom_gap = y_sphere - y_bottom
        ax.annotate(
            "", xy=(x_arrow, y_sphere), xytext=(x_arrow, y_bottom),
            arrowprops=dict(arrowstyle="<->", color="cyan", lw=2)
        )
        ax.text(x_arrow + 5, (y_sphere + y_bottom) / 2,
                f"{bottom_gap:.1f}px", color="cyan", va="center")

    ax.set_title(f"Best frame {best_frame_idx}")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -------------------------------------------------------
# Main pipeline
# -------------------------------------------------------
def process_every_10(folder, csv_output="cine_summary_4mm_geometric.csv"):
    folder = Path(folder)

    csv_path = OUTPUT_ROOT / csv_output
    if csv_path.exists():
        try:
            csv_path.unlink()
        except:
            pass

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "droplet_id", "camera", "cine_file",
            "first_frame", "last_frame",
            "best_frame", "dark_fraction",
            "y_top", "y_bottom", "y_sphere",
            "crop_path"
        ])

        groups = group_cines_by_droplet(folder)

        for idx, (droplet_id, cams) in enumerate(groups):

            if idx % 10 != 0:
                continue

            print(f"\n=== Droplet {droplet_id} (index {idx}) ===")

            for cam in ("g", "v"):
                path = cams.get(cam)
                if path is None:
                    print(f"  [SKIP] missing {cam}")
                    continue

                print(f"  Processing {cam}-camera: {path}")

                c = cine.Cine.from_filepath(str(path))
                if c is None:
                    print("    [ERROR] cannot read cine")
                    continue

                cine_name = path.name
                prefix = path.stem

                dark_info = analyze_cine(c)
                curve = dark_info["darkness_curve"]
                first = dark_info["first_frame"]
                last = dark_info["last_frame"]

                best_frame_idx = choose_best_frame_geometric(c, curve, first)
                pos = best_frame_idx - first
                best_dark_fraction = float(curve[pos])

                geo = analyze_frame_geometric(c, best_frame_idx)
                y_top = geo["y_top"]
                y_bottom = geo["y_bottom"]
                y_sphere = geo["y_bottom_sphere"]
                cx = geo["cx"]

                crop_path = ""
                if y_top is not None and y_bottom is not None:
                    crop = crop_autocenter_simple(
                        geo["frame"], y_top, y_bottom, cx,
                        target_w=CNN_W, target_h=CNN_H
                    )
                    out_name = f"{prefix}_crop.png"
                    out_path = OUTPUT_ROOT / out_name
                    cv2.imwrite(str(out_path), crop)
                    crop_path = str(out_path)
                    print(f"    Saved crop: {out_name}")

                writer.writerow([
                    droplet_id, cam, cine_name,
                    first, last,
                    best_frame_idx, best_dark_fraction,
                    y_top, y_bottom, y_sphere,
                    crop_path
                ])

                save_darkness_plot(
                    OUTPUT_ROOT / f"{prefix}_darkness.png",
                    curve, first, last, best_frame_idx, cine_name
                )
                save_geometric_overlay(
                    OUTPUT_ROOT / f"{prefix}_overlay.png",
                    geo, best_frame_idx
                )

    print(f"\nCSV saved to: {csv_path}")
