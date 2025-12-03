# cine_iterator_test_4mm_every10.py
#
# Iterate through F:\spheres\4mm-borosilicate, group g/v pairs by droplet ID,
# process every 10th droplet (both cameras), and save:
#   - darkness curve plot (with best frame marked)
#   - Otsu-overlay image of best frame
#   - Cropped frame above bottom sphere
#   - CSV summary

from pathlib import Path
import csv
import re
import os

import matplotlib.pyplot as plt
import numpy as np

from pyphantom import Phantom, cine
from frame_selector_test_4mm_viz import analyze_cine
from frame_cropping import crop_below_dark_sphere


# Keep SDK initialised
ph = Phantom()

# Where to put visual outputs
OUTPUT_ROOT = Path(
    r"C:\Users\justi\Downloads\coursework\testing_4mm\OUTPUT"
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def group_cines_by_droplet(folder):
    folder = Path(folder)
    groups = {}

    for path in sorted(folder.rglob("*.cine")):
        stem = path.stem  # e.g. 'sphere0843g'
        m = re.search(r"(\d+)([gv])$", stem, re.IGNORECASE)
        if not m:
            continue

        droplet_id = m.group(1)
        cam = m.group(2).lower()

        if droplet_id not in groups:
            groups[droplet_id] = {"g": None, "v": None}

        groups[droplet_id][cam] = path

    sorted_ids = sorted(groups.keys(), key=lambda x: int(x))
    return [(did, groups[did]) for did in sorted_ids]


def save_darkness_plot(out_path, info, cine_name):
    first = info["first_frame"]
    last = info["last_frame"]
    curve = info["darkness_curve"]
    best = info["best_frame"]

    x = np.arange(first, last + 1, dtype=int)

    plt.figure(figsize=(10, 4))
    plt.plot(x, curve, label="Darkness fraction")
    plt.axvline(best, color="r", linestyle="--", label=f"Best frame {best}")
    plt.xlabel("Frame index")
    plt.ylabel("Dark fraction")
    plt.title(f"Darkness Curve\n{cine_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_best_frame_overlay(out_path, info):
    frame = info["best_frame_image"]
    mask = info["best_mask_dark"]

    norm = cv2_normalise_to_8bit(frame)

    import cv2
    rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)

    red = np.zeros_like(rgb)
    red[..., 0] = 255

    alpha = 0.3
    rgb[mask] = (1 - alpha) * rgb[mask] + alpha * red[mask]

    plt.figure(figsize=(5, 5))
    plt.imshow(rgb)
    plt.axis("off")
    plt.title(f"Best frame {info['best_frame']}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def cv2_normalise_to_8bit(arr):
    import cv2
    arr = arr.astype(np.float32)
    norm = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def process_every_10(folder, csv_output="cine_summary_4mm_every10.csv"):
    folder = Path(folder)
    results = {}

    csv_path = OUTPUT_ROOT / csv_output
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "folder_label",
            "droplet_id",
            "camera",
            "cine_file",
            "first_frame",
            "last_frame",
            "total_frames",
            "best_frame",
            "best_dark_fraction",
        ])

        groups = group_cines_by_droplet(folder)

        for idx, (droplet_id, cams) in enumerate(groups):
            if idx % 10 != 0:
                continue

            print(f"\n=== Droplet {droplet_id} (index {idx}) ===")

            for cam in ("g", "v"):
                path = cams.get(cam)
                if path is None:
                    print(f"  [SKIP] No {cam}-camera file for droplet {droplet_id}")
                    continue

                print(f"  Processing {cam}-camera: {path}")

                c = cine.Cine.from_filepath(str(path))
                if c is None:
                    print(f"    [ERROR] Could not load cine: {path}")
                    continue

                info = analyze_cine(c)
                results[str(path)] = info

                folder_label = path.parent.name
                cine_name = path.name
                cine_prefix = path.stem

                writer.writerow([
                    folder_label,
                    droplet_id,
                    cam,
                    cine_name,
                    info["first_frame"],
                    info["last_frame"],
                    info["total_frames"],
                    info["best_frame"],
                    f"{info['best_dark_fraction']:.4f}",
                ])

                # -----------------------------
                # FLAT OUTPUT
                # -----------------------------
                darkness_png = OUTPUT_ROOT / f"{cine_prefix}_darkness.png"
                overlay_png  = OUTPUT_ROOT / f"{cine_prefix}_overlay.png"

                save_darkness_plot(darkness_png, info, cine_name)
                save_best_frame_overlay(overlay_png, info)

                # -----------------------------
                # CROPPING STEP
                # -----------------------------
                crop_result = crop_below_dark_sphere(info["best_frame_image"])

                cropped_png = OUTPUT_ROOT / f"{cine_prefix}_cropped.png"
                

                plt.imsave(cropped_png, crop_result["cropped"], cmap="gray")
                

                print(f"    Saved: {darkness_png.name}, {overlay_png.name}, "
                      f"{cropped_png.name}")

    print(f"\nCSV saved to: {csv_path}")
    return results
