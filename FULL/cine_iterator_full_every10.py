# cine_iterator_full_every10.py
#
# Iterates through ALL subfolders in F:\spheres\, groups cines by droplet,
# processes ONLY every 10th droplet per subfolder,
# applies FULL geometric best-frame selection,
# and saves outputs into:
#   FULL/OUTPUT/<subfolder_name>/

from pathlib import Path
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pyphantom import cine
from pyphantom import Phantom
ph = Phantom()


from frame_selector_full import (
    analyze_cine_full as analyze_cine,
    choose_best_frame_full as choose_best_frame,
)

from frame_cropping_full import (
    analyze_frame_geometric,
    crop_autocenter_simple,
)

# -------------------------------------------------------
# Output base directory
# -------------------------------------------------------
OUTPUT_ROOT = Path(
    r"C:\Users\justi\Downloads\coursework\testing_4mm\FULL\OUTPUT"
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


CNN_W = 320
CNN_H = 240

def safe_load_cine(path):
    try:
        c = cine.Cine.from_filepath(str(path))
        if c is None:
            return None
        # access one attribute to ensure handle is valid 
        _ = c.range
        return c
    except Exception as e:
        print(f"    [SDK ERROR loading {path.name}]: {e}")
        return None



# -------------------------------------------------------
# Group cine files by droplet ID inside a folder
# -------------------------------------------------------
def group_cines_by_droplet(folder):
    folder = Path(folder)
    groups = {}

    for path in sorted(folder.glob("*.cine")):
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
def save_darkness_plot(out_path, curve, first, last, best_frame, name):
    x = np.arange(first, last + 1)
    plt.figure(figsize=(10, 4))
    plt.plot(x, curve)
    plt.axvline(best_frame, color="r", linestyle="--")
    plt.title(f"Darkness Curve\n{name}")
    plt.xlabel("Frame")
    plt.ylabel("Dark fraction")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -------------------------------------------------------
# Save geometric overlay
# -------------------------------------------------------
def save_geometric_overlay(out_path, geo, best_frame_idx):
    raw = geo["frame"]
    mask = geo["mask"]
    y_top = geo["y_top"]
    y_bottom = geo["y_bottom"]
    y_sphere = geo["y_bottom_sphere"]

    H, W = raw.shape

    rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    rgb[mask] = (rgb[mask] * 0.7 + np.array([255, 0, 0]) * 0.3).astype(np.uint8)

    plt.figure(figsize=(8, 7))
    plt.imshow(rgb, cmap="gray")
    ax = plt.gca()

    # Lines
    ax.axhline(0, color="blue", linewidth=2, label="Top of image")
    if y_top is not None:
        ax.axhline(y_top, color="green", linewidth=2, label="Top droplet")
    if y_bottom is not None:
        ax.axhline(y_bottom, color="orange", linewidth=2, label="Bottom droplet")
    if y_sphere is not None:
        ax.axhline(y_sphere, color="red", linewidth=2, label="Top sphere")

    # Arrows if possible
    if (
        y_top is not None
        and y_bottom is not None
        and y_sphere is not None
    ):
        x_arrow = int(W * 0.92)

        # Top margin
        top_margin = y_top
        ax.annotate(
            "", xy=(x_arrow, y_top), xytext=(x_arrow, 0),
            arrowprops=dict(arrowstyle="<->", color="yellow", lw=2)
        )
        ax.text(x_arrow + 5, y_top / 2,
                f"{top_margin:.1f}px", color="yellow", va="center")

        # Bottom gap
        bottom_gap = y_sphere - y_bottom
        ax.annotate(
            "", xy=(x_arrow, y_sphere), xytext=(x_arrow, y_bottom),
            arrowprops=dict(arrowstyle="<->", color="cyan", lw=2)
        )
        ax.text(x_arrow + 5, (y_sphere + y_bottom)/2,
                f"{bottom_gap:.1f}px", color="cyan", va="center")

    ax.set_title(f"Best frame {best_frame_idx}")
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



# -------------------------------------------------------
# Main iterator: process every 10th droplet per subfolder
# -------------------------------------------------------
def process_every_10_all_subfolders(root_folder):
    root = Path(root_folder)

    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue

        print(f"\n=============== SUBFOLDER: {sub.name} ===============")

        # Create output subfolder
        out_sub = OUTPUT_ROOT / sub.name
        out_sub.mkdir(parents=True, exist_ok=True)

        csv_path = out_sub / f"{sub.name}_summary_full.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "droplet_id", "camera", "cine_file",
                "first_frame", "last_frame",
                "best_frame", "dark_fraction",
                "y_top", "y_bottom", "y_sphere",
                "crop_path"
            ])

            groups = group_cines_by_droplet(sub)

            for idx, (droplet_id, cams) in enumerate(groups):

                # process only every 10th per subfolder
                if idx % 10 != 0:
                    continue

                print(f"\n--- droplet {droplet_id} (index {idx}) ---")

                for cam in ("g", "v"):
                    path = cams.get(cam)
                    if path is None:
                        print(f"  [SKIP] no {cam}-camera file")
                        continue

                    print(f"  Processing {cam}-camera: {path}")
                    c = safe_load_cine(path)
                    if c is None:
                        print(f"    [SKIP] could not open cine: {path.name}")
                        continue


                    # darkness curve
                    dark_info = analyze_cine(c)
                    curve = dark_info["darkness_curve"]
                    first = dark_info["first_frame"]
                    last = dark_info["last_frame"]

                    # best frame (pre-collision + equidistance)
                    best_frame = choose_best_frame(c, curve, first)
                    dark_val = float(curve[best_frame - first])

                    # geometry for best frame
                    geo = analyze_frame_geometric(c, best_frame)
                    y_top = geo["y_top"]
                    y_bottom = geo["y_bottom"]
                    y_sphere = geo["y_bottom_sphere"]
                    cx = geo["cx"]

                    # crop
                    crop_path = ""
                    if y_top is not None and y_bottom is not None:
                        crop = crop_autocenter_simple(
                            geo["frame"], y_top, y_bottom, cx,
                            target_w=CNN_W, target_h=CNN_H
                        )
                        out_crop = out_sub / f"{path.stem}_crop.png"
                        cv2.imwrite(str(out_crop), crop)
                        crop_path = str(out_crop)
                        print(f"    Saved crop: {out_crop.name}")

                    # darkness plot
                    save_darkness_plot(
                        out_sub / f"{path.stem}_darkness.png",
                        curve, first, last, best_frame, path.name
                    )
                    # overlay
                    save_geometric_overlay(
                        out_sub / f"{path.stem}_overlay.png",
                        geo, best_frame
                    )

                    # CSV
                    writer.writerow([
                        droplet_id, cam, path.name,
                        first, last,
                        best_frame, dark_val,
                        y_top, y_bottom, y_sphere,
                        crop_path
                    ])

        print(f"CSV saved: {csv_path}")
