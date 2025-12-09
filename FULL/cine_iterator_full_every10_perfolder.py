# cine_iterator_full_every10_perfolder.py
#
# PER-FOLDER mode:
#   - For each subfolder, only uses droplets that will be processed (every 10th).
#   - For those cines, computes best frame + geometry ONCE.
#   - Uses that geometry to calibrate a crop size PER FOLDER.
#   - Then uses stored analysis to save crops, overlays, and CSV,
#     without recomputing anything heavy.

from pathlib import Path
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cv2

# ---------------------------------------------------------
# SAFE PHANTOM LOADER
# ---------------------------------------------------------
try:
    from pyphantom import Phantom
    ph = Phantom(init_camera=False)   # safest path for newer SDKs
except Exception:
    try:
        ph = Phantom()                # fallback for older SDKs
    except Exception as e:
        print("[WARNING] Phantom() could not fully initialize, but cine reading may still work.")
        print("Details:", e)
        ph = None

from pyphantom import cine

from frame_selector_full import (
    analyze_cine_full as analyze_cine,
    choose_best_frame_full as choose_best_frame,
)

from frame_cropping_full import (
    analyze_frame_geometric,
    crop_autocenter_simple,
)

# ---------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------
OUTPUT_ROOT = Path(
    r"C:\Users\justi\Downloads\coursework\testing_4mm\FULL\OUTPUT"
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

MAX_CNN_SIZE = 512
MIN_CNN_SIZE = 64


# ---------------------------------------------------------
# Safe cine loader
# ---------------------------------------------------------
def safe_load_cine(path: Path):
    try:
        c = cine.Cine.from_filepath(str(path))
        if c is None:
            return None
        _ = c.range
        return c
    except Exception as e:
        print(f"    [CINE LOAD ERROR] {path.name}: {e}")
        return None


# ---------------------------------------------------------
# Group by droplet ID
# ---------------------------------------------------------
def group_cines_by_droplet(folder: Path):
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


# ---------------------------------------------------------
# Darkness plot
# ---------------------------------------------------------
def save_darkness_plot(out_path, curve, first, last, best, name):
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


# ---------------------------------------------------------
# Geometric overlay with arrows + key
# ---------------------------------------------------------
def save_geometric_overlay(out_path, geo, best_idx):
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

    ax.axhline(0, color="blue", linewidth=1.5, label="Top of image")
    if y_top is not None:
        ax.axhline(y_top, color="green", linewidth=1.5, label="Top of droplet")
    if y_bottom is not None:
        ax.axhline(y_bottom, color="orange", linewidth=1.5, label="Bottom of droplet")
    if y_sphere is not None:
        ax.axhline(y_sphere, color="red", linewidth=1.5, label="Top of bottom sphere")

    x_arrow = int(W * 0.92)

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

    legend_elements = [
        Line2D([0], [0], color="blue",   lw=1.5, label="Top of image"),
        Line2D([0], [0], color="green",  lw=1.5, label="Top of droplet"),
        Line2D([0], [0], color="orange", lw=1.5, label="Bottom of droplet"),
        Line2D([0], [0], color="red",    lw=1.5, label="Top of bottom sphere"),
        Line2D([0], [0], color="yellow", lw=1.5, label="Top margin"),
        Line2D([0], [0], color="cyan",   lw=1.5, label="Bottom gap"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=6)

    ax.set_title(f"Best frame {best_idx}")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# =========================================================
#                 MAIN: PER-FOLDER MODE
# =========================================================
def process_every_10_perfolder(root_folder):
    root = Path(root_folder)
    subfolders = [p for p in sorted(root.iterdir()) if p.is_dir()]
    n_folders = len(subfolders)

    print(f"\n[PER-FOLDER] Found {n_folders} subfolders.\n")

    for f_idx, sub in enumerate(subfolders, start=1):
        print(f"\n==============================")
        print(f"[FOLDER {f_idx}/{n_folders}] {sub.name}")
        print(f"==============================")

        groups = group_cines_by_droplet(sub)
        n_groups = len(groups)
        if n_groups == 0:
            print("  [INFO] No droplets in this folder.")
            continue

        selected_indices = list(range(0, n_groups, 10))
        print(f"[{sub.name}] Using {len(selected_indices)} droplets (every 10th).")

        # ----------------------------------------------
        # PHASE 1: analyse only used cines for this folder
        # ----------------------------------------------
        analyses = {}  # key: (droplet_id, cam) -> dict(...)
        diams = []
        gaps = []

        for sel_idx, g_index in enumerate(selected_indices, start=1):
            droplet_id, cams = groups[g_index]
            print(f"  [CAL:{sub.name}] Droplet {sel_idx}/{len(selected_indices)}: ID {droplet_id}")

            for cam in ("g", "v"):
                path = cams.get(cam)
                if path is None:
                    continue

                print(f"    [CAL:{sub.name}] {cam}-cine: {path.name}")
                c = safe_load_cine(path)
                if c is None:
                    print("      [SKIP] Cine failed to load.")
                    continue

                dark = analyze_cine(c)
                curve = dark["darkness_curve"]
                first = dark["first_frame"]
                last = dark["last_frame"]

                best = choose_best_frame(c, curve, first)
                geo = analyze_frame_geometric(c, best)

                y_top = geo["y_top"]
                y_bottom = geo["y_bottom"]
                y_sphere = geo["y_bottom_sphere"]

                analyses[(droplet_id, cam)] = {
                    "path": path,
                    "first": first,
                    "last": last,
                    "curve": curve,
                    "best": best,
                    "geo": geo,
                }

                if (
                    y_top is not None
                    and y_bottom is not None
                    and y_sphere is not None
                    and y_bottom < y_sphere
                ):
                    diameter = float(y_bottom - y_top)
                    gap = float(y_sphere - y_bottom)
                    diams.append(diameter)
                    gaps.append(gap)

        # ----------------------------------------------
        # PHASE 2: calibrate crop size for THIS folder
        # ----------------------------------------------
        if not gaps:
            CNN_SIZE = 128
            print(f"[CAL:{sub.name}] No valid geometry; using fallback {CNN_SIZE}×{CNN_SIZE}.")
        else:
            max_diam = max(diams)
            min_gap = min(gaps)
            safety = 5.0

            crop_h = int(max_diam + 2 * max(0.0, (min_gap - safety)))
            crop_h = int(max(MIN_CNN_SIZE, min(MAX_CNN_SIZE, crop_h)))

            CNN_SIZE = crop_h

            print(f"\n[CAL RESULTS:{sub.name}]")
            print(f"  Used samples  : {len(diams)}")
            print(f"  Max diameter  : {max_diam:.2f} px")
            print(f"  Min gap       : {min_gap:.2f} px")
            print(f"  => Folder crop: {CNN_SIZE} × {CNN_SIZE} px\n")

        # ----------------------------------------------
        # PHASE 3: crops, overlays, CSV using stored analysis
        # ----------------------------------------------
        out_sub = OUTPUT_ROOT / sub.name
        out_sub.mkdir(parents=True, exist_ok=True)

        csv_path = out_sub / f"{sub.name}_summary_full.csv"
        print(f"[{sub.name}] Writing results to: {csv_path}")

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "droplet_id", "camera", "cine_file",
                "first_frame", "last_frame",
                "best_frame", "dark_fraction",
                "y_top", "y_bottom", "y_sphere",
                "crop_size_px", "crop_path"
            ])

            for g_counter, g_index in enumerate(selected_indices, start=1):
                droplet_id, cams = groups[g_index]
                print(f"\n  [Droplet {g_counter}/{len(selected_indices)}] ID = {droplet_id}")

                for cam in ("g", "v"):
                    path = cams.get(cam)
                    if path is None:
                        print(f"    [SKIP] No {cam}-camera cine.")
                        continue

                    info = analyses.get((droplet_id, cam))
                    if info is None:
                        print(f"    [SKIP] No analysis stored for {cam}-cine.")
                        continue

                    curve = info["curve"]
                    first = info["first"]
                    last = info["last"]
                    best = info["best"]
                    geo = info["geo"]

                    dark_val = float(curve[best - first])

                    y_top = geo["y_top"]
                    y_bottom = geo["y_bottom"]
                    y_sphere = geo["y_bottom_sphere"]
                    cx = geo["cx"]

                    crop_path = ""
                    if y_top is not None and y_bottom is not None:
                        crop = crop_autocenter_simple(geo["frame"], y_top, y_bottom, cx,
                              target_w=CNN_SIZE, target_h=CNN_SIZE,
                              y_sphere=y_sphere, safety=3)

                

                        out_crop = out_sub / f"{path.stem}_crop.png"
                        cv2.imwrite(str(out_crop), crop)
                        crop_path = str(out_crop)

                        print(f"    Crop size: {CNN_SIZE} × {CNN_SIZE} px")
                        print(f"    Saved crop: {out_crop.name}")
                    else:
                        print("    [SKIP] Droplet geometry invalid — no crop.")

                    save_darkness_plot(
                        out_sub / f"{path.stem}_darkness.png",
                        curve, first, last, best, path.name
                    )
                    save_geometric_overlay(
                        out_sub / f"{path.stem}_overlay.png",
                        geo, best
                    )

                    writer.writerow([
                        droplet_id, cam, path.name,
                        first, last,
                        best, dark_val,
                        y_top, y_bottom, y_sphere,
                        CNN_SIZE, crop_path
                    ])

        print(f"\n[FOLDER COMPLETE] CSV written: {csv_path}")
