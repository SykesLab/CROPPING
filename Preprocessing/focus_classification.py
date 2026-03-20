"""
Focus classification for pipeline outputs.

Classifies crops as sharp/medium/blurry using per-folder, per-camera thresholds
based on ERF blur sigma measurement. Lower ERF sigma = sharper.

Each camera within each material folder is classified independently to ensure
balanced training data across different optical setups.

Laplacian variance columns are retained in output CSVs for comparison.
"""

import logging
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import OUTPUT_ROOT
from crop_blur_measurement import measure_erf_blur
from focus_metrics import compute_all_focus_metrics

logger = logging.getLogger(__name__)


def classify_by_erf_sigma(
    erf_sigmas: np.ndarray,
    sharp_percentile: float = 25.0,
    blur_percentile: float = 75.0,
) -> Tuple[np.ndarray, float, float]:
    """
    Classify focus using ERF sigma values. Lower sigma = sharper.

    Args:
        erf_sigmas: Array of ERF sigma values (NaN entries are classified as None)
        sharp_percentile: Percentile below which crops are 'sharp' (default 25th)
        blur_percentile: Percentile above which crops are 'blurry' (default 75th)

    Returns:
        (classifications array, sharp_threshold, blur_threshold)
    """
    valid = erf_sigmas[~np.isnan(erf_sigmas)]
    if len(valid) < 4:
        return np.full(len(erf_sigmas), None, dtype=object), 0.0, 0.0

    sharp_thresh = float(np.percentile(valid, sharp_percentile))
    blur_thresh = float(np.percentile(valid, blur_percentile))

    classifications = []
    for sigma in erf_sigmas:
        if np.isnan(sigma):
            classifications.append(None)
        elif sigma <= sharp_thresh:
            classifications.append("sharp")
        elif sigma >= blur_thresh:
            classifications.append("blurry")
        else:
            classifications.append("medium")

    return np.array(classifications, dtype=object), sharp_thresh, blur_thresh


def run_focus_classification() -> None:
    """
    Run per-folder, per-camera focus classification on all summary CSVs.

    Uses ERF sigma as the primary ranking metric (lower = sharper).
    Retains laplacian variance columns for comparison.

    Outputs saved to OUTPUT/Focus/:
        - focus_classified_all.csv: All crops with classifications
        - sharp_crops.csv: Only sharp crops (for CNN training)
        - focus_folder_stats.csv: Per-folder+camera threshold statistics
        - focus_classification_summary.png: Visualisation
        - material/camera/: Sharp crops organized by material and camera
    """
    logger.info("Starting ERF-based focus classification...")

    all_data: List[pd.DataFrame] = []
    folder_stats: List[dict] = []
    total_sharp_copied = 0
    erf_fail_count = 0

    # Create Focus output directory
    focus_dir = OUTPUT_ROOT / "Focus"
    focus_dir.mkdir(parents=True, exist_ok=True)

    # Process each folder's CSV
    for csv_path in sorted(OUTPUT_ROOT.rglob("*_summary.csv")):
        # Skip CSVs inside the Focus directory
        if "Focus" in csv_path.parts:
            continue

        try:
            df = pd.read_csv(csv_path)

            if 'camera' not in df.columns:
                print(f"  Skipping {csv_path.name} (no camera column)")
                continue

            folder_name = csv_path.parent.name

            # Add folder column
            df['folder'] = folder_name

            # Measure ERF sigma for every crop in this folder
            print(f"  {folder_name}: measuring ERF sigma on {len(df)} crops...")
            erf_sigmas = []
            for _, row in df.iterrows():
                crop_path = row.get('crop_path', '')
                sigma = None
                if crop_path and Path(crop_path).exists():
                    try:
                        img = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            sigma = measure_erf_blur(img)
                    except Exception:
                        pass
                if sigma is None:
                    erf_fail_count += 1
                erf_sigmas.append(sigma if sigma is not None else float('nan'))

            df['erf_sigma'] = erf_sigmas
            n_measured = sum(1 for s in erf_sigmas if not np.isnan(s))
            print(f"    ERF measured: {n_measured}/{len(df)}")

            # Initialize focus_class column
            df['focus_class'] = None

            # Classify per-camera within this folder using ERF sigma
            folder_sharp = 0
            folder_medium = 0
            folder_blurry = 0

            for cam in df['camera'].unique():
                cam_mask = df['camera'] == cam
                cam_df = df[cam_mask]
                cam_erf = cam_df['erf_sigma'].values

                valid_erf = cam_erf[~np.isnan(cam_erf)]
                if len(valid_erf) < 4:
                    continue

                # Per-folder+camera classification using ERF sigma
                classifications, sharp_thresh, blur_thresh = classify_by_erf_sigma(cam_erf)

                # Apply classifications to this camera's rows
                df.loc[cam_mask, 'focus_class'] = classifications

                # Count for this camera
                cam_sharp = (df.loc[cam_mask, 'focus_class'] == 'sharp').sum()
                cam_medium = (df.loc[cam_mask, 'focus_class'] == 'medium').sum()
                cam_blurry = (df.loc[cam_mask, 'focus_class'] == 'blurry').sum()

                folder_sharp += cam_sharp
                folder_medium += cam_medium
                folder_blurry += cam_blurry

                # Collect per-camera stats
                folder_stats.append({
                    'folder': folder_name,
                    'camera': cam,
                    'n_total': len(cam_df),
                    'n_measured': int(np.sum(~np.isnan(cam_erf))),
                    'n_sharp': cam_sharp,
                    'n_medium': cam_medium,
                    'n_blurry': cam_blurry,
                    'erf_sharp_thresh': sharp_thresh,
                    'erf_blur_thresh': blur_thresh,
                    'erf_sigma_mean': float(np.nanmean(cam_erf)),
                    'erf_sigma_std': float(np.nanstd(cam_erf)),
                    # Keep laplacian stats for comparison
                    'laplacian_mean': float(cam_df['laplacian_var'].mean())
                        if 'laplacian_var' in cam_df.columns else None,
                })

                # Copy sharp images to Focus/folder/camera/
                sharp_cam_df = df[cam_mask & (df['focus_class'] == 'sharp')]
                if len(sharp_cam_df) > 0:
                    cam_focus_dir = focus_dir / folder_name / cam
                    cam_focus_dir.mkdir(parents=True, exist_ok=True)

                    for _, row in sharp_cam_df.iterrows():
                        crop_path = row.get('crop_path', '')
                        if crop_path and Path(crop_path).exists():
                            dest_path = cam_focus_dir / Path(crop_path).name
                            shutil.copy2(crop_path, dest_path)
                            total_sharp_copied += 1

            # Save updated CSV (with erf_sigma and focus_class columns)
            df.to_csv(csv_path, index=False)

            all_data.append(df)

            print(f"    Classified: {folder_sharp} sharp / {folder_medium} medium / {folder_blurry} blurry")

        except Exception as e:
            print(f"  Error processing {csv_path.name}: {e}")
            import traceback
            traceback.print_exc()

    if not all_data:
        print("  No valid CSVs found for focus classification")
        return

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)

    # Save combined CSV to Focus directory
    combined_path = focus_dir / "focus_classified_all.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"\n  Saved combined dataset: Focus/{combined_path.name} ({len(combined_df)} crops)")

    # Save sharp-only CSV to Focus directory
    sharp_only_df = combined_df[combined_df['focus_class'] == 'sharp'].copy()
    # Add diameter column
    sharp_only_df["diameter_px"] = sharp_only_df["y_bottom"] - sharp_only_df["y_top"]
    # Add filename column from crop_path
    sharp_only_df["filename"] = sharp_only_df["crop_path"].apply(
        lambda p: Path(p).name if pd.notna(p) else "")
    # native_blur_sigma is now just the erf_sigma column (already measured)
    sharp_only_df["native_blur_sigma"] = sharp_only_df["erf_sigma"]

    sharp_path = focus_dir / "sharp_crops.csv"
    sharp_only_df.to_csv(sharp_path, index=False)
    print(f"  Saved sharp crops list: Focus/{sharp_path.name} ({len(sharp_only_df)} crops)")

    # Save folder statistics to Focus directory
    stats_df = pd.DataFrame(folder_stats)
    stats_path = focus_dir / "focus_folder_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"  Saved folder statistics: Focus/{stats_path.name}")

    # Print summary
    total_sharp = combined_df['focus_class'].eq('sharp').sum()
    total_medium = combined_df['focus_class'].eq('medium').sum()
    total_blurry = combined_df['focus_class'].eq('blurry').sum()

    print(f"\n  FOCUS CLASSIFICATION SUMMARY (ERF sigma, per-folder+camera thresholds)")
    print(f"  ─────────────────────────────────────────────────────────────────────────")
    print(f"  Total crops:     {len(combined_df)}")
    print(f"  ERF measured:    {len(combined_df) - erf_fail_count}/{len(combined_df)}")
    print(f"  Sharp:           {total_sharp} ({100*total_sharp/len(combined_df):.1f}%)")
    print(f"  Medium:          {total_medium} ({100*total_medium/len(combined_df):.1f}%)")
    print(f"  Blurry:          {total_blurry} ({100*total_blurry/len(combined_df):.1f}%)")
    print(f"  Sharp copied:    {total_sharp_copied}")

    # Per-camera summary
    if 'camera' in combined_df.columns:
        print(f"\n  Per-camera breakdown:")
        for cam in sorted(combined_df['camera'].dropna().unique()):
            cam_df = combined_df[combined_df['camera'] == cam]
            cam_sharp = (cam_df['focus_class'] == 'sharp').sum()
            cam_total = len(cam_df)
            cam_erf_mean = cam_df['erf_sigma'].mean()
            if cam_total > 0:
                print(f"    Camera {cam}: {cam_sharp}/{cam_total} sharp "
                      f"({100*cam_sharp/cam_total:.1f}%), mean ERF σ={cam_erf_mean:.2f}px")

    # Generate summary plot
    _generate_summary_plot(combined_df, stats_df, focus_dir, total_sharp, total_medium, total_blurry)


def _generate_summary_plot(
    combined_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    focus_dir: Path,
    total_sharp: int,
    total_medium: int,
    total_blurry: int,
) -> None:
    """Generate focus classification summary plot."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: classification breakdown
        ax1 = axes[0]
        counts = [total_sharp, total_medium, total_blurry]
        labels = [f'Sharp\n({total_sharp})', f'Medium\n({total_medium})', f'Blurry\n({total_blurry})']
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        ax1.pie(counts, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
        ax1.set_title('Focus Classification (ERF sigma, Per-Folder+Camera)')

        # Right: per-folder+camera sharp count
        ax2 = axes[1]

        cam_colors = {'g': '#22c55e', 'v': '#8b5cf6', 'm': '#ec4899'}

        if 'camera' in stats_df.columns and len(stats_df) > 0:
            folders = stats_df['folder'].unique()
            cam_order = ['m', 'g', 'v']

            y_positions = []
            bar_values = []
            bar_colors = []
            y_labels = []
            folder_label_positions = {}
            separator_positions = []

            y_pos = 0
            for folder in sorted(folders):
                folder_data = stats_df[stats_df['folder'] == folder]
                folder_start = y_pos

                for cam in cam_order:
                    cam_row = folder_data[folder_data['camera'] == cam]
                    if len(cam_row) > 0:
                        y_positions.append(y_pos)
                        bar_values.append(cam_row['n_sharp'].values[0])
                        bar_colors.append(cam_colors.get(cam, '#888888'))
                        y_labels.append(cam)
                        y_pos += 1

                if y_pos > folder_start:
                    folder_label_positions[folder] = (folder_start + y_pos - 1) / 2
                    separator_positions.append(y_pos - 0.5 + 0.25)
                    y_pos += 0.5

            ax2.barh(y_positions, bar_values, color=bar_colors, alpha=0.8, height=1.0)

            for sep_y in separator_positions[:-1]:
                ax2.axhline(y=sep_y, color='black', linewidth=0.8, linestyle='-')

            ax2.set_yticks(y_positions)
            ax2.set_yticklabels(y_labels, fontsize=8)

            for folder, y_mid in folder_label_positions.items():
                ax2.text(-0.08, y_mid, folder, transform=ax2.get_yaxis_transform(),
                         ha='right', va='center', fontsize=7, fontweight='bold')

            ax2.set_xlim(left=0)
            plt.subplots_adjust(left=0.18)
        else:
            stats_df_sorted = stats_df.sort_values('n_sharp', ascending=True)
            ax2.barh(range(len(stats_df_sorted)), stats_df_sorted['n_sharp'], color='#22c55e', alpha=0.7)
            ax2.set_yticks(range(len(stats_df_sorted)))
            ax2.set_yticklabels(stats_df_sorted['folder'].tolist(), fontsize=6)

        ax2.set_xlabel('Number of Sharp Crops')
        ax2.set_title('Sharp Crops per Material/Camera')

        plt.tight_layout()
        plot_path = focus_dir / "focus_classification_summary.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Saved summary plot: Focus/{plot_path.name}")

    except Exception as e:
        print(f"  Could not generate plot: {e}")
