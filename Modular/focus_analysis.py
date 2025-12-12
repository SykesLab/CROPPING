"""Focus quality analysis and visualisation.

Standalone script to analyse focus metrics from:
    1. Existing CSV output (if metrics already computed)
    2. Crop images directly (computes metrics on the fly)

Supports two classification modes:
    - Global: Single threshold for entire dataset (original behaviour)
    - Per-folder: Each folder gets its own thresholds (for diverse datasets)

Generates:
    - Distribution histograms
    - Suggested thresholds
    - Focus quality report
    - Ranked lists of sharpest/blurriest crops
    - Focus/ folder with sharp images copied (per-folder mode)

Usage:
    python focus_analysis.py                      # Scan OUTPUT_ROOT for crops
    python focus_analysis.py <folder>             # Scan folder for crop PNGs
    python focus_analysis.py --from-csv <csv>     # Read pre-computed from CSV
    python focus_analysis.py --per-folder         # Use per-folder thresholds
    python focus_analysis.py --copy-sharp         # Copy sharp images to Focus/
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config_modular import OUTPUT_ROOT
from focus_metrics_modular import (
    classify_folder_focus,
    compute_all_focus_metrics,
    compute_dataset_statistics,
    suggest_thresholds,
)


def compute_metrics_from_crops(
    root: Path,
    pattern: str = "*_crop.png",
) -> pd.DataFrame:
    """Scan folder for crop images and compute focus metrics.

    Args:
        root: Root directory to scan (recursive).
        pattern: Glob pattern for crop files.

    Returns:
        DataFrame with crop paths and computed metrics.
    """
    crop_files = list(root.rglob(pattern))
    
    # Exclude Focus folder
    crop_files = [p for p in crop_files if "Focus" not in p.parts]
    
    if not crop_files:
        raise FileNotFoundError(
            f"No crop images matching '{pattern}' found in {root}"
        )
    
    print(f"Found {len(crop_files)} crop images")
    print("Computing focus metrics...")
    
    records = []
    for crop_path in tqdm(crop_files, desc="Analysing crops"):
        try:
            # Load image
            img = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  Warning: Could not load {crop_path}")
                continue
            
            # Compute metrics
            metrics = compute_all_focus_metrics(img)
            
            # Build record
            record = {
                "crop_path": str(crop_path),
                "filename": crop_path.name,
                "folder": crop_path.parent.name,
                **metrics,
            }
            records.append(record)
            
        except Exception as e:
            print(f"  Warning: Error processing {crop_path}: {e}")
    
    df = pd.DataFrame(records)
    print(f"Successfully processed {len(df)} crops")
    
    return df


def load_all_csvs(root: Path) -> pd.DataFrame:
    """Load and concatenate all summary CSVs from output folder.

    Args:
        root: Output root directory.

    Returns:
        Combined DataFrame with all crops.
    """
    csv_files = list(root.rglob("*_summary.csv"))
    
    # Exclude Focus folder
    csv_files = [p for p in csv_files if "Focus" not in p.parts]
    
    if not csv_files:
        raise FileNotFoundError(f"No summary CSVs found in {root}")
    
    print(f"Found {len(csv_files)} CSV files")
    
    dfs = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            df["folder"] = csv_path.parent.name
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: Could not load {csv_path.name}: {e}")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined)} total crops")
    
    return combined


def analyse_focus_distribution(
    df: pd.DataFrame,
    metric: str = "laplacian_var",
) -> Dict[str, float]:
    """Analyse distribution of a focus metric.

    Args:
        df: DataFrame with focus metrics.
        metric: Column name to analyse.

    Returns:
        Statistics dictionary.
    """
    # Drop NaN values
    values = df[metric].dropna().values
    
    if len(values) == 0:
        print(f"Warning: No valid values for {metric}")
        return {}
    
    stats = compute_dataset_statistics(values)
    return stats


def plot_focus_distribution(
    df: pd.DataFrame,
    metric: str = "laplacian_var",
    save_path: Optional[Path] = None,
    sharp_thresh: Optional[float] = None,
    blur_thresh: Optional[float] = None,
) -> None:
    """Plot histogram of focus metric distribution.

    Args:
        df: DataFrame with focus metrics.
        metric: Column name to plot.
        save_path: Where to save plot (None = show).
        sharp_thresh: Threshold for "sharp" classification.
        blur_thresh: Threshold for "blurry" classification.
    """
    values = df[metric].dropna().values
    
    if len(values) == 0:
        print(f"No data for {metric}")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Histogram
    ax1 = axes[0]
    ax1.hist(values, bins=50, edgecolor="black", alpha=0.7)
    ax1.set_xlabel(metric)
    ax1.set_ylabel("Count")
    ax1.set_title(f"Distribution of {metric}")
    
    # Add threshold lines
    if sharp_thresh is not None:
        ax1.axvline(sharp_thresh, color="green", linestyle="--", 
                    label=f"Sharp threshold: {sharp_thresh:.0f}")
    if blur_thresh is not None:
        ax1.axvline(blur_thresh, color="red", linestyle="--",
                    label=f"Blur threshold: {blur_thresh:.0f}")
    
    if sharp_thresh is not None or blur_thresh is not None:
        ax1.legend()
    
    # Right: Log-scale histogram (shows tails better)
    ax2 = axes[1]
    ax2.hist(values, bins=50, edgecolor="black", alpha=0.7)
    ax2.set_xlabel(metric)
    ax2.set_ylabel("Count (log)")
    ax2.set_yscale("log")
    ax2.set_title(f"Distribution of {metric} (log scale)")
    
    if sharp_thresh is not None:
        ax2.axvline(sharp_thresh, color="green", linestyle="--")
    if blur_thresh is not None:
        ax2.axvline(blur_thresh, color="red", linestyle="--")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_metric_comparison(
    df: pd.DataFrame,
    metrics: List[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """Plot correlation between different focus metrics.

    Args:
        df: DataFrame with focus metrics.
        metrics: List of metric columns to compare.
        save_path: Where to save plot (None = show).
    """
    if metrics is None:
        metrics = [
            "laplacian_var",
            "tenengrad_var", 
            "brenner",
            "norm_laplacian",
        ]
    
    # Filter to only existing columns
    metrics = [m for m in metrics if m in df.columns]
    
    if len(metrics) < 2:
        print("Need at least 2 metrics for comparison")
        return
    
    n = len(metrics)
    fig, axes = plt.subplots(n-1, n-1, figsize=(3*(n-1), 3*(n-1)))
    
    if n == 2:
        axes = np.array([[axes]])
    
    for i in range(n - 1):
        for j in range(n - 1):
            ax = axes[i, j]
            if j <= i:
                x_metric = metrics[j]
                y_metric = metrics[i + 1]
                
                x = df[x_metric].dropna()
                y = df[y_metric].dropna()
                
                # Align indices
                common = x.index.intersection(y.index)
                x = x.loc[common]
                y = y.loc[common]
                
                ax.scatter(x, y, alpha=0.3, s=5)
                ax.set_xlabel(x_metric, fontsize=8)
                ax.set_ylabel(y_metric, fontsize=8)
                
                # Correlation coefficient
                if len(x) > 0:
                    corr = np.corrcoef(x, y)[0, 1]
                    ax.set_title(f"r={corr:.2f}", fontsize=9)
            else:
                ax.axis("off")
    
    plt.suptitle("Focus Metric Correlations")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()


def classify_global(
    df: pd.DataFrame,
    metric: str = "laplacian_var",
    sharp_percentile: float = 75.0,
    blur_percentile: float = 25.0,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Classify crops using global thresholds.

    Args:
        df: DataFrame with focus metrics.
        metric: Metric to use for classification.
        sharp_percentile: Percentile above which images are "sharp".
        blur_percentile: Percentile below which images are "blurry".

    Returns:
        Tuple of (classified DataFrame, counts dict).
    """
    values = df[metric].dropna()
    
    sharp_thresh, blur_thresh = suggest_thresholds(
        values.values,
        sharp_percentile=sharp_percentile,
        blur_percentile=blur_percentile,
    )
    
    def classify(val):
        if pd.isna(val):
            return "unknown"
        elif val >= sharp_thresh:
            return "sharp"
        elif val <= blur_thresh:
            return "blurry"
        else:
            return "medium"
    
    df = df.copy()
    df["focus_class"] = df[metric].apply(classify)
    
    counts = df["focus_class"].value_counts().to_dict()
    
    print("\n" + "=" * 50)
    print("FOCUS QUALITY REPORT (Global Thresholds)")
    print("=" * 50)
    print(f"\nMetric used: {metric}")
    print(f"Sharp threshold (p{sharp_percentile:.0f}): {sharp_thresh:.1f}")
    print(f"Blur threshold (p{blur_percentile:.0f}): {blur_thresh:.1f}")
    print(f"\nClassification breakdown:")
    for cls, count in sorted(counts.items()):
        pct = 100 * count / len(df)
        print(f"  {cls}: {count} ({pct:.1f}%)")
    
    return df, counts


def classify_per_folder(
    df: pd.DataFrame,
    metric: str = "laplacian_var",
    sharp_percentile: float = 75.0,
    blur_percentile: float = 25.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Classify crops using per-folder thresholds.

    Args:
        df: DataFrame with focus metrics and 'folder' column.
        metric: Metric to use for classification.
        sharp_percentile: Percentile above which images are "sharp".
        blur_percentile: Percentile below which images are "blurry".

    Returns:
        Tuple of (classified DataFrame, folder stats DataFrame, counts dict).
    """
    if "folder" not in df.columns:
        raise ValueError("DataFrame must have 'folder' column for per-folder classification")
    
    df = df.copy()
    df["focus_class"] = None
    
    folder_stats = []
    
    print("\n" + "=" * 60)
    print("PER-FOLDER FOCUS CLASSIFICATION")
    print("=" * 60)
    
    for folder in sorted(df["folder"].unique()):
        folder_mask = df["folder"] == folder
        folder_df = df.loc[folder_mask]
        
        scores = folder_df[metric].dropna().values
        
        if len(scores) < 4:
            print(f"  {folder}: Skipped (too few samples: {len(scores)})")
            continue
        
        # Per-folder classification
        classifications, sharp_thresh, blur_thresh = classify_folder_focus(
            scores, sharp_percentile, blur_percentile
        )
        
        # Apply classifications
        valid_idx = folder_df[metric].notna()
        df.loc[folder_mask & valid_idx, "focus_class"] = classifications
        
        # Collect stats
        n_sharp = (classifications == "sharp").sum()
        n_medium = (classifications == "medium").sum()
        n_blurry = (classifications == "blurry").sum()
        
        folder_stats.append({
            "folder": folder,
            "n_total": len(folder_df),
            "n_sharp": n_sharp,
            "n_medium": n_medium,
            "n_blurry": n_blurry,
            "sharp_thresh": sharp_thresh,
            "blur_thresh": blur_thresh,
            "mean_laplacian": folder_df[metric].mean(),
        })
        
        print(f"  {folder}: {n_sharp} sharp / {n_medium} medium / {n_blurry} blurry "
              f"(thresh: {blur_thresh:.0f}-{sharp_thresh:.0f})")
    
    stats_df = pd.DataFrame(folder_stats)
    counts = df["focus_class"].value_counts().to_dict()
    
    print(f"\n" + "-" * 60)
    print("TOTALS")
    print("-" * 60)
    for cls, count in sorted(counts.items()):
        if cls is not None:
            pct = 100 * count / len(df[df["focus_class"].notna()])
            print(f"  {cls}: {count} ({pct:.1f}%)")
    
    return df, stats_df, counts


def copy_sharp_images(
    df: pd.DataFrame,
    output_root: Path,
) -> int:
    """Copy sharp images to Focus/{folder}/ directory.

    Args:
        df: Classified DataFrame with 'focus_class' and 'crop_path' columns.
        output_root: Output root directory.

    Returns:
        Number of images copied.
    """
    focus_dir = output_root / "Focus"
    focus_dir.mkdir(parents=True, exist_ok=True)
    
    sharp_df = df[df["focus_class"] == "sharp"]
    
    if len(sharp_df) == 0:
        print("No sharp images to copy")
        return 0
    
    copied = 0
    print(f"\nCopying {len(sharp_df)} sharp images to Focus/...")
    
    for _, row in tqdm(sharp_df.iterrows(), total=len(sharp_df), desc="Copying"):
        crop_path = Path(row["crop_path"])
        folder = row.get("folder", crop_path.parent.name)
        
        if not crop_path.exists():
            continue
        
        dest_dir = focus_dir / folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = dest_dir / crop_path.name
        shutil.copy2(crop_path, dest_path)
        copied += 1
    
    print(f"Copied {copied} images to Focus/")
    return copied


def show_extreme_crops(
    df: pd.DataFrame,
    metric: str = "laplacian_var",
    n: int = 5,
) -> None:
    """Print paths of sharpest and blurriest crops.

    Args:
        df: DataFrame with focus metrics and crop_path.
        metric: Metric to sort by.
        n: Number of examples to show.
    """
    # Filter to rows with valid metric and crop path
    valid = df[df[metric].notna()].copy()
    if "crop_path" in valid.columns:
        valid = valid[valid["crop_path"].notna()]
    
    if len(valid) == 0:
        print("No valid crops with metrics")
        return
    
    sorted_df = valid.sort_values(metric, ascending=False)
    
    print(f"\n{'='*50}")
    print(f"TOP {n} SHARPEST CROPS ({metric})")
    print("=" * 50)
    for i, (_, row) in enumerate(sorted_df.head(n).iterrows()):
        path_str = row.get('crop_path', row.get('filename', 'N/A'))
        folder = row.get('folder', '')
        print(f"{i+1}. {row[metric]:.1f} — {folder}/{Path(path_str).name}")
    
    print(f"\n{'='*50}")
    print(f"TOP {n} BLURRIEST CROPS ({metric})")
    print("=" * 50)
    for i, (_, row) in enumerate(sorted_df.tail(n).iterrows()):
        path_str = row.get('crop_path', row.get('filename', 'N/A'))
        folder = row.get('folder', '')
        print(f"{i+1}. {row[metric]:.1f} — {folder}/{Path(path_str).name}")


def generate_summary_plot(
    df: pd.DataFrame,
    stats_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Generate summary visualisation for per-folder classification.

    Args:
        df: Classified DataFrame.
        stats_df: Folder statistics DataFrame.
        save_path: Where to save the plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Classification pie chart
    ax1 = axes[0]
    counts = df["focus_class"].value_counts()
    colors = {"sharp": "#2ecc71", "medium": "#f39c12", "blurry": "#e74c3c"}
    pie_colors = [colors.get(c, "gray") for c in counts.index]
    
    wedges, texts, autotexts = ax1.pie(
        counts.values,
        labels=[f"{c}\n({v})" for c, v in counts.items()],
        colors=pie_colors,
        autopct="%1.0f%%",
        startangle=90,
    )
    ax1.set_title("Focus Classification\n(Per-Folder Thresholds)")
    
    # Right: Sharp count per folder
    ax2 = axes[1]
    stats_sorted = stats_df.sort_values("n_sharp", ascending=True)
    
    bar_colors = []
    for _, row in stats_sorted.iterrows():
        if row["n_sharp"] > row["n_medium"]:
            bar_colors.append("#2ecc71")
        elif row["n_blurry"] > row["n_medium"]:
            bar_colors.append("#e74c3c")
        else:
            bar_colors.append("#f39c12")
    
    ax2.barh(range(len(stats_sorted)), stats_sorted["n_sharp"], color=bar_colors, alpha=0.7)
    ax2.set_yticks(range(len(stats_sorted)))
    ax2.set_yticklabels(stats_sorted["folder"], fontsize=8)
    ax2.set_xlabel("Number of Sharp Crops")
    ax2.set_title("Sharp Crops per Folder")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyse focus metrics from crop images or CSV"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to folder with crops OR path to CSV file (default: OUTPUT_ROOT)",
    )
    parser.add_argument(
        "--from-csv",
        type=str,
        default=None,
        help="Read pre-computed metrics from CSV instead of images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for Focus folder (default: same as input)",
    )
    parser.add_argument(
        "--pattern",
        default="*_crop.png",
        help="Glob pattern for crop files (default: *_crop.png)",
    )
    parser.add_argument(
        "--metric",
        default="laplacian_var",
        help="Focus metric to use (default: laplacian_var)",
    )
    parser.add_argument(
        "--sharp-percentile",
        type=float,
        default=75.0,
        help="Percentile for sharp threshold (default: 75)",
    )
    parser.add_argument(
        "--blur-percentile",
        type=float,
        default=25.0,
        help="Percentile for blur threshold (default: 25)",
    )
    parser.add_argument(
        "--per-folder",
        action="store_true",
        default=True,
        help="Use per-folder thresholds instead of global (default: True)",
    )
    parser.add_argument(
        "--global-thresh",
        action="store_true",
        help="Use global thresholds instead of per-folder",
    )
    parser.add_argument(
        "--copy-sharp",
        action="store_true",
        default=True,
        help="Copy sharp images to Focus/ directory (default: True)",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Don't copy sharp images",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots to files instead of showing",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="Save computed metrics to this CSV path",
    )
    
    args = parser.parse_args()
    
    # Determine input mode
    csv_path = None
    
    # Check if positional arg is a CSV file
    if args.path and args.path.endswith('.csv'):
        csv_path = Path(args.path)
    elif args.from_csv:
        csv_path = Path(args.from_csv)
    
    if csv_path:
        # Mode 1: Read from existing CSV (no recomputation!)
        if not csv_path.exists():
            print(f"Error: CSV not found: {csv_path}")
            sys.exit(1)
        
        print(f"Loading metrics from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = csv_path.parent
        
        print(f"Loaded {len(df)} crops from CSV (no recomputation needed)")
        
    else:
        # Mode 2: Compute metrics from crop images
        if args.path is None:
            input_path = OUTPUT_ROOT
        else:
            input_path = Path(args.path)
        
        if not input_path.is_dir():
            print(f"Error: {input_path} is not a directory")
            sys.exit(1)
        
        output_dir = args.output_dir if args.output_dir else input_path
        output_dir = Path(output_dir)
        
        df = compute_metrics_from_crops(input_path, pattern=args.pattern)
        
        # Save computed metrics
        if args.save_csv:
            save_path = Path(args.save_csv)
        else:
            save_path = output_dir / "focus_metrics_computed.csv"
        
        df.to_csv(save_path, index=False)
        print(f"Saved computed metrics: {save_path}")
    
    # Check for focus metrics
    if args.metric not in df.columns:
        print(f"Error: Column '{args.metric}' not found in data")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Ensure folder column exists
    if "folder" not in df.columns and "crop_path" in df.columns:
        df["folder"] = df["crop_path"].apply(lambda p: Path(p).parent.name)
    
    # Analyse distribution
    print("\n" + "=" * 50)
    print("FOCUS METRIC STATISTICS")
    print("=" * 50)
    
    stats = analyse_focus_distribution(df, args.metric)
    for k, v in stats.items():
        print(f"  {k}: {v:.2f}")
    
    # Handle flag overrides
    use_per_folder = args.per_folder and not args.global_thresh
    do_copy_sharp = args.copy_sharp and not args.no_copy
    
    # Classification
    if use_per_folder:
        df_classified, folder_stats, counts = classify_per_folder(
            df,
            metric=args.metric,
            sharp_percentile=args.sharp_percentile,
            blur_percentile=args.blur_percentile,
        )
        
        # Create Focus directory
        focus_dir = output_dir / "Focus"
        focus_dir.mkdir(parents=True, exist_ok=True)
        
        # Save outputs to Focus/
        df_classified.to_csv(focus_dir / "focus_classified_all.csv", index=False)
        print(f"\nSaved: Focus/focus_classified_all.csv ({len(df_classified)} crops)")
        
        sharp_df = df_classified[df_classified["focus_class"] == "sharp"]
        sharp_df.to_csv(focus_dir / "sharp_crops.csv", index=False)
        print(f"Saved: Focus/sharp_crops.csv ({len(sharp_df)} crops)")
        
        folder_stats.to_csv(focus_dir / "focus_folder_stats.csv", index=False)
        print(f"Saved: Focus/focus_folder_stats.csv")
        
        # Generate summary plot
        generate_summary_plot(
            df_classified,
            folder_stats,
            focus_dir / "focus_classification_summary.png",
        )
        
        # Copy sharp images if requested
        if do_copy_sharp:
            copy_sharp_images(df_classified, output_dir)
        
    else:
        df_classified, counts = classify_global(
            df,
            metric=args.metric,
            sharp_percentile=args.sharp_percentile,
            blur_percentile=args.blur_percentile,
        )
        
        # Save classified CSV
        output_csv = output_dir / "focus_classified.csv"
        df_classified.to_csv(output_csv, index=False)
        print(f"\nSaved classified data: {output_csv}")
    
    # Show examples
    show_extreme_crops(df_classified, args.metric, n=5)
    
    # Distribution plots (only for global mode)
    if not use_per_folder:
        sharp_thresh, blur_thresh = suggest_thresholds(
            df[args.metric].dropna().values,
            sharp_percentile=args.sharp_percentile,
            blur_percentile=args.blur_percentile,
        )
        
        if args.save_plots:
            plot_focus_distribution(
                df,
                args.metric,
                save_path=output_dir / f"focus_distribution_{args.metric}.png",
                sharp_thresh=sharp_thresh,
                blur_thresh=blur_thresh,
            )
            plot_metric_comparison(
                df,
                save_path=output_dir / "focus_metric_correlations.png",
            )
        else:
            plot_focus_distribution(
                df,
                args.metric,
                sharp_thresh=sharp_thresh,
                blur_thresh=blur_thresh,
            )
            plot_metric_comparison(df)
    
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
