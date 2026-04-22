"""
Compare Ground Truth Blur vs Model Predictions

Compares the ground truth blur values from synthetic data generation (metadata.csv)
against model predictions when running inference on the same synthetic blur images.

This validates how accurately the model predicts the blur values it was trained on.

Usage:
    python compare_results.py --metadata synthetic_data/metadata.csv \
                              --inference inference_results/inference_20260115_204507/blur/blur_results.csv \
                              --output comparison_results
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

N_BINS = 4  # consistent with training/test binning


def _equal_bins(values: np.ndarray, n: int = N_BINS):
    """Compute n equal-width bins from the data range. Returns (edges, labels)."""
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    edges = np.linspace(lo, hi, n + 1).tolist()
    labels = [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(n)]
    return edges, labels


def _blur_term(df: pd.DataFrame) -> str:
    """Return display term based on columns: 'σ' for direct mode, 'CoC' for optical."""
    return "σ" if 'sigma_px' in df.columns else "CoC"


def _blur_col(df: pd.DataFrame) -> str:
    """Return the blur column name from DataFrame."""
    return 'sigma_px' if 'sigma_px' in df.columns else 'coc_px'


def load_ground_truth(metadata_path: Path) -> pd.DataFrame:
    """Load ground truth blur values from metadata.csv."""
    df = pd.read_csv(metadata_path)

    # Ensure index is string for matching
    df['index'] = df['index'].astype(str).str.zfill(6)
    df = df.set_index('index')

    bt = _blur_term(df)
    col = _blur_col(df)
    print(f"Loaded ground truth: {len(df)} samples")
    print(f"  {bt} range: [{df[col].min():.2f}, {df[col].max():.2f}] px")
    print(f"  Defocus range: [{df['defocus_mm'].min():.3f}, {df['defocus_mm'].max():.3f}] mm")

    return df


def load_inference_results(inference_path: Path) -> pd.DataFrame:
    """Load inference predictions from blur_results.csv."""
    df = pd.read_csv(inference_path)

    # Extract index from filename (e.g., "000000.png" -> "000000")
    df['index'] = df['filename'].str.replace('.png', '', regex=False).str.zfill(6)
    df = df.set_index('index')

    bt = _blur_term(df)
    col = _blur_col(df)
    print(f"Loaded inference results: {len(df)} samples")
    print(f"  Predicted {bt} range: [{df[col].min():.2f}, {df[col].max():.2f}] px")

    return df


def merge_datasets(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """Merge ground truth and predictions on sample index."""

    # Join on index
    merged = gt_df.join(pred_df, lsuffix='_gt', rsuffix='_pred', how='inner')

    print(f"\nMatched {len(merged)} samples between ground truth and predictions")

    if len(merged) < len(gt_df):
        print(f"  Warning: {len(gt_df) - len(merged)} ground truth samples not found in predictions")
    if len(merged) < len(pred_df):
        print(f"  Warning: {len(pred_df) - len(merged)} predictions not found in ground truth")

    # Create standardized blur columns regardless of mode (sigma_px or coc_px)
    col = _blur_col(gt_df)  # 'sigma_px' or 'coc_px'
    merged['blur_px_gt'] = merged[f'{col}_gt']
    merged['blur_px_pred'] = merged[f'{col}_pred']
    merged['blur_error_px'] = merged['blur_px_pred'] - merged['blur_px_gt']
    merged['blur_abs_error_px'] = merged['blur_error_px'].abs()
    merged['blur_pct_error'] = (merged['blur_error_px'] / merged['blur_px_gt']) * 100

    merged['defocus_error_mm'] = merged['defocus_mm_pred'] - merged['defocus_mm_gt']
    merged['defocus_abs_error_mm'] = merged['defocus_error_mm'].abs()
    merged['defocus_pct_error'] = (merged['defocus_error_mm'] / merged['defocus_mm_gt']) * 100

    return merged


def compute_metrics(merged: pd.DataFrame) -> Dict:
    """Compute comprehensive error metrics."""

    metrics = {
        'n_samples': len(merged),

        # Blur metrics
        'blur_mae_px': merged['blur_abs_error_px'].mean(),
        'blur_rmse_px': np.sqrt((merged['blur_error_px'] ** 2).mean()),
        'blur_std_px': merged['blur_error_px'].std(),
        'blur_bias_px': merged['blur_error_px'].mean(),
        'blur_mape': merged['blur_pct_error'].abs().mean(),
        'blur_max_error_px': merged['blur_abs_error_px'].max(),

        # Defocus metrics
        'defocus_mae_mm': merged['defocus_abs_error_mm'].mean(),
        'defocus_rmse_mm': np.sqrt((merged['defocus_error_mm'] ** 2).mean()),
        'defocus_mape': merged['defocus_pct_error'].abs().mean(),

        # Correlation
        'blur_correlation': merged['blur_px_gt'].corr(merged['blur_px_pred']),
        'defocus_correlation': merged['defocus_mm_gt'].corr(merged['defocus_mm_pred']),
    }

    # Binned metrics (accuracy at different blur levels, derived from data range)
    blur_edges, blur_labels = _equal_bins(merged['blur_px_gt'].values)
    merged['blur_bin'] = pd.cut(merged['blur_px_gt'], bins=blur_edges, labels=blur_labels,
                                include_lowest=True)

    binned_mae = merged.groupby('blur_bin')['blur_abs_error_px'].mean()
    binned_count = merged.groupby('blur_bin').size()

    metrics['binned_mae'] = binned_mae.to_dict()
    metrics['binned_count'] = binned_count.to_dict()

    return metrics


def compute_per_slice_metrics(merged: pd.DataFrame) -> Dict:
    """Break down errors by camera, defocus magnitude, and droplet size.

    Extracts camera ID from filename patterns like 'sphere0042g_crop.png'
    where the letter before _crop is the camera (g/v/m).

    Returns dict of slice_name -> {slice_value -> {mae, rmse, n, ...}}.
    """
    slices = {}

    # --- By camera ---
    if 'filename' in merged.columns:
        # Try to extract camera letter from filename (e.g. '...g_crop.png' -> 'g')
        cam = merged.index.to_series().str.extract(r'([gvm])(?:_crop)?\.png$', expand=False)
        if cam.notna().sum() > 0:
            merged['camera'] = cam
            cam_groups = {}
            for name, group in merged.groupby('camera'):
                if len(group) < 3:
                    continue
                cam_groups[name] = {
                    'n': len(group),
                    'blur_mae_px': float(group['blur_abs_error_px'].mean()),
                    'defocus_mae_mm': float(group['defocus_abs_error_mm'].mean()),
                    'defocus_rmse_mm': float(np.sqrt((group['defocus_error_mm'] ** 2).mean())),
                }
            if cam_groups:
                slices['camera'] = cam_groups

    # --- By defocus magnitude ---
    if 'defocus_mm_gt' in merged.columns:
        defocus_edges, defocus_labels = _equal_bins(merged['defocus_mm_gt'].values)
        merged['defocus_bin'] = pd.cut(
            merged['defocus_mm_gt'], bins=defocus_edges, labels=defocus_labels,
            include_lowest=True,
        )
        defocus_groups = {}
        for name, group in merged.groupby('defocus_bin', observed=True):
            if len(group) < 3:
                continue
            defocus_groups[name] = {
                'n': len(group),
                'blur_mae_px': float(group['blur_abs_error_px'].mean()),
                'defocus_mae_mm': float(group['defocus_abs_error_mm'].mean()),
                'defocus_rmse_mm': float(np.sqrt((group['defocus_error_mm'] ** 2).mean())),
            }
        if defocus_groups:
            slices['defocus_range_mm'] = defocus_groups

    # --- Near-focus vs far-from-focus ---
    if 'defocus_mm_gt' in merged.columns:
        # Split at median defocus — adapts to whatever range is in the data
        median_z = float(merged['defocus_mm_gt'].median())
        near = merged[merged['defocus_mm_gt'] <= median_z]
        far = merged[merged['defocus_mm_gt'] > median_z]
        focus_groups = {}
        if len(near) >= 3:
            focus_groups[f'near (<={median_z:.1f}mm)'] = {
                'n': len(near),
                'blur_mae_px': float(near['blur_abs_error_px'].mean()),
                'defocus_mae_mm': float(near['defocus_abs_error_mm'].mean()),
            }
        if len(far) >= 3:
            focus_groups[f'far (>{median_z:.1f}mm)'] = {
                'n': len(far),
                'blur_mae_px': float(far['blur_abs_error_px'].mean()),
                'defocus_mae_mm': float(far['defocus_abs_error_mm'].mean()),
            }
        if focus_groups:
            slices['focus_proximity'] = focus_groups

    # --- By droplet size (if diameter available) ---
    if 'diameter_px' in merged.columns:
        valid = merged[merged['diameter_px'] > 0]
        if len(valid) >= 10:
            valid_copy = valid.copy()
            size_edges, size_labels = _equal_bins(valid_copy['diameter_px'].values)
            valid_copy['size_bin'] = pd.cut(
                valid_copy['diameter_px'], bins=size_edges, labels=size_labels,
                include_lowest=True,
            )
            size_groups = {}
            for name, group in valid_copy.groupby('size_bin', observed=True):
                if len(group) < 3:
                    continue
                size_groups[name] = {
                    'n': len(group),
                    'blur_mae_px': float(group['blur_abs_error_px'].mean()),
                    'defocus_mae_mm': float(group['defocus_abs_error_mm'].mean()),
                }
            if size_groups:
                slices['droplet_size'] = size_groups

    return slices


def plot_comparison(merged: pd.DataFrame, metrics: Dict, output_dir: Path, blur_term: str = "CoC"):
    """Create comprehensive comparison plots."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Scatter: Predicted vs Ground Truth
    ax = axes[0, 0]
    ax.scatter(merged['blur_px_gt'], merged['blur_px_pred'], alpha=0.4, s=15, c='blue')

    # Perfect prediction line
    max_val = max(merged['blur_px_gt'].max(), merged['blur_px_pred'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect prediction')

    # Fit line
    z = np.polyfit(merged['blur_px_gt'], merged['blur_px_pred'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, max_val, 100)
    ax.plot(x_line, p(x_line), 'g-', linewidth=1.5,
            label=f'Fit: y={z[0]:.3f}x + {z[1]:.3f}')

    ax.set_xlabel(f'Ground Truth {blur_term} (px)', fontsize=11)
    ax.set_ylabel(f'Predicted {blur_term} (px)', fontsize=11)
    ax.set_title(f'{blur_term}: Predicted vs Ground Truth\n(R² = {metrics["blur_correlation"]**2:.4f})',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)
    ax.set_aspect('equal', adjustable='box')

    # 2. Error histogram
    ax = axes[0, 1]
    ax.hist(merged['blur_error_px'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax.axvline(metrics['blur_bias_px'], color='orange', linestyle='-', linewidth=2,
               label=f'Mean bias: {metrics["blur_bias_px"]:.3f} px')
    ax.set_xlabel(f'{blur_term} Prediction Error (px)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Error Distribution\nMAE: {metrics["blur_mae_px"]:.3f} px, RMSE: {metrics["blur_rmse_px"]:.3f} px',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 3. Error vs Ground Truth blur
    ax = axes[0, 2]
    scatter = ax.scatter(merged['blur_px_gt'], merged['blur_error_px'],
                         alpha=0.4, s=15, c=merged['blur_abs_error_px'], cmap='viridis')
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    plt.colorbar(scatter, ax=ax, label='|Error| (px)')
    ax.set_xlabel(f'Ground Truth {blur_term} (px)', fontsize=11)
    ax.set_ylabel('Prediction Error (px)', fontsize=11)
    ax.set_title('Error vs Blur Level\n(positive = over-prediction)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

    # 4. Binned MAE
    ax = axes[1, 0]
    bin_labels = list(metrics['binned_mae'].keys())
    bin_maes = [metrics['binned_mae'].get(b, 0) for b in bin_labels]
    bin_counts = [metrics['binned_count'].get(b, 0) for b in bin_labels]

    x_pos = np.arange(len(bin_labels))
    bars = ax.bar(x_pos, bin_maes, color='steelblue', alpha=0.7, edgecolor='black')

    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, bin_counts)):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'n={count}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel(f'Ground Truth {blur_term} Range (px)', fontsize=11)
    ax.set_ylabel('Mean Absolute Error (px)', fontsize=11)
    ax.set_title('MAE by Blur Level', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # 5. Percentage error distribution
    ax = axes[1, 1]
    # Clip extreme percentages for visualization
    pct_clipped = merged['blur_pct_error'].clip(-100, 100)
    ax.hist(pct_clipped, bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvline(merged['blur_pct_error'].median(), color='green', linestyle='-', linewidth=2,
               label=f'Median: {merged["blur_pct_error"].median():.1f}%')
    ax.set_xlabel(f'{blur_term} Percentage Error (%)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Percentage Error Distribution\nMAPE: {metrics["blur_mape"]:.1f}%',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 6. Defocus distance comparison
    ax = axes[1, 2]
    ax.scatter(merged['defocus_mm_gt'], merged['defocus_mm_pred'], alpha=0.4, s=15, c='purple')
    max_def = max(merged['defocus_mm_gt'].max(), merged['defocus_mm_pred'].max())
    ax.plot([0, max_def], [0, max_def], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('Ground Truth Defocus (mm)', fontsize=11)
    ax.set_ylabel('Predicted Defocus (mm)', fontsize=11)
    ax.set_title(f'Defocus Distance Comparison\nMAE: {metrics["defocus_mae_mm"]:.4f} mm',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, max_def * 1.05)
    ax.set_ylim(0, max_def * 1.05)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(output_dir / 'blur_prediction_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: blur_prediction_accuracy.png")


def plot_diameter_comparison(merged: pd.DataFrame, output_dir: Path, blur_term: str = "CoC"):
    """Compare ground truth diameter vs measured diameters."""

    if 'diameter_px' not in merged.columns:
        print("Skipping diameter comparison - no ground truth diameter in metadata")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Filter to samples with valid measurements
    valid = merged[merged['diameter_original_px'] > 0].copy()

    if len(valid) == 0:
        print("No valid diameter measurements to compare")
        return

    # 1. GT diameter vs original (blurred) diameter
    ax = axes[0]
    ax.scatter(valid['diameter_px'], valid['diameter_original_px'], alpha=0.4, s=15, c='blue')
    max_d = max(valid['diameter_px'].max(), valid['diameter_original_px'].max())
    ax.plot([0, max_d], [0, max_d], 'r--', linewidth=2, label='1:1 line')
    ax.set_xlabel('Ground Truth Diameter (px)', fontsize=11)
    ax.set_ylabel('Measured Diameter - Blurred (px)', fontsize=11)
    ax.set_title('Original vs GT Diameter\n(blur causes measurement error)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 2. GT diameter vs deblurred diameter (if available)
    ax = axes[1]
    deblur_valid = valid[valid['diameter_deblurred_px'] > 0]

    if len(deblur_valid) > 0:
        ax.scatter(deblur_valid['diameter_px'], deblur_valid['diameter_deblurred_px'],
                   alpha=0.4, s=15, c='green')
        max_d = max(deblur_valid['diameter_px'].max(), deblur_valid['diameter_deblurred_px'].max())
        ax.plot([0, max_d], [0, max_d], 'r--', linewidth=2, label='1:1 line')

        # Calculate improvement
        orig_error = (deblur_valid['diameter_original_px'] - deblur_valid['diameter_px']).abs().mean()
        deblur_error = (deblur_valid['diameter_deblurred_px'] - deblur_valid['diameter_px']).abs().mean()
        improvement = ((orig_error - deblur_error) / orig_error) * 100 if orig_error > 0 else 0

        ax.set_title(f'Deblurred vs GT Diameter\nMAE improved by {improvement:.1f}%',
                     fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No deblurred samples\n(all in-focus?)',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Deblurred vs GT Diameter', fontsize=12, fontweight='bold')

    ax.set_xlabel('Ground Truth Diameter (px)', fontsize=11)
    ax.set_ylabel('Measured Diameter - Deblurred (px)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 3. Diameter error vs blur
    ax = axes[2]
    valid['orig_diameter_error'] = valid['diameter_original_px'] - valid['diameter_px']

    ax.scatter(valid['blur_px_gt'], valid['orig_diameter_error'], alpha=0.4, s=15,
               c='blue', label='Original (blurred)')

    if len(deblur_valid) > 0:
        deblur_valid['deblur_diameter_error'] = deblur_valid['diameter_deblurred_px'] - deblur_valid['diameter_px']
        ax.scatter(deblur_valid['blur_px_gt'], deblur_valid['deblur_diameter_error'],
                   alpha=0.4, s=15, c='green', label='Deblurred')

    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel(f'Ground Truth {blur_term} (px)', fontsize=11)
    ax.set_ylabel('Diameter Measurement Error (px)', fontsize=11)
    ax.set_title('Diameter Error vs Blur Level', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'diameter_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: diameter_accuracy.png")


def generate_report(merged: pd.DataFrame, metrics: Dict, output_dir: Path, blur_term: str = "CoC"):
    """Generate text summary report."""

    lines = [
        "=" * 70,
        "GROUND TRUTH vs MODEL PREDICTION COMPARISON REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "1. DATASET",
        "-" * 40,
        f"Matched samples: {metrics['n_samples']}",
        f"Ground Truth {blur_term} range: [{merged['blur_px_gt'].min():.2f}, {merged['blur_px_gt'].max():.2f}] px",
        f"Predicted {blur_term} range: [{merged['blur_px_pred'].min():.2f}, {merged['blur_px_pred'].max():.2f}] px",
        "",
        f"2. {blur_term.upper()} PREDICTION ACCURACY",
        "-" * 40,
        f"Mean Absolute Error (MAE):     {metrics['blur_mae_px']:.4f} px",
        f"Root Mean Square Error (RMSE): {metrics['blur_rmse_px']:.4f} px",
        f"Mean Bias (systematic error):  {metrics['blur_bias_px']:+.4f} px",
        f"Standard Deviation:            {metrics['blur_std_px']:.4f} px",
        f"Mean Absolute % Error (MAPE):  {metrics['blur_mape']:.2f}%",
        f"Maximum Error:                 {metrics['blur_max_error_px']:.4f} px",
        f"Correlation (R):               {metrics['blur_correlation']:.4f}",
        f"R-squared (R²):                {metrics['blur_correlation']**2:.4f}",
        "",
        "3. DEFOCUS DISTANCE ACCURACY",
        "-" * 40,
        f"MAE:  {metrics['defocus_mae_mm']:.5f} mm",
        f"RMSE: {metrics['defocus_rmse_mm']:.5f} mm",
        f"MAPE: {metrics['defocus_mape']:.2f}%",
        f"Correlation: {metrics['defocus_correlation']:.4f}",
        "",
        "4. ACCURACY BY BLUR LEVEL",
        "-" * 40,
        f"{blur_term} Range (px) | Samples | MAE (px)",
        "-" * 40,
    ]

    for bin_label in metrics['binned_mae'].keys():
        mae = metrics['binned_mae'].get(bin_label, 0)
        count = metrics['binned_count'].get(bin_label, 0)
        if count > 0:
            lines.append(f"  {bin_label:12s} | {count:7d} | {mae:.4f}")

    # Diameter analysis if available
    if 'diameter_px' in merged.columns:
        valid = merged[merged['diameter_original_px'] > 0]
        if len(valid) > 0:
            orig_mae = (valid['diameter_original_px'] - valid['diameter_px']).abs().mean()

            lines.extend([
                "",
                "5. DIAMETER MEASUREMENT ACCURACY",
                "-" * 40,
                f"Original (blurred) diameter MAE: {orig_mae:.2f} px",
            ])

            deblur_valid = valid[valid['diameter_deblurred_px'] > 0]
            if len(deblur_valid) > 0:
                deblur_mae = (deblur_valid['diameter_deblurred_px'] - deblur_valid['diameter_px']).abs().mean()
                improvement = ((orig_mae - deblur_mae) / orig_mae) * 100 if orig_mae > 0 else 0
                lines.extend([
                    f"Deblurred diameter MAE:          {deblur_mae:.2f} px",
                    f"Improvement from deblurring:     {improvement:.1f}%",
                ])

    # Per-slice breakdown
    slice_metrics = compute_per_slice_metrics(merged)
    if slice_metrics:
        section_num = 6
        lines.extend(["", f"{section_num}. PER-SLICE ERROR BREAKDOWN", "-" * 40])

        for slice_name, groups in slice_metrics.items():
            lines.append(f"\n  By {slice_name}:")
            lines.append(f"  {'Slice':<16s} {'n':>6s} {'Blur MAE':>10s} {'Defocus MAE':>12s}")
            lines.append(f"  {'-'*48}")
            for val, m in groups.items():
                lines.append(
                    f"  {str(val):<16s} {m['n']:>6d} "
                    f"{m['blur_mae_px']:>10.3f} {m['defocus_mae_mm']:>12.4f}"
                )

    # Interpretation
    lines.extend([
        "",
        "7. INTERPRETATION",
        "-" * 40,
    ])

    if metrics['blur_mae_px'] < 1.0:
        lines.append(f"[OK] Excellent {blur_term} prediction accuracy (MAE < 1 px)")
    elif metrics['blur_mae_px'] < 2.0:
        lines.append(f"[OK] Good {blur_term} prediction accuracy (MAE < 2 px)")
    elif metrics['blur_mae_px'] < 3.0:
        lines.append(f"[--] Moderate {blur_term} prediction accuracy (MAE < 3 px)")
    else:
        lines.append(f"[!!] Poor {blur_term} prediction accuracy (MAE >= 3 px)")

    if abs(metrics['blur_bias_px']) < 0.5:
        lines.append("[OK] Low systematic bias (|bias| < 0.5 px)")
    elif metrics['blur_bias_px'] > 0:
        lines.append(f"[--] Model tends to OVER-predict {blur_term} by {metrics['blur_bias_px']:.2f} px")
    else:
        lines.append(f"[--] Model tends to UNDER-predict {blur_term} by {abs(metrics['blur_bias_px']):.2f} px")

    if metrics['blur_correlation'] > 0.95:
        lines.append("[OK] Excellent correlation with ground truth (R > 0.95)")
    elif metrics['blur_correlation'] > 0.90:
        lines.append("[OK] Good correlation with ground truth (R > 0.90)")
    else:
        lines.append(f"[--] Moderate correlation with ground truth (R = {metrics['blur_correlation']:.3f})")

    lines.extend([
        "",
        "=" * 70,
        "END OF REPORT",
        "=" * 70,
    ])

    report_text = "\n".join(lines)

    report_path = output_dir / 'comparison_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\nSaved: comparison_report.txt")
    print("\n" + report_text)

    return report_text


def save_merged_csv(merged: pd.DataFrame, output_dir: Path):
    """Save merged data with errors to CSV."""

    output_cols = [
        'blur_px_gt', 'blur_px_pred', 'blur_error_px', 'blur_abs_error_px', 'blur_pct_error',
        'defocus_mm_gt', 'defocus_mm_pred', 'defocus_error_mm', 'defocus_pct_error'
    ]

    if 'diameter_px' in merged.columns:
        output_cols.extend(['diameter_px', 'diameter_original_px', 'diameter_deblurred_px'])

    # Only include columns that exist
    output_cols = [c for c in output_cols if c in merged.columns]

    csv_path = output_dir / 'merged_comparison.csv'
    merged[output_cols].to_csv(csv_path)
    print(f"Saved: merged_comparison.csv")


def auto_detect_paths() -> tuple:
    """Auto-detect metadata.csv and latest inference results."""
    script_dir = Path(__file__).parent

    # Find metadata.csv in synthetic_data
    metadata_candidates = [
        script_dir / 'training_output' / 'synthetic_data' / 'metadata.csv',
        script_dir / 'synthetic_data' / 'metadata.csv',
    ]
    # Also search for any metadata.csv
    metadata_candidates.extend(script_dir.glob('**/synthetic_data/metadata.csv'))

    metadata_path = None
    for candidate in metadata_candidates:
        if candidate.exists():
            metadata_path = candidate
            break

    # Find latest inference results on blur folder
    inference_path = None
    inference_dir = script_dir / 'inference_results'
    if inference_dir.exists():
        # Get all inference_* folders sorted by name (timestamp)
        inference_folders = sorted(
            [d for d in inference_dir.iterdir() if d.is_dir() and d.name.startswith('inference_')],
            reverse=True  # Most recent first
        )
        for folder in inference_folders:
            # Look for blur/blur_results.csv (synthetic data inference), fall back to old name
            blur_csv = folder / 'blur' / 'blur_results.csv'
            if not blur_csv.exists():
                blur_csv = folder / 'blur' / 'coc_results.csv'
            if blur_csv.exists():
                inference_path = blur_csv
                break

    return metadata_path, inference_path


def main():
    parser = argparse.ArgumentParser(
        description='Compare ground truth blur values from metadata.csv vs model predictions'
    )
    parser.add_argument('--metadata', '-m', type=str, default=None,
                        help='Path to metadata.csv (auto-detected if not provided)')
    parser.add_argument('--inference', '-i', type=str, default=None,
                        help='Path to blur_results.csv (auto-detected if not provided)')
    parser.add_argument('--output', '-o', type=str, default='comparison_results',
                        help='Output directory for comparison results')

    args = parser.parse_args()

    # Auto-detect paths if not provided
    if args.metadata is None or args.inference is None:
        auto_metadata, auto_inference = auto_detect_paths()

        if args.metadata is None:
            if auto_metadata is None:
                print("Error: Could not auto-detect metadata.csv")
                print("Please provide --metadata path or ensure synthetic_data/metadata.csv exists")
                return
            metadata_path = auto_metadata
            print(f"Auto-detected metadata: {metadata_path}")
        else:
            metadata_path = Path(args.metadata)

        if args.inference is None:
            if auto_inference is None:
                print("Error: Could not auto-detect inference results")
                print("Please provide --inference path or run inference on synthetic blur folder first")
                return
            inference_path = auto_inference
            print(f"Auto-detected inference: {inference_path}")
        else:
            inference_path = Path(args.inference)
    else:
        metadata_path = Path(args.metadata)
        inference_path = Path(args.inference)

    output_dir = Path(args.output)

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / f"comparison_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("GROUND TRUTH vs MODEL PREDICTION COMPARISON")
    print(f"{'='*60}")
    print(f"Ground truth (metadata.csv): {metadata_path}")
    print(f"Predictions (blur_results.csv): {inference_path}")
    print(f"Output directory: {output_dir}")
    print()

    # Load data
    gt_df = load_ground_truth(metadata_path)
    pred_df = load_inference_results(inference_path)

    # Merge and compute errors
    merged = merge_datasets(gt_df, pred_df)

    if len(merged) == 0:
        print("\nError: No matching samples found between ground truth and predictions")
        return

    # Compute metrics
    metrics = compute_metrics(merged)

    # Detect display term from ground truth data
    bt = _blur_term(gt_df)

    # Generate outputs
    print("\nGenerating comparison plots...")
    plot_comparison(merged, metrics, output_dir, blur_term=bt)
    plot_diameter_comparison(merged, output_dir, blur_term=bt)

    print("\nSaving data...")
    save_merged_csv(merged, output_dir)

    print("\nGenerating report...")
    generate_report(merged, metrics, output_dir, blur_term=bt)

    print(f"\n{'='*60}")
    print(f"Comparison complete! Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
