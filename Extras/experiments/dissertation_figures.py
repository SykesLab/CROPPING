"""
Dissertation Figure Generators

Standalone script to generate publication-quality figures for the dissertation.
Run after synthetic data generation and/or after training.

Usage:
    python dissertation_figures.py --data-dir training_output/synthetic_data
    python dissertation_figures.py --all --data-dir training_output/synthetic_data
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import pandas as pd
import argparse


def generate_sample_grid(data_dir: str, output_path: str = None, n_per_row: int = 5):
    """
    Generate a grid showing training samples at evenly spaced blur levels.

    Picks one sample near each target σ and displays the blurred image
    with the σ value annotated. Shows what the network sees during training.

    Goes in Section 3.6/3.7 of the dissertation.
    """
    data_dir = Path(data_dir)
    metadata_path = data_dir / 'metadata.csv'
    blur_dir = data_dir / 'blur'

    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found")
        return

    df = pd.read_csv(metadata_path, dtype={'index': str})
    if 'index' not in df.columns:
        df = df.reset_index()

    col = 'sigma_px' if 'sigma_px' in df.columns else 'coc_px'
    blur_term = 'σ' if 'sigma_px' in df.columns else 'CoC'

    sigma_min = df[col].abs().min()
    sigma_max = df[col].abs().max()

    # Pick evenly spaced target sigmas
    n_targets = n_per_row * 3  # 3 rows
    targets = np.linspace(sigma_min, sigma_max, n_targets)

    fig, axes = plt.subplots(3, n_per_row, figsize=(n_per_row * 2.5, 3 * 2.5))

    for i, target_sigma in enumerate(targets):
        row = i // n_per_row
        col_idx = i % n_per_row

        # Find closest sample
        idx = (df[col].abs() - target_sigma).abs().idxmin()
        sample = df.loc[idx]
        stem = str(sample['index']).zfill(6) if isinstance(sample['index'], (int, float)) else sample['index']
        actual_sigma = abs(sample[col])

        # Load image
        img_path = blur_dir / f'{stem}.png'
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        ax = axes[row, col_idx]
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'{blur_term} = {actual_sigma:.1f} px', fontsize=9, fontweight='bold')
        ax.axis('off')

    fig.suptitle(f'Synthetic Training Samples at Increasing Blur ({blur_term})',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path is None:
        output_path = data_dir.parent / 'sample_grid.png'

    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved sample grid: {output_path}")


def generate_loss_motivation(output_path: str = None, max_blur: float = 13.72, eps: float = 0.01):
    """
    Generate a figure showing why log-space MSE is better than linear MSE
    for blur estimation across a wide range.

    Shows that linear MSE gives equal weight to absolute errors regardless
    of blur magnitude, while log-space MSE penalises relative error uniformly.

    Goes in Section 3.9 of the dissertation.
    """
    sigma_range = np.linspace(0.9, max_blur, 200)

    # For a fixed 1 px absolute error at each sigma level
    abs_error = 1.0  # px

    # Linear MSE: loss = (pred - target)^2 = 1.0 for all sigma
    linear_loss = np.ones_like(sigma_range) * abs_error**2

    # Log-space MSE: loss = (log(sigma + eps) - log(sigma + abs_error + eps))^2
    log_loss = (np.log(sigma_range + eps) - np.log(sigma_range + abs_error + eps))**2

    # Also show relative error for context
    relative_error = abs_error / sigma_range * 100  # percentage

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: Linear MSE loss for 1px error
    ax = axes[0]
    ax.plot(sigma_range, linear_loss, 'b-', linewidth=2)
    ax.set_xlabel('True σ (px)', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Linear MSE: 1 px Error', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.5)
    ax.grid(True, alpha=0.3)
    ax.annotate('Same loss regardless\nof blur magnitude',
                xy=(7, 1.0), fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    # Panel 2: Log-space MSE loss for 1px error
    ax = axes[1]
    ax.plot(sigma_range, log_loss, 'r-', linewidth=2)
    ax.set_xlabel('True σ (px)', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Log-space MSE: 1 px Error', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.annotate('Higher loss at low σ\n(1 px error matters more)',
                xy=(3, log_loss[20] * 0.8), fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    # Panel 3: Physical interpretation — relative error
    ax = axes[2]
    ax.plot(sigma_range, relative_error, 'g-', linewidth=2)
    ax.set_xlabel('True σ (px)', fontsize=11)
    ax.set_ylabel('Relative Error (%)', fontsize=11)
    ax.set_title('1 px Error: Physical Significance', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.annotate(f'At σ=1: {100:.0f}% error\nAt σ=13: {100/13:.0f}% error',
                xy=(5, 60), fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('Why Log-space Loss? — Equal Absolute Error Has Unequal Physical Impact',
                 fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()

    if output_path is None:
        output_path = 'loss_motivation.png'

    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved loss motivation figure: {output_path}")


def generate_data_distribution(data_dir: str, output_path: str = None):
    """
    Generate a figure showing the training data distribution.

    Shows histogram of σ values and the uniform coverage across the blur range.
    Useful for Section 3.7 (Data Generation).
    """
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / 'metadata.csv')
    col = 'sigma_px' if 'sigma_px' in df.columns else 'coc_px'
    blur_term = 'σ' if 'sigma_px' in df.columns else 'CoC'
    vals = df[col].abs()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: Histogram
    ax = axes[0]
    ax.hist(vals, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_xlabel(f'{blur_term} at model scale (px)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Training Data: {blur_term} Distribution (n={len(vals):,})', fontsize=12, fontweight='bold')
    ax.axvline(vals.min(), color='red', linestyle='--', alpha=0.7, label=f'Min: {vals.min():.2f} px')
    ax.axvline(vals.max(), color='red', linestyle='--', alpha=0.7, label=f'Max: {vals.max():.2f} px')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: 4-bin bar chart
    ax = axes[1]
    min_v, max_v = vals.min(), vals.max()
    bin_size = (max_v - min_v) / 4
    bin_labels = []
    bin_counts = []
    for i in range(4):
        low = min_v + i * bin_size
        high = min_v + (i + 1) * bin_size
        count = ((vals >= low) & (vals < high)).sum() if i < 3 else ((vals >= low) & (vals <= high)).sum()
        bin_labels.append(f'{low:.1f}-{high:.1f}')
        bin_counts.append(count)

    bars = ax.bar(bin_labels, bin_counts, color=['#2196F3', '#4CAF50', '#FF9800', '#F44336'],
                  edgecolor='white', alpha=0.85)
    for bar, count in zip(bars, bin_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{count:,}\n({count/len(vals)*100:.0f}%)', ha='center', fontsize=9)
    ax.set_xlabel(f'{blur_term} Bin (px)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Samples per Blur Bin', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path is None:
        output_path = data_dir.parent / 'data_distribution.png'

    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved data distribution: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dissertation figures')
    parser.add_argument('--data-dir', type=str, default='training_output/synthetic_data',
                       help='Path to synthetic data directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for figures (default: training_output/)')
    parser.add_argument('--all', action='store_true', help='Generate all figures')
    parser.add_argument('--sample-grid', action='store_true', help='Generate sample grid')
    parser.add_argument('--loss-motivation', action='store_true', help='Generate loss motivation figure')
    parser.add_argument('--data-distribution', action='store_true', help='Generate data distribution figure')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir) if args.output_dir else data_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all or args.sample_grid:
        generate_sample_grid(str(data_dir), str(out_dir / 'sample_grid.png'))

    if args.all or args.loss_motivation:
        generate_loss_motivation(str(out_dir / 'loss_motivation.png'))

    if args.all or args.data_distribution:
        generate_data_distribution(str(data_dir), str(out_dir / 'data_distribution.png'))

    if not (args.all or args.sample_grid or args.loss_motivation or args.data_distribution):
        print("No figures requested. Use --all or specify individual figures.")
        print("Available: --sample-grid, --loss-motivation, --data-distribution")
