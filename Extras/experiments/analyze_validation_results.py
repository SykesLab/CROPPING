"""
Analyze Validation Results - Post-Processing Script

Analyzes dual_test_results.csv and generates worst-case visualizations
based on diameter error percentage without re-running validation.

Usage:
    python analyze_validation_results.py --results test_output/dual_test_results.csv --data data/synthetic_test --model checkpoints/best_model.pth --output test_output
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import argparse
from typing import Dict, List
from tqdm import tqdm

from test_model import ModelTester, DiameterMeasurer


def analyze_results(
    results_csv: Path,
    data_dir: Path,
    model_path: Path,
    output_dir: Path,
    num_worst: int = 10,
    device: str = 'auto'
):
    """
    Analyze validation results and create worst-case visualizations.

    Args:
        results_csv: Path to dual_test_results.csv
        data_dir: Path to test data directory
        model_path: Path to model checkpoint (for loading samples)
        output_dir: Output directory for visualizations
        num_worst: Number of worst cases to visualize
        device: Device to use ('cuda', 'cpu', or 'auto')
    """

    # Load results CSV
    print(f"Loading results from: {results_csv}")
    df = pd.read_csv(results_csv)

    print(f"Total samples: {len(df)}")
    print(f"\nDiameter Error Statistics:")
    print(f"  Mean Error: {df['diameter_error_px'].mean():.2f} px ({df['diameter_error_pct'].mean():.2f}%)")
    print(f"  Median Error: {df['diameter_error_px'].median():.2f} px ({df['diameter_error_pct'].median():.2f}%)")
    print(f"  Max Error: {df['diameter_error_px'].max():.2f} px ({df['diameter_error_pct'].max():.2f}%)")
    print(f"  Std Dev: {df['diameter_error_px'].std():.2f} px")

    # Sort by diameter error percentage
    df_sorted = df.sort_values('diameter_error_pct', ascending=False)
    worst_samples = df_sorted.head(num_worst)

    print(f"\nTop {num_worst} Worst Diameter Errors:")
    for idx, row in worst_samples.iterrows():
        print(f"  {row['sample']}: {row['diameter_error_pct']:.2f}% ({row['diameter_error_px']:.2f} px) - PSNR: {row['psnr_db']:.1f} dB")

    # Create ModelTester to load samples
    print(f"\nLoading model from: {model_path}")
    tester = ModelTester(model_path=str(model_path), device=device)

    # Create output directory
    worst_dir = output_dir / 'worst_cases_analysis'
    worst_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nGenerating visualizations for {len(worst_samples)} worst cases...")

    # Process each worst sample
    for idx, row in tqdm(worst_samples.iterrows(), total=len(worst_samples), desc="Creating visualizations"):
        sample_name = f"{int(row['sample']):06d}"
        sample_path = data_dir / 'blur' / f'{sample_name}.png'

        if not sample_path.exists():
            print(f"Warning: Sample not found: {sample_path}")
            continue

        # Load sample data
        blur, blur_gt, sharp_gt, blur_value_gt = tester.load_sample(sample_path, data_dir)

        # Forward pass to get predictions
        with torch.no_grad():
            pred_sharp, pred_blur_map = tester.model(blur)

        # Create visualization
        filename = f"{sample_name}_DiamErr{row['diameter_error_pct']:.2f}pct_{row['diameter_error_px']:.2f}px"
        tester._create_comparison_image(
            blur, pred_sharp, sharp_gt,
            pred_blur_map, blur_gt,
            filename,
            worst_dir
        )

    print(f"\n✓ Saved visualizations to: {worst_dir}")

    # Generate summary plot
    _create_summary_plots(df, output_dir)

    print("\n✓ Analysis complete!")


def _create_summary_plots(df: pd.DataFrame, output_dir: Path):
    """Create summary analysis plots."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Diameter error distribution
    axes[0, 0].hist(df['diameter_error_px'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].axvline(df['diameter_error_px'].mean(), color='red', linestyle='--',
                       label=f"Mean: {df['diameter_error_px'].mean():.2f} px")
    axes[0, 0].set_xlabel('Diameter Error (px)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Diameter Error Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 2. Diameter error percentage distribution
    axes[0, 1].hist(df['diameter_error_pct'], bins=30, edgecolor='black', alpha=0.7, color='coral')
    axes[0, 1].axvline(df['diameter_error_pct'].mean(), color='red', linestyle='--',
                       label=f"Mean: {df['diameter_error_pct'].mean():.2f}%")
    axes[0, 1].set_xlabel('Diameter Error (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Diameter Error Percentage Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # 3. PSNR vs Diameter Error
    axes[1, 0].scatter(df['psnr_db'], df['diameter_error_pct'], alpha=0.5, s=20, color='green')
    axes[1, 0].set_xlabel('PSNR (dB)')
    axes[1, 0].set_ylabel('Diameter Error (%)')
    axes[1, 0].set_title('PSNR vs Diameter Error')
    axes[1, 0].grid(alpha=0.3)

    # 4. Blur Error vs Diameter Error
    blur_err_col = 'sigma_error_px' if 'sigma_error_px' in df.columns else 'coc_error_px'
    blur_label = 'σ' if 'sigma_error_px' in df.columns else 'CoC'
    axes[1, 1].scatter(df[blur_err_col], df['diameter_error_pct'], alpha=0.5, s=20, color='purple')
    axes[1, 1].set_xlabel(f'{blur_label} Error (px)')
    axes[1, 1].set_ylabel('Diameter Error (%)')
    axes[1, 1].set_title(f'{blur_label} Error vs Diameter Error')
    axes[1, 1].grid(alpha=0.3)

    plt.suptitle('Diameter Error Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = output_dir / 'diameter_error_analysis.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved analysis plots to: {plot_path}")


def main():
    print("="*60)
    print("VALIDATION RESULTS ANALYZER")
    print("="*60)
    print("\nThis script analyzes existing validation results and generates")
    print("worst-case visualizations based on diameter error.\n")

    # Get inputs interactively
    results_csv = input("Path to dual_test_results.csv (or press Enter for default 'test_results/dual_test_results.csv'): ").strip()
    if not results_csv:
        results_csv = "test_results/dual_test_results.csv"
    results_csv = Path(results_csv)

    data_dir = input("Path to test data directory (or press Enter for default 'training_output/synthetic_data'): ").strip()
    if not data_dir:
        data_dir = "training_output/synthetic_data"
    data_dir = Path(data_dir)

    model_path = input("Path to model checkpoint (or press Enter for default 'training_output/checkpoints/best_model.pth'): ").strip()
    if not model_path:
        model_path = "training_output/checkpoints/best_model.pth"
    model_path = Path(model_path)

    output_dir = input("Output directory (or press Enter for default 'test_results'): ").strip()
    if not output_dir:
        output_dir = "test_results"
    output_dir = Path(output_dir)

    num_worst_input = input("Number of worst cases to visualize (or press Enter for default '10'): ").strip()
    if not num_worst_input:
        num_worst = 10
    else:
        num_worst = int(num_worst_input)

    device_input = input("Device (cuda/cpu/auto, or press Enter for default 'auto'): ").strip()
    if not device_input:
        device = 'auto'
    else:
        device = device_input

    print("\n" + "="*60)
    print("Configuration:")
    print(f"  Results CSV: {results_csv}")
    print(f"  Data Dir: {data_dir}")
    print(f"  Model: {model_path}")
    print(f"  Output Dir: {output_dir}")
    print(f"  Worst Cases: {num_worst}")
    print(f"  Device: {device}")
    print("="*60 + "\n")

    # Validate inputs
    if not results_csv.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_csv}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis 
    analyze_results(
        results_csv=results_csv,
        data_dir=data_dir,
        model_path=model_path,
        output_dir=output_dir,
        num_worst=num_worst,
        device=device
    )


if __name__ == "__main__":
    main()
