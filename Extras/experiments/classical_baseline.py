"""
Classical Focus Metric Baseline Comparison

Runs all 6 classical focus metrics through the calibration chain and compares
their blur estimation accuracy against the CNN. This answers: does the CNN
actually outperform simpler methods?

Each classical metric produces a single scalar per crop (e.g. Laplacian
variance). To convert this to a blur estimate in pixels, we fit a linear
mapping from the metric score to the ground truth blur using half the data,
then evaluate on the other half. This is the fairest comparison — the
classical methods get the same calibration opportunity as the CNN.

Usage:
    python classical_baseline.py --data path/to/synthetic_data
    python classical_baseline.py --data path/to/synthetic_data \
                                 --inference path/to/blur_results.csv

If --inference is provided, the CNN results are included in the comparison.
Otherwise, only classical metrics are compared against each other.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add preprocessing to path for focus_metrics
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_REPO_ROOT / "Preprocessing", _REPO_ROOT / "Training"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from focus_metrics import compute_all_focus_metrics  # noqa: E402


# ── Data loading ──────────────────────────────────────────────────────────

def load_dataset(data_dir: Path) -> Tuple[pd.DataFrame, Path]:
    """Load metadata and return (dataframe, blur_image_dir)."""
    metadata_path = data_dir / "metadata.csv"
    blur_dir = data_dir / "blur"

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found in {data_dir}")
    if not blur_dir.is_dir():
        raise FileNotFoundError(f"blur/ directory not found in {data_dir}")

    df = pd.read_csv(metadata_path)
    df["index"] = df["index"].astype(str).str.zfill(6)
    df = df.set_index("index")

    blur_col = "sigma_px" if "sigma_px" in df.columns else "coc_px"
    print(f"Loaded {len(df)} samples from {data_dir.name}")
    print(f"  Blur column: {blur_col}, range: [{df[blur_col].min():.2f}, {df[blur_col].max():.2f}] px")
    print(f"  Defocus range: [{df['defocus_mm'].min():.3f}, {df['defocus_mm'].max():.3f}] mm")

    return df, blur_dir


def compute_classical_scores(
    df: pd.DataFrame,
    blur_dir: Path,
) -> pd.DataFrame:
    """Compute all 6 classical focus metrics for every crop."""
    metric_names = [
        "laplacian_var", "tenengrad", "tenengrad_var",
        "brenner", "norm_laplacian", "energy_gradient",
    ]
    rows = []
    total = len(df)

    for i, (idx, _row) in enumerate(df.iterrows()):
        img_path = blur_dir / f"{idx}.png"
        if not img_path.exists():
            continue

        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        scores = compute_all_focus_metrics(gray)
        scores["index"] = idx
        rows.append(scores)

        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"  Computed metrics: {i + 1}/{total}")

    scores_df = pd.DataFrame(rows).set_index("index")
    print(f"  Got scores for {len(scores_df)} crops")
    return scores_df


# ── Calibration: metric score → blur px ──────────────────────────────────

def calibrate_metric(
    scores: np.ndarray,
    blur_gt: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Fit a linear mapping: blur_pred = a * log(score + 1) + b.

    We use log(score + 1) because focus metrics scale roughly exponentially
    with blur (sharp images have very high scores, blurry images very low).

    Returns (a, b, r_squared) from the fit.
    """
    log_scores = np.log(scores + 1)

    # Remove inf/nan
    valid = np.isfinite(log_scores) & np.isfinite(blur_gt)
    log_scores = log_scores[valid]
    blur_gt = blur_gt[valid]

    if len(log_scores) < 10:
        return 0.0, 0.0, 0.0

    # Fit linear: blur = a * log_score + b
    A = np.vstack([log_scores, np.ones(len(log_scores))]).T
    result = np.linalg.lstsq(A, blur_gt, rcond=None)
    a, b = result[0]

    # R²
    pred = a * log_scores + b
    ss_res = np.sum((blur_gt - pred) ** 2)
    ss_tot = np.sum((blur_gt - np.mean(blur_gt)) ** 2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return a, b, r_sq


def predict_blur(scores: np.ndarray, a: float, b: float) -> np.ndarray:
    """Apply calibrated mapping to predict blur from metric scores."""
    return a * np.log(scores + 1) + b


# ── Evaluation ────────────────────────────────────────────────────────────

def evaluate_method(
    blur_pred: np.ndarray,
    blur_gt: np.ndarray,
) -> Dict[str, float]:
    """Compute error metrics for a blur prediction method."""
    errors = blur_pred - blur_gt
    abs_errors = np.abs(errors)

    # Avoid div-by-zero for MAPE
    nonzero = blur_gt > 0.1
    mape = np.mean(np.abs(errors[nonzero]) / blur_gt[nonzero]) * 100 if nonzero.sum() > 5 else float("nan")

    corr = np.corrcoef(blur_gt, blur_pred)[0, 1] if len(blur_gt) > 2 else 0.0

    return {
        "mae_px": float(np.mean(abs_errors)),
        "rmse_px": float(np.sqrt(np.mean(errors ** 2))),
        "bias_px": float(np.mean(errors)),
        "max_error_px": float(np.max(abs_errors)),
        "mape": float(mape),
        "r_squared": float(corr ** 2),
    }


def run_comparison(
    df: pd.DataFrame,
    scores_df: pd.DataFrame,
    cnn_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Split-half calibration and evaluation for all methods.

    Splits data 50/50: calibrate on first half, evaluate on second half.
    """
    blur_col = "sigma_px" if "sigma_px" in df.columns else "coc_px"

    # Align indices
    common = df.index.intersection(scores_df.index)
    if cnn_df is not None:
        common = common.intersection(cnn_df.index)
    df = df.loc[common]
    scores_df = scores_df.loc[common]

    # Shuffle and split
    np.random.seed(42)
    indices = np.random.permutation(len(df))
    split = len(indices) // 2
    cal_idx = indices[:split]
    test_idx = indices[split:]

    blur_gt_all = df[blur_col].values
    metric_names = scores_df.columns.tolist()

    results = {}
    all_preds = {}

    print(f"\nCalibrating on {len(cal_idx)} samples, testing on {len(test_idx)} samples")
    print(f"{'Method':<20s} {'MAE (px)':>10s} {'RMSE (px)':>10s} {'R²':>8s} {'MAPE':>8s}")
    print("-" * 60)

    for metric in metric_names:
        scores_all = scores_df[metric].values

        # Calibrate on first half
        a, b, r_sq_cal = calibrate_metric(scores_all[cal_idx], blur_gt_all[cal_idx])

        # Predict on test half
        pred_test = predict_blur(scores_all[test_idx], a, b)
        gt_test = blur_gt_all[test_idx]

        metrics = evaluate_method(pred_test, gt_test)
        metrics["cal_r_squared"] = r_sq_cal
        metrics["cal_a"] = a
        metrics["cal_b"] = b
        results[metric] = metrics
        all_preds[metric] = pred_test

        print(f"{metric:<20s} {metrics['mae_px']:>10.3f} {metrics['rmse_px']:>10.3f} "
              f"{metrics['r_squared']:>8.4f} {metrics['mape']:>7.1f}%")

    # CNN results (if provided)
    if cnn_df is not None:
        cnn_df = cnn_df.loc[common]
        cnn_blur_col = "sigma_px" if "sigma_px" in cnn_df.columns else "coc_px"
        cnn_pred_test = cnn_df[cnn_blur_col].values[test_idx]
        gt_test = blur_gt_all[test_idx]

        cnn_metrics = evaluate_method(cnn_pred_test, gt_test)
        cnn_metrics["cal_r_squared"] = float("nan")
        cnn_metrics["cal_a"] = float("nan")
        cnn_metrics["cal_b"] = float("nan")
        results["CNN"] = cnn_metrics
        all_preds["CNN"] = cnn_pred_test

        print(f"{'CNN':<20s} {cnn_metrics['mae_px']:>10.3f} {cnn_metrics['rmse_px']:>10.3f} "
              f"{cnn_metrics['r_squared']:>8.4f} {cnn_metrics['mape']:>7.1f}%")

    print("-" * 60)

    # Build summary DataFrame
    summary = pd.DataFrame(results).T
    summary.index.name = "method"

    return summary, all_preds, blur_gt_all[test_idx]


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_comparison(
    summary: pd.DataFrame,
    all_preds: Dict[str, np.ndarray],
    blur_gt_test: np.ndarray,
    output_dir: Path,
):
    """Generate comparison plots."""
    methods = list(all_preds.keys())
    n_methods = len(methods)

    # 1. Bar chart: MAE comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    colours = ["steelblue"] * n_methods
    if "CNN" in methods:
        colours[methods.index("CNN")] = "firebrick"

    bars = ax.bar(range(n_methods), [summary.loc[m, "mae_px"] for m in methods],
                  color=colours, edgecolor="black", alpha=0.8)
    ax.set_xticks(range(n_methods))
    ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("MAE (px)")
    ax.set_title("Blur Estimation: Mean Absolute Error by Method")
    ax.grid(axis="y", alpha=0.3)

    for bar, m in zip(bars, methods):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{summary.loc[m, 'mae_px']:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "mae_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Scatter: predicted vs ground truth for each method
    cols = min(4, n_methods)
    rows = (n_methods + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows), squeeze=False)

    for i, method in enumerate(methods):
        ax = axes[i // cols][i % cols]
        pred = all_preds[method]
        ax.scatter(blur_gt_test, pred, s=8, alpha=0.3,
                   c="firebrick" if method == "CNN" else "steelblue")
        max_val = max(blur_gt_test.max(), pred.max()) * 1.05
        ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, alpha=0.5)
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("GT blur (px)", fontsize=9)
        ax.set_ylabel("Predicted (px)", fontsize=9)
        r2 = summary.loc[method, "r_squared"]
        mae = summary.loc[method, "mae_px"]
        ax.set_title(f"{method}\nR²={r2:.3f}  MAE={mae:.2f}", fontsize=10)
        ax.grid(alpha=0.3)

    # Hide empty subplots
    for j in range(n_methods, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle("Predicted vs Ground Truth Blur", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Summary table as image
    fig, ax = plt.subplots(figsize=(10, 0.4 * (n_methods + 2)))
    ax.axis("off")
    cols_display = ["mae_px", "rmse_px", "r_squared", "bias_px", "mape"]
    col_labels = ["MAE (px)", "RMSE (px)", "R²", "Bias (px)", "MAPE (%)"]
    cell_text = []
    for m in methods:
        row = []
        for c in cols_display:
            v = summary.loc[m, c]
            if c == "mape":
                row.append(f"{v:.1f}")
            elif c == "r_squared":
                row.append(f"{v:.4f}")
            else:
                row.append(f"{v:.3f}")
        cell_text.append(row)

    table = ax.table(
        cellText=cell_text, rowLabels=methods, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Highlight CNN row
    if "CNN" in methods:
        cnn_idx = methods.index("CNN") + 1  # +1 for header row
        for j in range(len(col_labels)):
            table[cnn_idx, j].set_facecolor("#ffe0e0")

    # Highlight best MAE
    best_idx = summary["mae_px"].idxmin()
    best_row = methods.index(best_idx) + 1
    table[best_row, 0].set_facecolor("#d0ffd0")

    fig.suptitle("Classical Baselines vs CNN — Summary", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "summary_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlots saved to {output_dir}")


# ── Report ────────────────────────────────────────────────────────────────

def write_report(
    summary: pd.DataFrame,
    output_dir: Path,
    has_cnn: bool,
):
    """Write text summary report."""
    methods = summary.index.tolist()
    best = summary["mae_px"].idxmin()
    worst = summary["mae_px"].idxmax()

    lines = [
        "=" * 65,
        "CLASSICAL BASELINE COMPARISON REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 65,
        "",
        "Split-half evaluation: calibrated on 50%, tested on 50%",
        f"Methods compared: {len(methods)}",
        "",
        f"{'Method':<20s} {'MAE':>8s} {'RMSE':>8s} {'R²':>8s} {'Bias':>8s} {'MAPE':>7s}",
        "-" * 65,
    ]

    for m in methods:
        r = summary.loc[m]
        marker = " <-- BEST" if m == best else ""
        lines.append(
            f"{m:<20s} {r['mae_px']:>8.3f} {r['rmse_px']:>8.3f} "
            f"{r['r_squared']:>8.4f} {r['bias_px']:>+8.3f} {r['mape']:>6.1f}%{marker}"
        )

    lines.append("-" * 65)
    lines.append("")
    lines.append(f"Best method: {best} (MAE = {summary.loc[best, 'mae_px']:.3f} px)")

    if has_cnn and "CNN" in methods:
        cnn_mae = summary.loc["CNN", "mae_px"]
        classical_best = summary.drop("CNN")["mae_px"].idxmin()
        classical_mae = summary.loc[classical_best, "mae_px"]
        if cnn_mae < classical_mae:
            improvement = (1 - cnn_mae / classical_mae) * 100
            lines.append(f"CNN outperforms best classical ({classical_best}) by {improvement:.1f}%")
        else:
            lines.append(f"Best classical ({classical_best}) matches or beats CNN")

    lines.extend(["", "=" * 65])

    report = "\n".join(lines)
    (output_dir / "baseline_report.txt").write_text(report, encoding="utf-8")
    print(report)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare classical focus metrics against CNN for blur estimation"
    )
    parser.add_argument("--data", "-d", type=str, required=True,
                        help="Path to synthetic data directory (contains metadata.csv + blur/)")
    parser.add_argument("--inference", "-i", type=str, default=None,
                        help="Path to CNN blur_results.csv (optional, for CNN comparison)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory (default: data_dir/classical_baseline_<timestamp>)")
    args = parser.parse_args()

    data_dir = Path(args.data)

    # Output dir
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = data_dir / f"classical_baseline_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df, blur_dir = load_dataset(data_dir)

    # Compute classical metrics
    print("\nComputing classical focus metrics on all crops...")
    scores_df = compute_classical_scores(df, blur_dir)

    # Load CNN results if provided
    cnn_df = None
    if args.inference:
        inference_path = Path(args.inference)
        if inference_path.exists():
            cnn_df = pd.read_csv(inference_path)
            cnn_df["index"] = cnn_df["filename"].str.replace(".png", "", regex=False).str.zfill(6)
            cnn_df = cnn_df.set_index("index")
            print(f"Loaded CNN predictions: {len(cnn_df)} samples")
        else:
            print(f"Warning: CNN results not found at {inference_path}")

    # Run comparison
    summary, all_preds, blur_gt_test = run_comparison(df, scores_df, cnn_df)

    # Save outputs
    summary.to_csv(output_dir / "baseline_summary.csv")
    plot_comparison(summary, all_preds, blur_gt_test, output_dir)
    write_report(summary, output_dir, has_cnn=cnn_df is not None)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
