"""Three-way comparison on synthetic dataset: applied sigma (truth) vs
ERF-measured (already in metadata.csv) vs model prediction.

If all three agree -> ERF is a faithful measurement of what the model
sees, validating ERF as the calibration measurement tool.

If model agrees with truth but ERF disagrees -> ERF measurement has bias.
If model disagrees with truth but ERF agrees -> model has issues even on
its own training distribution.
If all three disagree wildly -> something deeper is broken.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _module in ("Calibration", "Training", ""):
    _p = str(_REPO_ROOT / _module) if _module else str(_REPO_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SYNTH_DIR = (
    _REPO_ROOT / "Training" / "training_output" / "datasets"
    / "20260423_200211_newpreprocessingallcams"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "flatten_alignment"
N_SAMPLES = 200  # how many synthetic samples to evaluate


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading synthetic metadata...")
    meta = pd.read_csv(SYNTH_DIR / "metadata.csv")
    print(f"  Total samples: {len(meta)}")
    print(f"  Columns of interest: sigma_applied_px, sigma_measured_erf, "
          f"erf_r_squared, diameter_model_px")

    # Need samples where ERF measurement actually succeeded
    has_erf = meta['sigma_measured_erf'].notna() & (meta['erf_r_squared'] > 0.5)
    print(f"  Samples with usable ERF measurement: {has_erf.sum()}/{len(meta)} "
          f"({has_erf.sum()/len(meta)*100:.1f}%)")

    # Stratified sample across the sigma range
    eligible = meta[has_erf].copy()
    if len(eligible) > N_SAMPLES:
        # Sort by sigma_applied and pick evenly
        eligible = eligible.sort_values('sigma_applied_px').reset_index(drop=True)
        idxs = np.linspace(0, len(eligible) - 1, N_SAMPLES).astype(int)
        sample = eligible.iloc[idxs].copy()
    else:
        sample = eligible.copy()
    print(f"\nEvaluating {len(sample)} samples (stratified across sigma range)")
    print(f"  sigma_applied: min={sample['sigma_applied_px'].min():.2f}, "
          f"max={sample['sigma_applied_px'].max():.2f}")
    print(f"  diameter_model_px: min={sample['diameter_model_px'].min():.0f}, "
          f"max={sample['diameter_model_px'].max():.0f}")

    print("\nLoading model...")
    models = sorted((_REPO_ROOT / "Training" / "training_output" / "models")
                    .glob("*/checkpoints/dme_best.pth"), reverse=True)
    sys.path.insert(0, str(_REPO_ROOT / "Training"))
    from inference_real_crops import RealCropInference  # type: ignore
    inf = RealCropInference(model_path=str(models[0]), device='cpu')

    print("\nRunning model on each sample...")
    rows = []
    for i, (_, r) in enumerate(sample.iterrows()):
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(sample)}")
        idx = int(r['index'])
        png = SYNTH_DIR / "blur" / f"{idx:06d}.png"
        if not png.is_file():
            continue
        img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        try:
            pred = float(inf.estimate_blur_from_image(img))
        except Exception:
            pred = float('nan')
        rows.append({
            'index': idx,
            'sigma_applied': float(r['sigma_applied_px']),
            'sigma_erf': float(r['sigma_measured_erf']),
            'sigma_model': pred,
            'diameter_model_px': float(r['diameter_model_px']),
            'erf_r_squared': float(r['erf_r_squared']),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "three_way_synthetic.csv", index=False)
    print(f"\nWrote: {OUTPUT_DIR / 'three_way_synthetic.csv'}")

    # Summary statistics — gaps
    df['gap_model_vs_applied'] = df['sigma_model'] - df['sigma_applied']
    df['gap_erf_vs_applied'] = df['sigma_erf'] - df['sigma_applied']
    df['gap_model_vs_erf'] = df['sigma_model'] - df['sigma_erf']

    print("\n" + "=" * 80)
    print("AGREEMENT STATS (synthetic dataset, n={} samples)".format(len(df)))
    print("=" * 80)
    for col, label in [
        ('gap_model_vs_applied', 'model vs APPLIED (truth)'),
        ('gap_erf_vs_applied',   'ERF   vs APPLIED (truth)'),
        ('gap_model_vs_erf',     'model vs ERF'),
    ]:
        g = df[col].dropna()
        print(f"\n{label}:")
        print(f"  median gap: {g.median():+.3f} px")
        print(f"  mean abs:   {g.abs().mean():.3f} px")
        print(f"  RMSE:       {np.sqrt((g**2).mean()):.3f} px")
        print(f"  max abs:    {g.abs().max():.3f} px")

    # Plot — three panels: model vs truth, erf vs truth, model vs erf
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (x, y, xlabel, ylabel, title) in zip(axs, [
        ('sigma_applied', 'sigma_model', 'sigma_applied (truth, px)',
         'sigma_model (predicted, px)',
         'Model vs truth'),
        ('sigma_applied', 'sigma_erf', 'sigma_applied (truth, px)',
         'sigma_erf (measured, px)',
         'ERF vs truth'),
        ('sigma_erf', 'sigma_model', 'sigma_erf (measured, px)',
         'sigma_model (predicted, px)',
         'Model vs ERF'),
    ]):
        ax.scatter(df[x], df[y], s=14, alpha=0.5, c=df['diameter_model_px'],
                   cmap='viridis')
        lim = max(df[x].max(), df[y].max()) * 1.05
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.5, linewidth=1, label='y=x')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)

    fig.suptitle(
        f"Three-way agreement on synthetic dataset (n={len(df)} samples, "
        f"colour = object diameter in model-px)",
        fontsize=11,
    )
    fig.tight_layout()
    p = OUTPUT_DIR / "three_way_synthetic.png"
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWrote: {p}")


if __name__ == '__main__':
    main()
