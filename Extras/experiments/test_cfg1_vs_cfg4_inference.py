"""
test_cfg1_vs_cfg4_inference.py
==============================

End-to-end inference test: swap the inference engine's preprocess step
from cfg4 (boundary_normalise — current default) to cfg1 (calibration-mode
flatten_sphere_crop, what tertiary training samples actually look like)
and measure the change in predicted defocus on calibration spheres of
known z.

Hypothesis (from compare_flatten_configs.py):
  cfg4 inflates apparent sigma by +0.4 to +1.1 px on out-of-focus inputs
  versus the raw frame. Through the model's calibration this should
  translate to ~0.5-1.0 mm of phantom over-prediction. cfg1 preserves
  the original sigma to within ~0.05 px.

Run from CROPPING root:
    python -m Extras.test_cfg1_vs_cfg4_inference
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _module in ("Calibration", "Inference", "Preprocessing", "Training", ""):
    _p = str(_REPO_ROOT / _module) if _module else str(_REPO_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cine_loader import CineLoader  # noqa: E402
from sphere_processing import find_sphere_center, crop_to_square  # noqa: E402
import inference_engine as ie  # noqa: E402


# ── User-configurable: checkpoint to load ────────────────────────────
CHECKPOINT_PATH = _REPO_ROOT / "Training" / "training_output" / "models" / \
                   "20260427_045039_model" / "checkpoints" / "dme_best.pth"

OUTPUT_DIR = Path(__file__).resolve().parent / "cfg1_vs_cfg4_output"


def find_calibration_picks() -> list[tuple]:
    """Locate ALL raw calibration .cine files, with true z from latest
    measurements.csv."""
    spheres_dir = _REPO_ROOT / "calibration spheres" / "9mm"
    runs_dir = _REPO_ROOT / "Calibration" / "runs"
    if not spheres_dir.is_dir() or not runs_dir.is_dir():
        return []
    latest = None
    for r in sorted(runs_dir.glob("*"), key=lambda p: p.stat().st_mtime,
                     reverse=True):
        if (r / "measurements.csv").is_file():
            latest = r
            break
    if latest is None:
        return []
    import pandas as pd
    m_orig = pd.read_csv(latest / "measurements.csv")
    # Map row index in original (unsorted) DataFrame -> cine number
    picks: list = []
    for row_idx, row in m_orig.iterrows():
        cine_num = int(row_idx) + 1  # 9mm_1.cine ... 9mm_61.cine
        cine_path = spheres_dir / f"9mm_{cine_num}.cine"
        if cine_path.is_file():
            picks.append((cine_path, float(row["z_mm"])))
    picks.sort(key=lambda t: t[1])
    return picks


def load_raw_crop(cine_path: Path) -> Optional[np.ndarray]:
    loader = CineLoader(str(cine_path))
    if loader.cine_obj is None:
        return None
    frame = loader.extract_frame(loader.frame_range[0])
    if frame is None:
        return None
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    det = find_sphere_center(frame, upper_only=True)
    if det is None:
        return None
    cx, cy, radius = det
    crop = crop_to_square(frame, cx, cy, radius, padding=1.2)
    if crop.dtype != np.uint8:
        crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return crop


def build_engine(flatten_mode: str) -> ie.InferenceEngine:
    """Construct an InferenceEngine with a specific flatten_mode."""
    import yaml
    cfg_path = CHECKPOINT_PATH.parent.parent / "training_config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    train = cfg.get("training", {})
    rho = float(train.get("rho_direct") or train.get("rho") or 1.0)
    sigma_0 = float(train.get("sigma_0", 0.0))
    s_calib = float(train.get("scale_calib_px_per_mm",
                                cfg.get("calibration", {})
                                  .get("reference_resolution", 1.0)))
    settings = {
        "model_path": str(CHECKPOINT_PATH),
        "rho": rho,
        "sigma_0": sigma_0,
        "s_calib": s_calib,
        "s_c": s_calib,
        "feather_px": 40,
        "crop_size": 299,
        "device": "cpu",
        "flatten_mode": flatten_mode,
        "inner_margin_px": 20,
    }
    engine = ie.InferenceEngine(settings)
    engine.load_model()
    return engine


def run_one(engine: ie.InferenceEngine, crop_uint8: np.ndarray) -> dict:
    """Run preprocess + model on a single crop, return result dict."""
    norm_img, tensor_input = engine.preprocess_crop(crop_uint8)
    native_size = max(crop_uint8.shape[0], crop_uint8.shape[1])
    return engine.run_inference(tensor_input, native_size), norm_img


def main():
    print("=" * 78)
    print("CFG1 vs CFG4 INFERENCE COMPARISON")
    print("=" * 78)
    print(f"Checkpoint: {CHECKPOINT_PATH.name}")
    print(f"Parent:     {CHECKPOINT_PATH.parent.parent.name}")

    if not CHECKPOINT_PATH.is_file():
        print(f"ERROR: checkpoint not found: {CHECKPOINT_PATH}")
        return

    eng_cfg4 = build_engine("boundary_normalise")
    eng_cfg1 = build_engine("calibration")
    print(f"  Engines loaded for cfg4 (boundary_normalise) and "
          f"cfg1 (calibration, inner_margin=20).")

    picks = find_calibration_picks()
    if not picks:
        print("No calibration .cine files found — aborting")
        return
    print(f"\nProcessing {len(picks)} calibration spheres "
          f"(z range {picks[0][1]:.1f} to {picks[-1][1]:.1f} mm)")

    rows: list = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for cine_path, true_z in picks:
        crop = load_raw_crop(cine_path)
        if crop is None:
            print(f"  SKIP {cine_path.name}: failed to load")
            continue

        res4, norm4 = run_one(eng_cfg4, crop)
        res1, norm1 = run_one(eng_cfg1, crop)

        err4 = res4['defocus_mm'] - abs(true_z)
        err1 = res1['defocus_mm'] - abs(true_z)
        rows.append({
            'cine': cine_path.name,
            'true_z_mm': true_z,
            'abs_true_z_mm': abs(true_z),
            'cfg4_sigma_native_px': res4['sigma_native'],
            'cfg1_sigma_native_px': res1['sigma_native'],
            'sigma_diff_px (cfg4-cfg1)': res4['sigma_native'] - res1['sigma_native'],
            'cfg4_pred_z_mm': res4['defocus_mm'],
            'cfg1_pred_z_mm': res1['defocus_mm'],
            'cfg4_err_mm': err4,
            'cfg1_err_mm': err1,
            'improvement_mm': abs(err4) - abs(err1),
        })
        print(f"  z={true_z:+.2f}: cfg4 sigma={res4['sigma_native']:.3f}px "
              f"-> z_pred={res4['defocus_mm']:+.3f} (err {err4:+.3f}) | "
              f"cfg1 sigma={res1['sigma_native']:.3f}px "
              f"-> z_pred={res1['defocus_mm']:+.3f} (err {err1:+.3f})")

    if rows:
        csv_path = OUTPUT_DIR / "cfg1_vs_cfg4_inference.csv"
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                              for k, v in r.items()})
        # Aggregate
        mae4 = float(np.mean([abs(r['cfg4_err_mm']) for r in rows]))
        mae1 = float(np.mean([abs(r['cfg1_err_mm']) for r in rows]))
        print()
        print("=" * 78)
        print(f"AGGREGATE — N={len(rows)}")
        print(f"  cfg4 MAE: {mae4:.3f} mm  (current inference default)")
        print(f"  cfg1 MAE: {mae1:.3f} mm  (calibration-mode flatten)")
        print(f"  improvement (cfg4 - cfg1): {mae4 - mae1:+.3f} mm")
        print(f"  CSV:  {csv_path}")


if __name__ == "__main__":
    main()
