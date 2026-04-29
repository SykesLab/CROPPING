"""
test_two_checkpoints.py
=======================

Compare two checkpoints on all 61 calibration cines using cfg1
preprocessing. CPU device so we don't fight a running GPU training.

Usage (defaults to comparing this run's dme_best vs dme_calib_best):
    python -m Extras.test_two_checkpoints
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _module in ("Calibration", "Inference", "Preprocessing", "Training", ""):
    _p = str(_REPO_ROOT / _module) if _module else str(_REPO_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cine_loader import CineLoader  # noqa: E402
from sphere_processing import find_sphere_center, crop_to_square  # noqa: E402
import inference_engine as ie  # noqa: E402

RUN_DIR = _REPO_ROOT / "Training" / "training_output" / "models" / \
            "hybrid_resumed_4" / "02_0321"
CHECKPOINTS = [
    ("dme_best", RUN_DIR / "checkpoints" / "dme_best.pth"),
    ("dme_calib_best", RUN_DIR / "checkpoints" / "dme_calib_best.pth"),
]
CONFIG_PATH = RUN_DIR / "training_config.yaml"
OUTPUT_DIR = Path(__file__).resolve().parent / "two_checkpoints_output"


def find_calibration_picks() -> list[tuple]:
    """All 61 raw calibration .cine files with true z from measurements.csv."""
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
    picks: list = []
    for row_idx, row in m_orig.iterrows():
        cine_num = int(row_idx) + 1
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


def build_engine(ckpt_path: Path) -> ie.InferenceEngine:
    import yaml
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    train = cfg.get("training", {})
    rho = float(train.get("rho_direct") or train.get("rho") or 1.0)
    sigma_0 = float(train.get("sigma_0", 0.0))
    s_calib = float(train.get("scale_calib_px_per_mm",
                                cfg.get("calibration", {})
                                  .get("reference_resolution", 1.0)))
    settings = {
        "model_path": str(ckpt_path),
        "rho": rho, "sigma_0": sigma_0,
        "s_calib": s_calib, "s_c": s_calib,
        "feather_px": 40, "crop_size": 299,
        "device": "cpu",          # avoid GPU contention with running training
        "flatten_mode": "calibration",  # cfg1
        "inner_margin_px": 20,
    }
    engine = ie.InferenceEngine(settings)
    engine.load_model()
    return engine


def run_one(engine: ie.InferenceEngine, crop: np.ndarray) -> dict:
    _, tensor_input = engine.preprocess_crop(crop)
    native_size = max(crop.shape[0], crop.shape[1])
    return engine.run_inference(tensor_input, native_size)


def main():
    print("=" * 78)
    print("TWO-CHECKPOINT COMPARISON (cfg1 preprocessing, CPU device)")
    print("=" * 78)
    for label, p in CHECKPOINTS:
        if not p.is_file():
            print(f"ERROR: missing checkpoint: {label} -> {p}")
            return
        print(f"  {label}: {p.name}")

    engines = []
    for label, ckpt in CHECKPOINTS:
        print(f"\n[loading {label}]")
        engines.append((label, build_engine(ckpt)))

    picks = find_calibration_picks()
    print(f"\nProcessing {len(picks)} calibration cines (CPU forward — slower than GPU)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list = []
    for cine_path, true_z in picks:
        crop = load_raw_crop(cine_path)
        if crop is None:
            continue
        results = {}
        for label, engine in engines:
            r = run_one(engine, crop)
            results[label] = r
        row = {
            'cine': cine_path.name,
            'true_z_mm': true_z,
            'abs_true_z_mm': abs(true_z),
        }
        for label, _ in engines:
            r = results[label]
            err = r['defocus_mm'] - abs(true_z)
            row[f'{label}_sigma_native'] = r['sigma_native']
            row[f'{label}_pred_z_mm'] = r['defocus_mm']
            row[f'{label}_err_mm'] = err
        rows.append(row)
        # Compact line print
        bits = [f"z={true_z:+.2f}"]
        for label, _ in engines:
            r = results[label]
            err = r['defocus_mm'] - abs(true_z)
            bits.append(f"{label}: z_pred={r['defocus_mm']:+.3f} (err {err:+.3f})")
        print("  " + " | ".join(bits))

    if not rows:
        print("No rows produced.")
        return

    # CSV
    csv_path = OUTPUT_DIR / "two_checkpoints_compare.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                          for k, v in r.items()})

    # Region-stratified MAE
    print()
    print("=" * 78)
    print("REGION-STRATIFIED MAE")
    print("=" * 78)

    def bin_label(z):
        if z < -4.0: return "A: far negative (z<-4)"
        if z < -1.0: return "B: mid negative (-4<=z<-1)"
        if z <= 1.0: return "C: near focus (|z|<=1)"
        if z <= 4.0: return "D: mid positive (1<z<=4)"
        return "E: far positive (z>4)"

    bins: dict = {}
    for r in rows:
        b = bin_label(r['true_z_mm'])
        bins.setdefault(b, []).append(r)

    header = f"{'Region':<32} {'n':>3}"
    for label, _ in engines:
        header += f"  {label+' MAE':>17}"
    header += "  delta"
    print(header)
    for b in sorted(bins.keys()):
        g = bins[b]
        line = f"{b:<32} {len(g):>3}"
        maes = []
        for label, _ in engines:
            mae = float(np.mean([abs(r[f'{label}_err_mm']) for r in g]))
            line += f"  {mae:>17.3f}"
            maes.append(mae)
        line += f"  {maes[0]-maes[1]:+.3f}"
        print(line)

    print("-" * len(header))
    line = f"{'AGGREGATE (all 61)':<32} {len(rows):>3}"
    maes = []
    for label, _ in engines:
        mae = float(np.mean([abs(r[f'{label}_err_mm']) for r in rows]))
        line += f"  {mae:>17.3f}"
        maes.append(mae)
    line += f"  {maes[0]-maes[1]:+.3f}"
    print(line)
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
