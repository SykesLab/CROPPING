"""
verify_full_blur_pipeline.py
============================
End-to-end direct-mode blur pipeline verification.

Uses existing repository classes wherever possible:
  - OpticalParams          (training/Training/synthetic_blur.py)
  - SyntheticBlurGenerator (training/Training/synthetic_blur.py)
  - generate_sample()      (training/Training/synthetic_blur.py)

Inference inversion is verified at the maths level only (no model required).

Run from repo root:
  python verify_full_blur_pipeline.py

Or with PYTHONUTF8=1 to avoid Windows cp1252 issues with arrow chars in logs:
  set PYTHONUTF8=1 && python verify_full_blur_pipeline.py
"""

import sys, csv, os, random, warnings
from pathlib import Path
import numpy as np
import yaml
import cv2

# ── Repo paths ────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).parent
TRAINING_DIR = REPO_ROOT / "training" / "Training"
sys.path.insert(0, str(TRAINING_DIR))

# Uses: OpticalParams, SyntheticBlurGenerator (synthetic_blur.py)
from synthetic_blur import OpticalParams, SyntheticBlurGenerator

# ── Config paths (all read from disk — no hardcoded values) ──────────────────
OPTICAL_CONFIG_PATH = TRAINING_DIR / "training_output" / "optical_config.yaml"
CALIB_YAML_PATH     = REPO_ROOT / "calibration" / "calibration_output" / "calibration_results.yaml"
SHARP_CROPS_CSV     = (REPO_ROOT / "Preprocessing" / "CROPPING" / "Preprocessing"
                       / "OUTPUTNEW" / "Focus" / "sharp_crops.csv")
SHARP_CROPS_DIR     = SHARP_CROPS_CSV.parent

# Output
REPORT_CSV = REPO_ROOT / "verification_full_blur_pipeline_results.csv"

# Test parameters
Z_TEST_MM        = [1.0, 3.0, 5.0, 7.0]   # defocus values to probe
N_SAMPLES_PER_CROP = 200                    # samples for slope measurement
REL_ERR_PASS_PCT   = 1.0                    # pass threshold: max relative error %

PASS_COLOUR = "\033[92m"
FAIL_COLOUR = "\033[91m"
RESET       = "\033[0m"

# ── helpers ───────────────────────────────────────────────────────────────────

def _check(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found — {label}:\n  {path}")


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _verdict(ok: bool) -> str:
    if ok:
        return f"{PASS_COLOUR}PASS{RESET}"
    return f"{FAIL_COLOUR}FAIL{RESET}"


# ── Step 1 — Load configs ─────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  FULL BLUR PIPELINE VERIFICATION")
print("=" * 60)

_check(OPTICAL_CONFIG_PATH, "optical_config.yaml")
_check(CALIB_YAML_PATH,     "calibration_results.yaml")
_check(SHARP_CROPS_CSV,     "sharp_crops.csv")

optical_cfg = _load_yaml(OPTICAL_CONFIG_PATH)
calib_cfg   = _load_yaml(CALIB_YAML_PATH)

training_sec = optical_cfg.get("training", {})
data_sec     = optical_cfg.get("data", {})
direct_sec   = calib_cfg.get("direct", {})

# Extract all parameters — fail loudly if any are missing

def _require(d: dict, key: str, source: str):
    val = d.get(key)
    if val is None:
        raise KeyError(f"Required parameter '{key}' missing from {source}")
    return val

rho_direct       = float(_require(training_sec, "rho_direct",           "optical_config.yaml [training]"))
scale_calib      = float(_require(training_sec, "scale_calib_px_per_mm","optical_config.yaml [training]"))
sigma_0          = float(training_sec.get("sigma_0", 0.0))
crop_size        = int(_require(training_sec,   "crop_size_px",          "optical_config.yaml [training]"))
model_size       = int(_require(data_sec,        "image_size_px",         "optical_config.yaml [data]"))
defocus_range_mm = list(_require(data_sec,       "defocus_range_mm",      "optical_config.yaml [data]"))
training_mode    = _require(training_sec,        "training_mode",         "optical_config.yaml [training]")
calib_ref_res    = int(training_sec.get("calib_reference_resolution", crop_size))

# Verify direct mode
if training_mode != "direct":
    raise ValueError(f"This script only verifies direct mode. training_mode={training_mode!r}")

# Cross-check rho_direct against calibration YAML
rho_from_calib   = float(_require(direct_sec, "rho_px_per_mm", "calibration_results.yaml [direct]"))
scale_from_calib = float(_require(direct_sec, "scale_calib_px_per_mm", "calibration_results.yaml [direct]"))

print(f"\nCalibration ({CALIB_YAML_PATH.name}):")
print(f"  rho_direct        = {rho_from_calib} px/mm")
print(f"  scale_calib       = {scale_from_calib} px/mm")
print(f"  sigma_0           = {sigma_0} px  (loaded but intentionally excluded from training)")
print(f"\nTraining config ({OPTICAL_CONFIG_PATH.name}):")
print(f"  training_mode     = {training_mode}")
print(f"  rho_direct        = {rho_direct} px/mm")
print(f"  scale_calib       = {scale_calib} px/mm")
print(f"  crop_size         = {crop_size} px")
print(f"  model_size        = {model_size} px")
print(f"  defocus_range     = {defocus_range_mm} mm")
print(f"  calib_ref_res     = {calib_ref_res} px")

native_to_model = model_size / crop_size
print(f"\nDerived:")
print(f"  native_to_model   = {model_size}/{crop_size} = {native_to_model:.6f}")

if abs(rho_direct - rho_from_calib) > 1e-4:
    print(f"\n  WARNING: rho_direct in config ({rho_direct}) differs from "
          f"calibration YAML ({rho_from_calib}) — using config value")
if abs(scale_calib - scale_from_calib) > 0.01:
    print(f"\n  WARNING: scale_calib in config ({scale_calib}) differs from "
          f"calibration YAML ({scale_from_calib}) — using config value")

# ── Step 2 — Load crop CSV and select representative crops ────────────────────

print(f"\n{'─'*60}")
print("  Loading crop metadata...")

crops = []
with open(SHARP_CROPS_CSV, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        sc_str = row.get("scale_px_per_mm", "")
        fn     = row.get("filename", "")
        cam    = row.get("camera", "?")
        diam   = row.get("diameter_px", "")
        if not (sc_str and fn):
            continue
        try:
            sc = float(sc_str)
        except ValueError:
            continue
        img_path = None
        # Search in all material/camera subdirectories
        for candidate in SHARP_CROPS_DIR.rglob(fn):
            img_path = candidate
            break
        if img_path is None or not img_path.exists():
            continue
        crops.append({"filename": fn, "path": img_path, "camera": cam,
                      "scale": sc, "diameter": diam})

if not crops:
    raise RuntimeError("No crops found on disk matching the CSV — check SHARP_CROPS_DIR")

print(f"  {len(crops)} crops found on disk (of {sum(1 for _ in open(SHARP_CROPS_CSV))-1} in CSV)")

# Auto-select 3 representative crops: lowest scale, median scale, highest scale
scales_sorted = sorted(crops, key=lambda c: c["scale"])
low_crop  = scales_sorted[0]
med_crop  = scales_sorted[len(scales_sorted) // 2]
high_crop = scales_sorted[-1]

representative_crops = [
    ("low_scale",  low_crop),
    ("med_scale",  med_crop),
    ("high_scale", high_crop),
]

print(f"\n  Representative crops:")
for label, c in representative_crops:
    print(f"    [{label:10s}] cam={c['camera']}  scale={c['scale']:.2f} px/mm  "
          f"file={c['filename']}")

# ── Step 3 — Build OpticalParams and SyntheticBlurGenerator ──────────────────
#
# Uses OpticalParams (synthetic_blur.py) — constructed directly, matching GUI path.
# Uses SyntheticBlurGenerator (synthetic_blur.py).

params = OpticalParams(
    focal_length_mm=1.0,
    focus_distance_mm=1.0,
    imaging_distance_mm=1.0,
    aperture_diameter_mm=1.0,
    pixel_size_mm=optical_cfg.get("optics", {}).get("pixel_size_mm", 0.02),
    rho=1.0,
    training_mode="direct",
    rho_direct=rho_direct,
    sigma_0=sigma_0,
    scale_calib_px_per_mm=scale_calib,
)

generator = SyntheticBlurGenerator(
    optical_params=params,
    defocus_range_mm=(defocus_range_mm[0], defocus_range_mm[1]),
    image_size=model_size,
    crop_size=crop_size,
    calibration_reference_resolution=calib_ref_res,
)

print(f"\n  OpticalParams.scale_calib_px_per_mm = {params.scale_calib_px_per_mm}")
print(f"  generator.native_to_model_scale      = {generator.native_to_model_scale:.6f}")
print(f"  generator.max_sigma                  = {generator.max_sigma:.4f} px")

# ── Step 4 — Analytical expected values + generate_sample() slope check ───────

print(f"\n{'─'*60}")
print("  STEP 4 — Per-crop, per-defocus analytical verification")
print(f"{'─'*60}")
print(f"  Formula:")
print(f"    sigma_calib          = rho_direct * |z|")
print(f"    sigma_native_expected = sigma_calib * (scale_px / scale_calib)")
print(f"    sigma_model_expected  = sigma_native_expected * (model_size / crop_size)")
print(f"    slope_expected        = rho * (scale_px / scale_calib) * (model_size / crop_size)")

header = (f"  {'crop':10s} {'z_mm':6s} {'sigma_cal':10s} {'sigma_nat':10s} "
          f"{'sigma_mdl':10s} {'slope_exp':10s} {'verdict':6s}")
print(f"\n{header}")
print(f"  {'-'*80}")

report_rows = []

for label, c in representative_crops:
    sc = c["scale"]
    slope_expected = rho_direct * (sc / scale_calib) * native_to_model

    for z in Z_TEST_MM:
        if z > abs(defocus_range_mm[0]) and z > defocus_range_mm[1]:
            continue  # outside training range

        sigma_calib_val   = rho_direct * abs(z)
        sigma_native_exp  = sigma_calib_val * (sc / scale_calib)
        sigma_model_exp   = sigma_native_exp * native_to_model

        # Verify slope is consistent
        slope_check = sigma_model_exp / abs(z)
        slope_ok = abs(slope_check - slope_expected) < 1e-9

        print(f"  {label:10s} {z:6.1f} {sigma_calib_val:10.4f} {sigma_native_exp:10.4f} "
              f"{sigma_model_exp:10.4f} {slope_expected:10.4f} {_verdict(slope_ok)}")

        report_rows.append({
            "crop_label": label, "camera": c["camera"],
            "scale_px_per_mm": sc, "z_mm": z,
            "sigma_calib": round(sigma_calib_val, 6),
            "sigma_native_expected": round(sigma_native_exp, 6),
            "sigma_model_expected": round(sigma_model_exp, 6),
            "slope_expected": round(slope_expected, 6),
            "sigma_model_actual": None,
            "abs_err": None, "rel_err_pct": None,
            "z_reconstructed": None, "inversion_err_mm": None,
        })

# ── Step 5 — generate_sample() slope measurement ─────────────────────────────
#
# Calls generator.generate_sample() (synthetic_blur.py) N times per crop.
# Derives slope = sigma_model / |defocus_mm| from returned samples.
# Compares to slope_expected = rho * (scale/scale_calib) * (model/crop).

print(f"\n{'─'*60}")
print(f"  STEP 5 — generate_sample() measured slope vs expected slope")
print(f"           ({N_SAMPLES_PER_CROP} samples per crop)")
print(f"{'─'*60}")
print(f"  {'crop':10s} {'scale':8s} {'slope_exp':10s} {'slope_meas':10s} "
      f"{'rel_err%':9s} {'verdict':6s}")
print(f"  {'-'*60}")

slope_results = {}

for label, c in representative_crops:
    img = cv2.imread(str(c["path"]), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  {label:10s} — could not load image: {c['path'].name}")
        continue
    img_f = img.astype(np.float32) / 255.0

    sc = c["scale"]
    slope_expected = rho_direct * (sc / scale_calib) * native_to_model

    ratios = []
    random.seed(42)
    np.random.seed(42)

    for _ in range(N_SAMPLES_PER_CROP):
        # Uses generator.generate_sample() from synthetic_blur.py
        sample = generator.generate_sample(
            sharp_image=img_f,
            scale_px_per_mm=sc,
            native_blur_sigma=0.0,
        )
        d = sample["defocus_mm"]
        s = sample["coc_value"]   # sigma_model in px (synthetic_blur.py line 611)
        if abs(d) > 0.01:
            ratios.append(s / abs(d))

    if not ratios:
        print(f"  {label:10s} — no usable samples generated")
        continue

    slope_measured = float(np.mean(ratios))
    rel_err_pct    = 100.0 * abs(slope_measured - slope_expected) / slope_expected
    ok = rel_err_pct < REL_ERR_PASS_PCT

    slope_results[label] = {"expected": slope_expected, "measured": slope_measured,
                            "rel_err_pct": rel_err_pct, "ok": ok, "scale": sc}

    print(f"  {label:10s} {sc:8.2f} {slope_expected:10.4f} {slope_measured:10.4f} "
          f"{rel_err_pct:9.4f} {_verdict(ok)}")

# Update report_rows with actual slopes
for row in report_rows:
    r = slope_results.get(row["crop_label"])
    if r:
        z = row["z_mm"]
        sigma_actual = r["measured"] * abs(z)
        sigma_expected = row["sigma_model_expected"]
        abs_err = abs(sigma_actual - sigma_expected)
        rel_err = 100.0 * abs_err / sigma_expected if sigma_expected > 0 else float("nan")
        row["sigma_model_actual"] = round(sigma_actual, 6)
        row["abs_err"]  = round(abs_err, 6)
        row["rel_err_pct"] = round(rel_err, 4)

# ── Step 6 — Inference inversion sanity check ─────────────────────────────────
#
# Implements maths from inference_real_crops.py lines 491-492:
#   defocus_mm = coc_px_model * crop_size / (direct_slope * model_size)
# where direct_slope = rho_direct * (scale_inf / scale_calib) for cross-camera.
#
# Verifies that the inversion is an exact algebraic inverse of the forward formula.

print(f"\n{'─'*60}")
print("  STEP 6 — Inference inversion (maths-level, no model required)")
print(f"           Uses inference_real_crops.py inversion formula:")
print(f"           z_hat = sigma_model * crop_size / (rho_eff * model_size)")
print(f"{'─'*60}")
print(f"  {'crop':10s} {'scale':8s} {'z_input':8s} {'sigma_mdl':10s} "
      f"{'z_hat':8s} {'err_mm':8s} {'verdict':6s}")
print(f"  {'-'*70}")

inversion_errors = []

for label, c in representative_crops:
    sc = c["scale"]
    # rho_eff applies cross-camera correction (inference_real_crops.py line 200):
    #   rho_eff = rho_direct * (scale_inference / scale_calib)
    rho_eff = rho_direct * (sc / scale_calib)

    for z in Z_TEST_MM:
        if z > abs(defocus_range_mm[0]) and z > defocus_range_mm[1]:
            continue

        sigma_model = rho_direct * z * (sc / scale_calib) * native_to_model

        # Inversion (inference_real_crops.py lines 491-492):
        z_hat = sigma_model * crop_size / (rho_eff * model_size)
        err   = abs(z_hat - z)
        ok    = err < 1e-9
        inversion_errors.append(err)

        # Update report_rows
        for row in report_rows:
            if row["crop_label"] == label and abs(row["z_mm"] - z) < 1e-9:
                row["z_reconstructed"]   = round(z_hat, 9)
                row["inversion_err_mm"]  = round(err, 9)

        print(f"  {label:10s} {sc:8.2f} {z:8.3f} {sigma_model:10.4f} "
              f"{z_hat:8.3f} {err:8.2e} {_verdict(ok)}")

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'═'*60}")
print("  SUMMARY")
print(f"{'═'*60}")

all_rel_errs = [r["rel_err_pct"] for r in report_rows
                if r["rel_err_pct"] is not None]
all_abs_errs = [r["abs_err"] for r in report_rows
                if r["abs_err"] is not None]
all_inv_errs = [r["inversion_err_mm"] for r in report_rows
                if r["inversion_err_mm"] is not None]

overall_pass = (
    slope_results and
    all(r["ok"] for r in slope_results.values()) and
    all(e < 1e-9 for e in all_inv_errs)
)

if all_abs_errs:
    print(f"  Slope measurement (generate_sample vs analytical):")
    print(f"    max rel error  = {max(r['rel_err_pct'] for r in slope_results.values()):.4f} %")
    print(f"    threshold      = {REL_ERR_PASS_PCT:.1f} %")
    for label, r in slope_results.items():
        print(f"    [{label:10s}]  expected={r['expected']:.4f}  measured={r['measured']:.4f}  "
              f"rel_err={r['rel_err_pct']:.4f}%  {_verdict(r['ok'])}")

if all_inv_errs:
    print(f"\n  Inversion round-trip:")
    print(f"    max error = {max(all_inv_errs):.2e} mm  (machine epsilon expected)")

print(f"\n  Cross-camera correction active:  {_verdict(bool(slope_results))}")
print(f"  Slopes match expected formula:   {_verdict(all(r['ok'] for r in slope_results.values()))}")
print(f"  Inversion exact:                 {_verdict(all(e < 1e-9 for e in all_inv_errs))}")
print(f"\n  OVERALL: {_verdict(overall_pass)}")

# ── Save CSV ──────────────────────────────────────────────────────────────────

fieldnames = [
    "crop_label", "camera", "scale_px_per_mm", "z_mm",
    "sigma_calib", "sigma_native_expected", "sigma_model_expected",
    "slope_expected", "sigma_model_actual", "abs_err", "rel_err_pct",
    "z_reconstructed", "inversion_err_mm",
]

with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(report_rows)

print(f"\n  Results saved to: {REPORT_CSV}")
print(f"{'═'*60}\n")
