"""
Stage 2: Sphere-Domain Training Dataset Packager

Selects near-focus processed sphere images from Stage 1 batch calibration outputs
and packages them into the exact format the training GUI's synthetic blur generator
expects (sharp_crops.csv schema).

Usage:
    python stage2_packager.py
"""

import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from paths_config import STAGE1_ROOT, STAGE2_OUTPUT_DIR

# -- Configuration ------------------------------------------------------------

OUTPUT_DIR = STAGE2_OUTPUT_DIR

# Which sphere's calibration to treat as authoritative
AUTHORITATIVE_SPHERE = "7mm"

# Frame selection
SELECTION_MODE = "sigma"   # "abs_defocus" or "sigma"
Z_THRESHOLD_MM = 1.0             # |z| < this (for abs_defocus mode)
SIGMA_THRESHOLD_PX = 1.5         # sigma < this (for sigma mode)


# -- Data structures ---------------------------------------------------------

class FrameRecord:
    """One row from per_frame_measurements.csv with derived fields."""

    def __init__(self, row: dict, sphere_folder: str):
        self.source_filename = row['source_filename']
        self.mechanical_position_mm = float(row['mechanical_position_mm'])
        self.defocus_z_mm = float(row['defocus_z_mm'])
        self.is_focus_frame = row['is_focus_frame'] == 'True'
        self.sphere_diameter_mm = float(row['sphere_diameter_mm'])
        self.px_per_mm = float(row['px_per_mm'])
        self.original_resolution = row['original_resolution']
        self.processed_resolution = row['processed_resolution']
        self.sigma_px = float(row['sigma_px'])
        self.fit_confidence = float(row['fit_confidence'])
        self.rays_accepted = int(row['rays_accepted'])
        self.sphere_folder = sphere_folder

        # Derived
        self.image_filename = Path(self.source_filename).stem + '.png'
        self.diameter_px = self.px_per_mm * self.sphere_diameter_mm


# -- Load Stage 1 outputs ----------------------------------------------------

def load_calibration_yaml(path: Path) -> dict:
    """Load a calibration_summary.yaml."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_per_frame_csv(path: Path, sphere_folder: str) -> List[FrameRecord]:
    """Load per_frame_measurements.csv into FrameRecord list."""
    records = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(FrameRecord(row, sphere_folder))
    return records


def discover_spheres(stage1_root: Path) -> List[str]:
    """Find sphere subfolders that have per_frame_measurements.csv."""
    spheres = []
    for sub in sorted(stage1_root.iterdir()):
        if sub.is_dir() and (sub / 'per_frame_measurements.csv').exists():
            spheres.append(sub.name)
    return spheres


# -- Frame selection ----------------------------------------------------------

def select_frames(
    records: List[FrameRecord],
    mode: str = "abs_defocus",
    z_threshold: float = 1.0,
    sigma_threshold: Optional[float] = None,
) -> List[FrameRecord]:
    """Select frames based on configurable criteria."""
    if mode == "abs_defocus":
        selected = [r for r in records if abs(r.defocus_z_mm) <= z_threshold]
    elif mode == "sigma":
        if sigma_threshold is None:
            raise ValueError("sigma_threshold required for sigma selection mode")
        selected = [r for r in records if r.sigma_px > 0 and r.sigma_px <= sigma_threshold]
    else:
        raise ValueError(f"Unknown selection mode: {mode}")

    return selected


# -- Output writers -----------------------------------------------------------

def write_sharp_crops_csv(path: Path, records: List[FrameRecord]):
    """Write training-GUI-ready sharp_crops.csv with the 4 mandatory columns."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'diameter_px', 'scale_px_per_mm', 'native_blur_sigma', 'camera'])
        for r in records:
            writer.writerow([
                r.image_filename,
                f'{r.diameter_px:.1f}',
                f'{r.px_per_mm:.2f}',
                f'{r.sigma_px:.6f}',
                'phantom',
            ])


def write_traceability_csv(
    path: Path,
    records: List[FrameRecord],
    selection_rule: str,
    auth_calibration_path: str,
):
    """Write rich traceability CSV with all provenance fields."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'filename', 'sphere_diameter_mm', 'source_folder', 'source_cine',
            'mechanical_position_mm', 'defocus_z_mm', 'sigma_px', 'fit_confidence',
            'rays_accepted', 'px_per_mm', 'processed_resolution', 'is_focus_frame',
            'diameter_px', 'scale_px_per_mm', 'native_blur_sigma', 'camera',
            'selection_rule', 'authoritative_calibration',
        ])
        for r in records:
            writer.writerow([
                r.image_filename,
                f'{r.sphere_diameter_mm:.1f}',
                r.sphere_folder,
                r.source_filename,
                f'{r.mechanical_position_mm:.2f}',
                f'{r.defocus_z_mm:.2f}',
                f'{r.sigma_px:.6f}',
                f'{r.fit_confidence:.4f}',
                r.rays_accepted,
                f'{r.px_per_mm:.2f}',
                r.processed_resolution,
                r.is_focus_frame,
                f'{r.diameter_px:.1f}',
                f'{r.px_per_mm:.2f}',
                f'{r.sigma_px:.6f}',
                'phantom',
                selection_rule,
                auth_calibration_path,
            ])


def write_dataset_summary(
    path: Path,
    auth_cal: dict,
    auth_sphere: str,
    selection_rule: str,
    per_sphere_counts: Dict[str, int],
    total_selected: int,
):
    """Write dataset_summary.yaml."""
    data = {
        'authoritative_calibration': f'{auth_sphere}/calibration_summary.yaml',
        'authoritative_rho': float(auth_cal['rho_px_per_mm']),
        'authoritative_sigma_0': float(auth_cal['sigma_0_px']),
        'authoritative_px_per_mm': float(auth_cal['px_per_mm']),
        'selection_rule': selection_rule,
        'total_selected': total_selected,
        'per_sphere': {k: int(v) for k, v in per_sphere_counts.items()},
        'crop_size': 960,
        'output_dir': str(OUTPUT_DIR),
        'timestamp': datetime.now().isoformat(),
    }
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def write_experiment_config(path: Path, auth_cal: dict):
    """Write experiment_config.yaml compatible with training GUI calibration loader."""
    defocus_range = list(auth_cal.get('z_range_mm', [-8.0, 4.0]))
    data = {
        'calibration_mode': 'direct',
        'rho_px_per_mm': float(auth_cal['rho_px_per_mm']),
        'sigma_0': float(auth_cal['sigma_0_px']),
        'scale_calib_px_per_mm': float(auth_cal['px_per_mm']),
        'reference_resolution': 960,
        'defocus_range_mm': [float(v) for v in defocus_range],
    }
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# -- Sanity summaries ---------------------------------------------------------

def print_sanity_summaries(
    selected: List[FrameRecord],
    auth_cal: dict,
    per_sphere: Dict[str, List[FrameRecord]],
):
    """Print sanity checks after selection."""
    auth_px_per_mm = float(auth_cal['px_per_mm'])

    print(f"\n{'-'*50}")
    print("SANITY CHECKS")
    print(f"{'-'*50}")

    # Per-sphere sigma distribution
    print(f"\n  native_blur_sigma (sigma_px) for selected frames:")
    for sphere, recs in sorted(per_sphere.items()):
        sigmas = [r.sigma_px for r in recs]
        print(f"    {sphere}: min={min(sigmas):.4f}  mean={sum(sigmas)/len(sigmas):.4f}  "
              f"max={max(sigmas):.4f} px")

    # Per-sphere diameter_px
    print(f"\n  diameter_px (sphere pixel diameter):")
    for sphere, recs in sorted(per_sphere.items()):
        d = recs[0].diameter_px
        print(f"    {sphere}: {d:.1f} px")

    # Cross-camera correction preview
    print(f"\n  cross-camera correction vs authoritative ({auth_px_per_mm:.2f} px/mm):")
    for sphere, recs in sorted(per_sphere.items()):
        px_mm = recs[0].px_per_mm
        cc = px_mm / auth_px_per_mm
        print(f"    {sphere}: px/mm={px_mm:.2f}  cc_factor={cc:.4f}")

    # Resolution check
    resolutions = set(r.processed_resolution for r in selected)
    print(f"\n  processed resolutions: {resolutions}")
    if resolutions != {'960x960'}:
        print("  WARNING: not all images are 960×960!")

    # Defocus coverage
    print(f"\n  defocus z distribution of selected frames:")
    z_bins = {}
    for r in selected:
        z_key = f"{r.defocus_z_mm:+.1f}"
        z_bins[z_key] = z_bins.get(z_key, 0) + 1
    for z_key in sorted(z_bins.keys(), key=lambda x: float(x)):
        print(f"    z={z_key} mm: {z_bins[z_key]} frames")

    print(f"{'-'*50}\n")


# -- Main pipeline ------------------------------------------------------------

def run_stage2():
    """Execute Stage 2 packager."""
    print("Stage 2: Sphere-Domain Dataset Packager")
    print(f"Stage 1 root: {STAGE1_ROOT}")
    print(f"Output:       {OUTPUT_DIR}")
    print()

    # -- 1. Discover spheres --
    spheres = discover_spheres(STAGE1_ROOT)
    if not spheres:
        print("ERROR: No sphere folders found with per_frame_measurements.csv")
        sys.exit(1)

    print(f"Found {len(spheres)} sphere(s): {', '.join(spheres)}")

    # -- 2. Load authoritative calibration --
    auth_path = STAGE1_ROOT / AUTHORITATIVE_SPHERE / 'calibration_summary.yaml'
    if not auth_path.exists():
        print(f"ERROR: Authoritative calibration not found: {auth_path}")
        sys.exit(1)

    auth_cal = load_calibration_yaml(auth_path)
    print(f"Authoritative calibration: {AUTHORITATIVE_SPHERE} "
          f"(rho={auth_cal['rho_px_per_mm']}, sigma_0={auth_cal['sigma_0_px']}, "
          f"px/mm={auth_cal['px_per_mm']})")
    print()

    # -- 3. Load and select frames --
    if SELECTION_MODE == "abs_defocus":
        selection_rule = f"abs_defocus < {Z_THRESHOLD_MM} mm"
    else:
        selection_rule = f"sigma < {SIGMA_THRESHOLD_PX} px"

    all_selected: List[FrameRecord] = []
    per_sphere_selected: Dict[str, List[FrameRecord]] = {}
    per_sphere_counts: Dict[str, int] = {}

    for sphere in spheres:
        csv_path = STAGE1_ROOT / sphere / 'per_frame_measurements.csv'
        records = load_per_frame_csv(csv_path, sphere)

        selected = select_frames(
            records,
            mode=SELECTION_MODE,
            z_threshold=Z_THRESHOLD_MM,
            sigma_threshold=SIGMA_THRESHOLD_PX,
        )

        per_sphere_selected[sphere] = selected
        per_sphere_counts[sphere] = len(selected)
        all_selected.extend(selected)

        print(f"[{sphere}] {len(records)} frames, selecting {selection_rule} -> "
              f"{len(selected)} selected")

    total = len(all_selected)
    if total == 0:
        print("\nERROR: No frames selected. Relax the selection threshold.")
        sys.exit(1)

    print(f"\nTotal selected: {total} images")

    # -- 4. Sanity summaries --
    print_sanity_summaries(all_selected, auth_cal, per_sphere_selected)

    # -- 5. Create output directory and copy images --
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    img_dir = OUTPUT_DIR / 'selected_images'
    img_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0
    for r in all_selected:
        src = STAGE1_ROOT / r.sphere_folder / 'processed_images' / r.image_filename
        dst = img_dir / r.image_filename
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"  WARNING: missing image {src}")
            missing += 1

    print(f"Copied {copied} images to selected_images/"
          + (f" ({missing} missing)" if missing else ""))

    # -- 6. Write sharp_crops.csv (training-GUI-ready) --
    write_sharp_crops_csv(OUTPUT_DIR / 'sharp_crops.csv', all_selected)
    print(f"Wrote sharp_crops.csv ({total} rows)")

    # -- 7. Write traceability CSV --
    write_traceability_csv(
        OUTPUT_DIR / 'selected_base_frames.csv',
        all_selected, selection_rule,
        f'{AUTHORITATIVE_SPHERE}/calibration_summary.yaml',
    )
    print(f"Wrote selected_base_frames.csv ({total} rows)")

    # -- 8. Write summary and config YAMLs --
    write_dataset_summary(
        OUTPUT_DIR / 'dataset_summary.yaml',
        auth_cal, AUTHORITATIVE_SPHERE, selection_rule,
        per_sphere_counts, total,
    )
    print("Wrote dataset_summary.yaml")

    write_experiment_config(OUTPUT_DIR / 'experiment_config.yaml', auth_cal)
    print("Wrote experiment_config.yaml")

    # -- Done --
    print(f"\n{'='*50}")
    print(f"Stage 2 complete")
    print(f"{'='*50}")
    print(f"\nOutputs in: {OUTPUT_DIR}")
    print(f"  selected_images/         {copied} files")
    print(f"  sharp_crops.csv          {total} rows (training-GUI-ready)")
    print(f"  selected_base_frames.csv {total} rows (full traceability)")
    print(f"  dataset_summary.yaml")
    print(f"  experiment_config.yaml")
    print(f"\nTo use with training GUI:")
    print(f"  1. Update optical_config.yaml with values from experiment_config.yaml")
    print(f"  2. Point synthetic generator sharp image dir at: {img_dir}")
    print(f"  3. Set sharp_crops_csv to: {OUTPUT_DIR / 'sharp_crops.csv'}")
    print()


if __name__ == '__main__':
    run_stage2()
