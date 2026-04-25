"""Check A — scale-conversion round-trip arithmetic check.

Synthetic generation walks DOWN the chain
    defocus_mm → sigma_calib_px → sigma_native_px → sigma_model_px → label

Inference walks UP it (label → defocus). They should be exact inverses.

If they're not, predictions silently shift by a constant factor regardless
of how good the model is. Catches: cross-camera scale inverted, wrong
reference resolution, missing scaling factor, mismatch between synthetic
generation's ``crop_size_px`` and inference's actual native crop size.

No images needed — purely arithmetic, runs in milliseconds.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

# ── Path setup so this module is runnable standalone (`python -m ...`) ──
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _module in ("Training", ""):
    _p = str(_REPO_ROOT / _module) if _module else str(_REPO_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml  # noqa: E402

from physics import (  # noqa: E402
    ScalingParams,
    defocus_to_sigma_calib,
    label_to_defocus,
    sigma_calib_to_native,
)


DEFAULT_TEST_DEFOCUSES_MM: Sequence[float] = (
    -12.0, -8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0, 12.0,
)
DEFAULT_TOLERANCE_MM = 0.001


@dataclass
class RoundTripPoint:
    """A single test defocus, with the chain values computed in both directions."""
    defocus_in_mm: float
    sigma_calib_px: float
    sigma_native_px: float
    sigma_model_px: float
    label: float
    sigma_model_recovered_px: float
    sigma_native_recovered_px: float
    defocus_recovered_mm: float
    delta_mm: float
    sign_recovered: bool
    passed: bool


@dataclass
class RoundTripResult:
    """Full round-trip-check result for a config."""
    overall_passed: bool
    points: List[RoundTripPoint]
    config_summary: dict
    diagnostics: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'overall_passed': self.overall_passed,
            'config_summary': self.config_summary,
            'points': [asdict(p) for p in self.points],
            'diagnostics': self.diagnostics,
        }


# ── Loaders ──────────────────────────────────────────────────────────────


def load_config_from_yaml(path: Path) -> dict:
    """Load a training_config.yaml-style file."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config_from_checkpoint(path: Path) -> dict:
    """Pull the config dict out of a .pth checkpoint without loading weights."""
    import torch
    ckpt = torch.load(str(path), map_location='cpu', weights_only=False)
    cfg = ckpt.get('config')
    if cfg is None:
        raise ValueError(f"Checkpoint {path} has no 'config' key")
    return cfg


def load_config_auto(path: Path) -> dict:
    """Auto-dispatch by file extension."""
    p = Path(path)
    if p.suffix.lower() in ('.yaml', '.yml'):
        return load_config_from_yaml(p)
    if p.suffix.lower() in ('.pth', '.pt'):
        return load_config_from_checkpoint(p)
    raise ValueError(f"Unrecognised config file extension: {p}")


# ── The check ────────────────────────────────────────────────────────────


def _build_scaling_params(
    config: dict,
    scale_inference_px_per_mm: Optional[float],
) -> ScalingParams:
    """Construct a ScalingParams from the config dict, mirroring how the
    inference engine sets it up at runtime."""
    training = config.get('training', {}) or {}
    data = config.get('data', {}) or {}
    rho = training.get('rho_direct')
    if rho is None or rho <= 0:
        raise ValueError("config.training.rho_direct missing or non-positive")
    sigma_0 = training.get('sigma_0') or 0.0
    s_calib = training.get('scale_calib_px_per_mm') or 1.0
    # If user didn't override the inference scale, assume same camera as calibration
    s_inf = scale_inference_px_per_mm if scale_inference_px_per_mm else s_calib
    blur_range = data.get('blur_range_px')
    if not blur_range:
        raise ValueError("config.data.blur_range_px missing — needed for max_blur")
    max_blur = float(blur_range[1])
    model_size = data.get('image_size_px')
    if not model_size:
        raise ValueError("config.data.image_size_px missing — needed for model_size")
    return ScalingParams(
        rho=float(rho),
        sigma_0=float(sigma_0),
        s_calib=float(s_calib),
        s_inference=float(s_inf),
        max_blur=float(max_blur),
        model_size=int(model_size),
    )


def round_trip_check(
    config: dict,
    inference_native_size: Optional[int] = None,
    scale_inference_px_per_mm: Optional[float] = None,
    test_defocuses_mm: Sequence[float] = DEFAULT_TEST_DEFOCUSES_MM,
    tolerance_mm: float = DEFAULT_TOLERANCE_MM,
) -> RoundTripResult:
    """Verify synthetic gen's chain and inference's chain are exact inverses.

    Args:
        config: training_config.yaml-style dict (or ``checkpoint['config']``).
        inference_native_size: Native crop size at inference. Defaults to the
            ``training.crop_size_px`` from config (i.e. assume inference uses
            the same native size as synthetic generation). Override to detect
            the resolution-mismatch bug class.
        scale_inference_px_per_mm: Inference camera scale. Defaults to the
            calibration camera scale (same camera). Override to test
            cross-camera scaling.
        test_defocuses_mm: Defocus values to round-trip through the chain.
        tolerance_mm: Max acceptable drift between input and recovered z.
    """
    diagnostics: List[str] = []
    try:
        params = _build_scaling_params(config, scale_inference_px_per_mm)
    except ValueError as e:
        return RoundTripResult(
            overall_passed=False,
            points=[],
            config_summary={'error': str(e)},
            diagnostics=[f"Cannot build ScalingParams: {e}"],
        )

    # Native size: inference's actual crop size. If unset, assume same as
    # synthetic generation's crop_size_px (no resolution mismatch).
    crop_size_synth = (config.get('training', {}) or {}).get('crop_size_px')
    if inference_native_size is None:
        inference_native_size = int(crop_size_synth) if crop_size_synth else int(params.model_size)

    if crop_size_synth and inference_native_size != int(crop_size_synth):
        diagnostics.append(
            f"Native crop size mismatch: synthetic gen used crop_size_px={crop_size_synth}, "
            f"inference is using {inference_native_size}. This is the resolution-mismatch "
            "bug class — Check A will detect the resulting round-trip drift."
        )

    # Use synthetic gen's crop_size_px for the DOWN chain, and inference's
    # native size for the UP chain. They differ when there's a resolution
    # mismatch — that's exactly the asymmetry Check A is supposed to catch.
    synth_native_size = int(crop_size_synth) if crop_size_synth else int(params.model_size)

    points: List[RoundTripPoint] = []
    overall_passed = True

    for z in test_defocuses_mm:
        # Down-chain (synthetic generation perspective):
        #   defocus → sigma_calib → sigma_synth_native → sigma_model → label
        sigma_calib = defocus_to_sigma_calib(z, params.rho, params.sigma_0)
        sigma_synth_native = sigma_calib_to_native(
            sigma_calib, params.s_inference, params.s_calib)
        if synth_native_size > 0:
            sigma_model = sigma_synth_native * (params.model_size / synth_native_size)
        else:
            sigma_model = sigma_synth_native
        label = sigma_model / params.max_blur if params.max_blur > 0 else 0.0

        # Up-chain (inference perspective): uses inference_native_size to
        # convert the model's output back to native, then applies rho_eff to
        # recover defocus.
        defocus_recovered, _clamped, _saturated = label_to_defocus(
            label, params, native_size=inference_native_size,
        )

        # The forward chain operates on signed z; the inverse returns |z|. So
        # for symmetric calibration the "round-trip" is on |z|.
        defocus_in_abs = abs(z)
        delta = abs(defocus_recovered - defocus_in_abs)
        passed = delta < tolerance_mm

        # Sign recovery: synthetic stores signed defocus_mm in metadata, but the
        # blur-only inversion can't recover sign on its own. Mark as "sign
        # recovered" trivially (the chain isn't responsible for sign).
        points.append(RoundTripPoint(
            defocus_in_mm=z,
            sigma_calib_px=sigma_calib,
            sigma_native_px=sigma_synth_native,
            sigma_model_px=sigma_model,
            label=label,
            sigma_model_recovered_px=label * params.max_blur,
            sigma_native_recovered_px=label * params.max_blur * (
                inference_native_size / params.model_size
                if params.model_size > 0 else 1.0),
            defocus_recovered_mm=defocus_recovered,
            delta_mm=delta,
            sign_recovered=True,
            passed=passed,
        ))
        if not passed:
            overall_passed = False

    # Annotate likely cause if it failed
    if not overall_passed:
        diagnostics.append(
            f"Round-trip failed for {sum(1 for p in points if not p.passed)} of "
            f"{len(points)} test defocuses (tolerance {tolerance_mm} mm)."
        )
        if abs(params.s_calib - params.s_inference) > 1e-9:
            diagnostics.append(
                f"Cross-camera scaling active "
                f"(s_calib={params.s_calib:.4g}, s_inference={params.s_inference:.4g}). "
                "Verify the ratio direction matches inference's expectation."
            )
        if crop_size_synth and inference_native_size != int(crop_size_synth):
            diagnostics.append(
                "Resolution mismatch already flagged above — most likely culprit."
            )

    return RoundTripResult(
        overall_passed=overall_passed,
        points=points,
        config_summary={
            'rho_direct': params.rho,
            'sigma_0': params.sigma_0,
            's_calib': params.s_calib,
            's_inference': params.s_inference,
            'max_blur': params.max_blur,
            'model_size': params.model_size,
            'inference_native_size': inference_native_size,
            'training_crop_size_px': crop_size_synth,
            'tolerance_mm': tolerance_mm,
        },
        diagnostics=diagnostics,
    )


# ── CLI ──────────────────────────────────────────────────────────────────


def format_text_report(result: RoundTripResult) -> str:
    """Human-readable text report of a round-trip-check result."""
    lines = []
    status = "PASS" if result.overall_passed else "FAIL"
    lines.append(f"Scale-chain round-trip check: {status}")
    lines.append("")
    lines.append("Configuration:")
    for k, v in result.config_summary.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    if result.diagnostics:
        lines.append("Diagnostics:")
        for d in result.diagnostics:
            lines.append(f"  - {d}")
        lines.append("")
    lines.append(
        "Per-point results "
        "(z_in -> sigma_model -> label -> z_recovered):"
    )
    lines.append(
        f"  {'z_in (mm)':>10}  {'sigma_model':>11}  {'label':>7}  "
        f"{'z_out (mm)':>11}  {'|delta|':>9}  status"
    )
    for p in result.points:
        flag = 'pass' if p.passed else 'FAIL'
        lines.append(
            f"  {p.defocus_in_mm:10.3f}  {p.sigma_model_px:11.4f}  "
            f"{p.label:7.4f}  {p.defocus_recovered_mm:11.4f}  "
            f"{p.delta_mm:9.6f}  {flag}"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scale-chain round-trip arithmetic check (Check A).")
    parser.add_argument(
        'config',
        help="Path to training_config.yaml or .pth checkpoint")
    parser.add_argument(
        '--inference-native-size', type=int, default=None,
        help="Native crop size at inference (default: training.crop_size_px)")
    parser.add_argument(
        '--scale-inference', type=float, default=None,
        help="Inference camera scale px/mm (default: calibration camera scale)")
    parser.add_argument(
        '--tolerance-mm', type=float, default=DEFAULT_TOLERANCE_MM,
        help=f"Max acceptable drift in mm (default: {DEFAULT_TOLERANCE_MM})")
    parser.add_argument(
        '--json', action='store_true',
        help="Emit JSON to stdout instead of human text")
    args = parser.parse_args(argv)

    cfg = load_config_auto(Path(args.config))
    result = round_trip_check(
        cfg,
        inference_native_size=args.inference_native_size,
        scale_inference_px_per_mm=args.scale_inference,
        tolerance_mm=args.tolerance_mm,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(format_text_report(result))
    return 0 if result.overall_passed else 1


if __name__ == '__main__':
    sys.exit(main())
