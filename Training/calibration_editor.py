"""Edit (ρ, σ₀) calibration constants in a trained checkpoint.

The trainer stores calibration under ``checkpoint['config']['training']`` as
``rho_direct`` and ``sigma_0``. At inference time the model's raw σ_pred is
mapped to physical depth via ``ẑ = (σ_pred − σ₀) / ρ``. This module lets you
replace (ρ, σ₀) with new values — either by hand or by absorbing a post-hoc
linear fit ``ẑ_corr = a·ẑ + b`` — and saves the result to a new checkpoint
file so the source in ``runs/<run>/checkpoints/`` is never mutated.

Each edited checkpoint carries a ``calibration_history`` list so every edit
can be traced back to the trained-from-scratch snapshot.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch


MIN_SLOPE = 1e-9  # below this magnitude, linear corrections become numerically unstable


class CalibrationError(Exception):
    """Raised when a checkpoint lacks the keys needed for calibration editing."""


@dataclass
class CalibrationSnapshot:
    """One (ρ, σ₀) entry in a checkpoint's calibration_history."""
    rho: float
    sigma_0: float
    source: str          # 'training' | 'manual' | 'fit'
    timestamp: str       # ISO8601, seconds resolution
    note: str = ""

    def to_dict(self) -> dict:
        return {
            'rho': float(self.rho),
            'sigma_0': float(self.sigma_0),
            'source': self.source,
            'timestamp': self.timestamp,
            'note': self.note,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CalibrationSnapshot':
        return cls(
            rho=float(d['rho']),
            sigma_0=float(d['sigma_0']),
            source=d.get('source', 'unknown'),
            timestamp=d.get('timestamp', ''),
            note=d.get('note', ''),
        )


# ── Pure math ────────────────────────────────────────────────────────────

def apply_linear_correction(rho: float, sigma_0: float,
                            a: float, b: float) -> Tuple[float, float]:
    """Bake ẑ_corr = a·ẑ + b into new (ρ, σ₀).

    From ẑ = (σ − σ₀)/ρ and ẑ_corr = a·ẑ + b:
        ẑ_corr = (σ − [σ₀ − b·ρ/a]) / (ρ/a)
    so ρ_new = ρ/a and σ₀_new = σ₀ − b·ρ/a.
    """
    if abs(a) < MIN_SLOPE:
        raise CalibrationError(
            f"slope a={a} is too small — fit is degenerate; correction would make ρ diverge")
    rho_new = rho / a
    sigma_0_new = sigma_0 - (b * rho / a)
    return rho_new, sigma_0_new


def invert_correction(rho_new: float, sigma_0_new: float,
                      rho_old: float, sigma_0_old: float) -> Tuple[float, float]:
    """Given two (ρ, σ₀) pairs, recover the (a, b) that links them.

    Useful for showing the user what correction is implicit in a manual edit.
    """
    if abs(rho_new) < MIN_SLOPE:
        raise CalibrationError("rho_new is zero — cannot invert")
    a = rho_old / rho_new
    b = (sigma_0_old - sigma_0_new) * a / rho_old
    return a, b


def correction_from_fit(slope: float, intercept: float) -> Tuple[float, float]:
    """Convert a pred-vs-true linear fit into the (a, b) that corrects predictions.

    If inference produced predictions that fit ``pred = slope·true + intercept``,
    the corrected predictor ``pred_corrected = a·pred + b`` recovers the truth:
        a = 1/slope,   b = -intercept/slope
    """
    if abs(slope) < MIN_SLOPE:
        raise CalibrationError(
            f"slope={slope} is too small — fit is degenerate; correction ill-defined")
    return 1.0 / slope, -intercept / slope


# ── Checkpoint I/O ───────────────────────────────────────────────────────

def load_checkpoint(path: Path) -> dict:
    """Load a checkpoint with weights_only=False (we need the config dict, not just tensors)."""
    return torch.load(Path(path), map_location='cpu', weights_only=False)


def read_calibration(checkpoint: dict) -> Tuple[float, float]:
    """Return (ρ, σ₀) from checkpoint['config']['training']. Raises if missing."""
    cfg = checkpoint.get('config', {})
    train_cfg = cfg.get('training', {})
    rho = train_cfg.get('rho_direct')
    sigma_0 = train_cfg.get('sigma_0')
    if rho is None:
        raise CalibrationError(
            "Checkpoint lacks config.training.rho_direct — not a direct-mode checkpoint "
            "or it predates calibration-in-config storage")
    return float(rho), float(sigma_0) if sigma_0 is not None else 0.0


def write_calibration(checkpoint: dict, rho: float, sigma_0: float) -> None:
    """Set training.rho_direct/sigma_0 in-place on the loaded checkpoint dict."""
    cfg = checkpoint.setdefault('config', {})
    train_cfg = cfg.setdefault('training', {})
    train_cfg['rho_direct'] = float(rho)
    train_cfg['sigma_0'] = float(sigma_0)


def read_history(checkpoint: dict) -> List[CalibrationSnapshot]:
    raw = checkpoint.get('calibration_history', [])
    return [CalibrationSnapshot.from_dict(d) for d in raw]


def write_history(checkpoint: dict, history: List[CalibrationSnapshot]) -> None:
    checkpoint['calibration_history'] = [s.to_dict() for s in history]


# ── Top-level save ───────────────────────────────────────────────────────

def save_corrected_checkpoint(
    source_checkpoint_path: Path,
    output_path: Path,
    rho_new: float,
    sigma_0_new: float,
    source_label: str,
    note: str = "",
) -> CalibrationSnapshot:
    """Load source, apply (ρ, σ₀), append to history, save to output_path.

    If the source has no ``calibration_history`` yet, seeds one with the
    pre-edit values marked ``source='training'`` so the lineage always
    traces back to the trained snapshot.

    Returns the new snapshot that was appended.
    """
    source_checkpoint_path = Path(source_checkpoint_path)
    output_path = Path(output_path)
    if source_checkpoint_path.resolve() == output_path.resolve():
        raise CalibrationError(
            "source and output paths are the same — refusing to overwrite the source")

    ckpt = load_checkpoint(source_checkpoint_path)
    rho_old, sigma_0_old = read_calibration(ckpt)

    history = read_history(ckpt)
    if not history:
        history.append(CalibrationSnapshot(
            rho=rho_old, sigma_0=sigma_0_old,
            source='training', timestamp=_now_iso(),
            note='Seeded from training config on first edit',
        ))

    write_calibration(ckpt, rho_new, sigma_0_new)
    snapshot = CalibrationSnapshot(
        rho=rho_new, sigma_0=sigma_0_new,
        source=source_label, timestamp=_now_iso(), note=note,
    )
    history.append(snapshot)
    write_history(ckpt, history)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output_path)
    return snapshot


def next_edit_filename(edits_dir: Path, base_stem: str = "dme_best") -> Path:
    """Return the next unused ``<stem>_v<N>.pth`` path inside edits_dir."""
    edits_dir = Path(edits_dir)
    n = 1
    while (edits_dir / f"{base_stem}_v{n}.pth").exists():
        n += 1
    return edits_dir / f"{base_stem}_v{n}.pth"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')
