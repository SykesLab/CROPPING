"""run_io — file-system layer for Inference results.

Handles two concepts:

  1. **Sessions** — implicit grouping of single-cine Process+Save events
     by `(date, source_folder)`. Each session lives at
     ``v2_runs/sessions/<YYYY-MM-DD>_<source_folder_name>/`` and
     accumulates as the user processes more cines from the same folder.

  2. **Batch runs** — one folder per Batch button click, at
     ``v2_runs/batch/<YYYY-MM-DD_HHMMSS>_<source_folder_name>/``.

For each saved cine the user can opt to write the raw frame, overlay,
crop, processed image, plus the always-on results CSV + metadata.

Pure I/O — no Tk dependencies. Imported by the App and the SaveDialog.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import yaml


# Default root for inference outputs — relative to the CROPPING repo root.
# After the V1 cleanup + V2 rename, the GUI lives at Inference/, so
# putting outputs at Inference/output keeps inputs and outputs in the
# same place.
DEFAULT_RUN_ROOT_REL = "Inference/output"


# ─── Save options ────────────────────────────────────────────────────

@dataclass
class SaveOptions:
    """What to write to disk on a Save action.

    ``raw_frame``    — full cine frame at the best frame (PNG)
    ``overlay``      — full frame + V1 geometry annotations (PNG)
    ``crop``         — extracted crop region (PNG)
    ``processed``    — exact image fed to the model (PNG)
    ``results_csv``  — append a row to the session/batch CSV (always
                       True for batch; toggleable for single)
    """
    raw_frame: bool = True
    overlay: bool = False
    crop: bool = True
    processed: bool = True
    results_csv: bool = True

    def to_dict(self) -> Dict[str, bool]:
        return {
            "raw_frame": self.raw_frame,
            "overlay": self.overlay,
            "crop": self.crop,
            "processed": self.processed,
            "csv": self.results_csv,
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "SaveOptions":
        d = d or {}
        return cls(
            raw_frame=bool(d.get("raw_frame", True)),
            overlay=bool(d.get("overlay", False)),
            crop=bool(d.get("crop", True)),
            processed=bool(d.get("processed", True)),
            results_csv=bool(d.get("csv", True)),
        )


# ─── Folder name sanitisation ────────────────────────────────────────

_INVALID_FS_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def sanitise_folder_name(raw: str) -> str:
    """Make a string safe to use as a folder name on Windows + POSIX.

    Replaces runs of disallowed characters with underscores, trims
    leading/trailing underscores. Returns 'unnamed' if everything got
    stripped.
    """
    s = _INVALID_FS_CHARS.sub("_", raw).strip("_")
    return s or "unnamed"


# ─── Session resolution ──────────────────────────────────────────────

def session_folder_for(
    cine_path: Path,
    run_root: Path,
    when: Optional[datetime.datetime] = None,
) -> Path:
    """Return the session folder a Save of ``cine_path`` should target.

    Naming: ``{run_root}/sessions/{YYYY-MM-DD}_{cine_parent_name}/``

    Auto-grouping: cines from the same parent folder, on the same date,
    share a session folder. New day or new parent folder → new session.
    """
    when = when or datetime.datetime.now()
    parent = sanitise_folder_name(cine_path.parent.name or "root")
    date_str = when.strftime("%Y-%m-%d")
    return run_root / "sessions" / f"{date_str}_{parent}"


def batch_folder_for(
    folder_path: Path,
    run_root: Path,
    when: Optional[datetime.datetime] = None,
) -> Path:
    """Return the batch folder for a Batch button run on ``folder_path``."""
    when = when or datetime.datetime.now()
    name = sanitise_folder_name(folder_path.name or "root")
    stamp = when.strftime("%Y-%m-%d_%H%M%S")
    return run_root / "batch" / f"{stamp}_{name}"


# ─── Per-cine subfolder name (timestamped) ───────────────────────────

def per_cine_subfolder(cine_name: str,
                        when: Optional[datetime.datetime] = None) -> str:
    """``sphere0959v_153022`` for cine_name='sphere0959v.cine'."""
    when = when or datetime.datetime.now()
    stem = Path(cine_name).stem
    return f"{sanitise_folder_name(stem)}_{when.strftime('%H%M%S')}"


# ─── Image saving ────────────────────────────────────────────────────

def save_image(path: Path, img: np.ndarray) -> None:
    """Write a numpy image to disk as PNG. Handles uint8 and float."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if img.dtype in (np.float32, np.float64):
        out = np.clip(img * 255, 0, 255).astype(np.uint8)
    else:
        out = img.astype(np.uint8)
    cv2.imwrite(str(path), out)


# ─── Metadata + summary writers ──────────────────────────────────────

def _model_sha256(model_path: Optional[Path]) -> Optional[str]:
    """SHA256 prefix of the .pth file (first 16 hex chars). None if
    file unreadable."""
    if model_path is None or not model_path.is_file():
        return None
    h = hashlib.sha256()
    try:
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:
        return None


def build_run_metadata(
    *,
    run_type: str,                       # "session" or "batch"
    when: datetime.datetime,
    engine: Any,
    settings: Dict[str, Any],
    auto_decision: Any,
    cine_path: Optional[Path] = None,
    folder_path: Optional[Path] = None,
    save_options: Optional[SaveOptions] = None,
) -> Dict[str, Any]:
    """Assemble the dict that becomes session_metadata.yaml or
    run_metadata.yaml. Pure function — no I/O."""
    cm = getattr(engine, 'calibration_model', None) if engine else None
    model_path_str = settings.get("model_path", "")
    model_path = Path(model_path_str) if model_path_str else None

    meta: Dict[str, Any] = {
        "run": {
            "type": run_type,
            "started_at": when.isoformat(timespec="seconds"),
            "inference_v2_version": "0.1",
        },
        "model": {
            "path": str(model_path) if model_path else "",
            "filename": model_path.name if model_path else "",
            "sha256": _model_sha256(model_path),
        },
        "calibration": {},
        "deployment": {
            "s_c_px_per_mm": float(settings.get("s_c", 0.0) or 0.0),
            "device": settings.get("device", "cpu"),
        },
        "preprocessing": {
            "flatten_mode_setting": settings.get("flatten_mode", "auto"),
            "flatten_mode_used": (auto_decision.mode
                                    if auto_decision is not None else None),
            "auto_rationale": (auto_decision.rationale
                                 if auto_decision is not None else None),
            "inner_margin_px": int(settings.get("inner_margin_px", 20)),
            "feather_px": int(settings.get("feather_px", 40)),
            "crop_size": int(settings.get("crop_size", 299)),
        },
        "input": {},
        "output": {
            "saved": (save_options.to_dict()
                      if save_options is not None else None),
        },
    }
    if cm is not None:
        meta["calibration"] = {
            "method": cm.method,
            "sha256": cm.sha256()[:16],
            "rho_px_per_mm": float(cm.rho_px_per_mm),
            "sigma_floor_calib_px": (
                float(cm.sigma_floor_calib_px)
                if cm.sigma_floor_calib_px is not None else None),
            "sigma_0_calib_px": (
                float(cm.sigma_0_calib_px)
                if cm.sigma_0_calib_px is not None else None),
            "s_calib_px_per_mm": (
                float(cm.s_calib_px_per_mm)
                if cm.s_calib_px_per_mm is not None else None),
            "sigma_max_trusted_calib_px": float(cm.sigma_max_trusted_calib_px),
            "source": "from_checkpoint",
        }
    else:
        meta["calibration"] = {"source": "legacy_fallback"}
    if cine_path is not None:
        meta["input"]["cine_path"] = str(cine_path)
        meta["input"]["cine_filename"] = cine_path.name
    if folder_path is not None:
        meta["input"]["folder_path"] = str(folder_path)
    return meta


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def append_csv_row(csv_path: Path, row: Dict[str, Any],
                    fieldnames: List[str]) -> None:
    """Append a row, writing the header if file doesn't exist yet."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.is_file()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


# Canonical CSV column order — matches what the App passes in
CSV_COLUMNS = [
    "timestamp",
    "cine_filename",
    "best_frame_idx",
    "sigma_native_px",
    "sigma_model_px",
    "pred_norm",
    "defocus_mm",
    "defocus_uncertainty_mm",
    "bounds_flag",
    "inversion_method",
    "auto_picked_mode",
    "auto_rationale",
    "detection_succeeded",
    "fill_ratio",
    "edge_clearance_px",
    "saved_views",
    "calibration_sha256",
    "model_sha256",
]


def build_csv_row(
    *,
    when: datetime.datetime,
    cine_path: Path,
    results: Dict[str, Any],
    auto_decision: Any,
    save_options: SaveOptions,
    engine: Any,
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    """Construct a single CSV row from a finished inference."""
    cm = getattr(engine, 'calibration_model', None) if engine else None
    model_path_str = settings.get("model_path", "")
    saved = [k for k, v in {
        "raw": save_options.raw_frame,
        "overlay": save_options.overlay,
        "crop": save_options.crop,
        "processed": save_options.processed,
    }.items() if v]
    info = (auto_decision.info if auto_decision is not None else {}) or {}
    return {
        "timestamp": when.strftime("%Y-%m-%d %H:%M:%S"),
        "cine_filename": cine_path.name,
        "best_frame_idx": results.get("frame_idx",
                                       results.get("best_frame_idx", "")),
        "sigma_native_px": f"{results.get('sigma_native', 0):.4f}",
        "sigma_model_px": f"{results.get('sigma_model', 0):.4f}",
        "pred_norm": f"{results.get('pred_norm', 0):.4f}",
        "defocus_mm": f"{results.get('defocus_mm', 0):.4f}",
        "defocus_uncertainty_mm": f"{results.get('defocus_uncertainty_mm', 0):.4f}",
        "bounds_flag": str(results.get("bounds_flag", "IN_RANGE")),
        "inversion_method": str(results.get("inversion_method", "")),
        "auto_picked_mode": (auto_decision.mode if auto_decision else ""),
        "auto_rationale": (auto_decision.rationale if auto_decision else ""),
        "detection_succeeded": info.get("detection_succeeded", ""),
        "fill_ratio": (f"{info['fill_ratio']:.4f}"
                        if "fill_ratio" in info and info["fill_ratio"] is not None
                        else ""),
        "edge_clearance_px": (f"{info['edge_clearance_px']:.1f}"
                                if "edge_clearance_px" in info
                                and info["edge_clearance_px"] is not None
                                else ""),
        "saved_views": ",".join(saved),
        "calibration_sha256": (cm.sha256()[:16] if cm is not None else ""),
        "model_sha256": _model_sha256(Path(model_path_str)) if model_path_str else "",
    }


# ─── Human-readable summary ──────────────────────────────────────────

def write_session_summary(session_dir: Path) -> None:
    """Regenerate summary.txt from the session's results.csv.

    Called after every successful Save. Reads the CSV, computes
    aggregate stats, writes a human-readable text file.
    """
    csv_path = session_dir / "results.csv"
    if not csv_path.is_file():
        return
    rows = list(csv.DictReader(open(csv_path, "r", encoding="utf-8")))
    if not rows:
        return
    metadata_path = session_dir / "session_metadata.yaml"
    meta = {}
    if metadata_path.is_file():
        try:
            meta = yaml.safe_load(open(metadata_path, "r", encoding="utf-8")) or {}
        except Exception:
            pass

    n = len(rows)
    flag_counts: Dict[str, int] = {}
    in_range_z: List[float] = []
    auto_modes: Dict[str, int] = {}
    for r in rows:
        f = r.get("bounds_flag", "IN_RANGE") or "IN_RANGE"
        flag_counts[f] = flag_counts.get(f, 0) + 1
        try:
            z = float(r.get("defocus_mm", "nan"))
            if f == "IN_RANGE":
                in_range_z.append(z)
        except ValueError:
            pass
        m = r.get("auto_picked_mode", "") or "?"
        auto_modes[m] = auto_modes.get(m, 0) + 1

    lines = []
    lines.append(f"SESSION: {session_dir.name}")
    if "run" in meta:
        lines.append(f"Started: {meta['run'].get('started_at', '?')}")
    lines.append(f"Last save: {rows[-1].get('timestamp', '?')}  "
                 f"({n} cines saved so far)")
    lines.append("")
    if "model" in meta:
        sha = meta["model"].get("sha256", "?")
        lines.append(f"Model:        {meta['model'].get('filename', '?')}  "
                      f"(sha={sha})")
    if "calibration" in meta and meta["calibration"]:
        cal = meta["calibration"]
        if cal.get("method"):
            extras = (f"sigma_floor={cal.get('sigma_floor_calib_px', '?')}"
                       if cal.get("method") in ("quadrature", "hybrid")
                       else f"sigma_0={cal.get('sigma_0_calib_px', '?')}")
            lines.append(f"Calibration:  {cal['method']}, "
                          f"rho={cal.get('rho_px_per_mm', '?')}, {extras}")
    if "deployment" in meta:
        lines.append(f"s_c:          {meta['deployment'].get('s_c_px_per_mm', '?')} px/mm")
    lines.append("")

    lines.append("Bounds flag distribution:")
    for flag, c in sorted(flag_counts.items()):
        pct = 100.0 * c / n
        lines.append(f"    {flag:14s} {c:>4d}  ({pct:.1f}%)")
    lines.append("")

    if in_range_z:
        arr = np.array(in_range_z)
        lines.append("Defocus statistics (IN_RANGE only):")
        lines.append(f"    Count:   {len(arr)}")
        lines.append(f"    Mean:    {arr.mean():.3f} mm")
        lines.append(f"    Median:  {float(np.median(arr)):.3f} mm")
        lines.append(f"    Std:     {arr.std():.3f} mm")
        lines.append(f"    Range:   {arr.min():.3f} to {arr.max():.3f} mm")
        lines.append("")

    lines.append("Auto-preprocess decisions:")
    for mode, c in sorted(auto_modes.items()):
        lines.append(f"    {mode:22s} {c:>4d}")
    lines.append("")

    lines.append("Cines processed:")
    for r in rows:
        ts = r.get("timestamp", "?").split(" ")[-1]
        f_ = r.get("bounds_flag", "?")
        try:
            z = float(r.get("defocus_mm", "nan"))
            z_str = f"{z:+7.3f} mm"
        except ValueError:
            z_str = "    ? mm"
        flag_note = "" if f_ == "IN_RANGE" else f"   <- {f_}"
        lines.append(f"    {r.get('cine_filename', '?'):30s} "
                      f"{ts}  z={z_str}{flag_note}")

    out_path = session_dir / "summary.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_single_result_text(
    target_dir: Path,
    cine_path: Path,
    results: Dict[str, Any],
    auto_decision: Any,
    engine: Any,
    settings: Dict[str, Any],
    save_options: SaveOptions,
) -> None:
    """Write the per-cine result.txt at the per-cine subfolder level."""
    cm = getattr(engine, 'calibration_model', None) if engine else None
    z = results.get("defocus_mm", 0.0)
    unc = results.get("defocus_uncertainty_mm", 0.0) or 0.0
    flag = results.get("bounds_flag", "IN_RANGE")
    sigma_n = results.get("sigma_native", 0.0)
    sigma_m = results.get("sigma_model", 0.0)
    pred = results.get("pred_norm", 0.0)
    best_idx = results.get("frame_idx", "?")

    when_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"DEFOCUS INFERENCE - {when_str}",
        "=" * 50,
        "",
        f"Cine:           {cine_path.name}",
        f"Best frame:     {best_idx}",
    ]
    if cm is not None:
        sha = cm.sha256()[:16]
        lines.append(f"Calibration:    {cm.method}  "
                      f"rho={cm.rho_px_per_mm:.4f}, "
                      f"sigma_floor={cm.sigma_floor_calib_px or 0:.4f}  "
                      f"(sha={sha})")
    if auto_decision is not None:
        lines.append(f"Preprocessing:  {auto_decision.mode}")
        lines.append(f"                {auto_decision.rationale}")
    lines.extend([
        "",
        "Result:",
        f"    Defocus:    {z:.3f}{f' +/- {unc:.3f}' if unc > 0 else ''} mm",
        f"    Bounds:     {flag}",
        f"    sigma_native: {sigma_n:.4f} px",
        f"    sigma_model:  {sigma_m:.4f} px",
        f"    pred_norm:    {pred:.4f}",
        "",
        "Saved:",
    ])
    saved_map = save_options.to_dict()
    for k, v in saved_map.items():
        lines.append(f"    {'[x]' if v else '[ ]'} {k}")
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "result.txt").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")


# ─── High-level "save now" entry points ──────────────────────────────

def save_single_cine(
    *,
    cine_path: Path,
    results: Dict[str, Any],
    auto_decision: Any,
    engine: Any,
    settings: Dict[str, Any],
    save_options: SaveOptions,
    run_root: Path,
    raw_frame: Optional[np.ndarray] = None,
    overlay_frame: Optional[np.ndarray] = None,
    crop: Optional[np.ndarray] = None,
    processed: Optional[np.ndarray] = None,
    when: Optional[datetime.datetime] = None,
) -> Path:
    """Save a single processed cine into its session folder.

    Returns the per-cine subfolder path that was written.
    """
    when = when or datetime.datetime.now()
    session_dir = session_folder_for(cine_path, run_root, when)
    cine_subfolder = session_dir / "per_cine" / per_cine_subfolder(
        cine_path.name, when)

    # Write images per options
    if save_options.raw_frame and raw_frame is not None:
        save_image(cine_subfolder / "raw_frame.png", raw_frame)
    if save_options.overlay and overlay_frame is not None:
        save_image(cine_subfolder / "overlay.png", overlay_frame)
    if save_options.crop and crop is not None:
        save_image(cine_subfolder / "crop.png", crop)
    if save_options.processed and processed is not None:
        save_image(cine_subfolder / "processed.png", processed)

    # Per-cine result.txt
    write_single_result_text(
        cine_subfolder, cine_path, results, auto_decision,
        engine, settings, save_options)

    # Session-level metadata (write once on first save)
    session_meta_path = session_dir / "session_metadata.yaml"
    if not session_meta_path.is_file():
        write_yaml(session_meta_path, build_run_metadata(
            run_type="session", when=when,
            engine=engine, settings=settings,
            auto_decision=auto_decision,
            cine_path=cine_path,
            save_options=save_options,
        ))

    # Append CSV row
    if save_options.results_csv:
        row = build_csv_row(
            when=when, cine_path=cine_path,
            results=results, auto_decision=auto_decision,
            save_options=save_options, engine=engine, settings=settings,
        )
        append_csv_row(session_dir / "results.csv", row, CSV_COLUMNS)

    # Regenerate summary.txt + bounds-flag plot (cheap, useful glance)
    write_session_summary(session_dir)
    try:
        write_bounds_flag_plot(session_dir)
    except Exception:
        pass

    return cine_subfolder


def write_bounds_flag_plot(target_dir: Path) -> Optional[Path]:
    """Read target_dir/results.csv and write bounds_flag_distribution.png.

    Returns the written path, or None if the CSV is missing/empty or
    matplotlib isn't available. Safe to call after every batch.
    """
    csv_path = target_dir / "results.csv"
    if not csv_path.is_file():
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    rows = list(csv.DictReader(open(csv_path, "r", encoding="utf-8")))
    if not rows:
        return None
    counts: Dict[str, int] = {}
    for r in rows:
        f = r.get("bounds_flag", "IN_RANGE") or "IN_RANGE"
        counts[f] = counts.get(f, 0) + 1
    # Stable order — IN_RANGE first
    canonical_order = ["IN_RANGE", "BELOW_FLOOR", "SATURATED"]
    extras = [k for k in counts if k not in canonical_order]
    flags = [k for k in canonical_order if k in counts] + sorted(extras)
    values = [counts[k] for k in flags]
    colours = {
        "IN_RANGE": "#4A9E4A",
        "BELOW_FLOOR": "#E8833A",
        "SATURATED": "#C0544E",
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(len(flags)), values,
                   color=[colours.get(f, "#5C5C5C") for f in flags],
                   edgecolor="white", linewidth=1.5)
    total = sum(values)
    for bar, v, f in zip(bars, values, flags):
        pct = 100.0 * v / total
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                f"{v}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(flags)))
    ax.set_xticklabels(flags, fontsize=10)
    ax.set_ylabel("Frame count", fontsize=11)
    ax.set_ylim(top=max(values) * 1.25 if values else 1)
    ax.set_title(f"Bounds Flag Distribution  (n={total})",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    out = target_dir / "bounds_flag_distribution.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def init_batch_run(
    *,
    folder_path: Path,
    engine: Any,
    settings: Dict[str, Any],
    save_options: SaveOptions,
    run_root: Path,
    when: Optional[datetime.datetime] = None,
) -> Path:
    """Create the batch folder and write its run_metadata.yaml.

    Returns the batch folder path; caller appends rows + saves images
    inside ``per_cine/`` as cines are processed.
    """
    when = when or datetime.datetime.now()
    target = batch_folder_for(folder_path, run_root, when)
    target.mkdir(parents=True, exist_ok=True)
    write_yaml(target / "run_metadata.yaml", build_run_metadata(
        run_type="batch", when=when,
        engine=engine, settings=settings,
        auto_decision=None,                # per-cine decisions logged in CSV
        folder_path=folder_path,
        save_options=save_options,
    ))
    return target
