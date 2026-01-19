"""Timing aggregation and profiling utilities."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from timing_utils_modular import format_time


def aggregate_timings(
    timing_list: List[Dict[str, float]],
    label: str = "",
) -> Dict[str, float]:
    """Aggregate timing dictionaries and print summary."""
    if not timing_list:
        return {}

    totals: Dict[str, float] = {}
    for t in timing_list:
        for k, v in t.items():
            totals[k] = totals.get(k, 0.0) + v

    print(f"\n  [TIMING {label}]")
    for k, v in sorted(totals.items()):
        if k in ("n_frames", "n_cines", "n_outputs"):
            print(f"    {k}: {int(v)}")
        else:
            print(f"    {k}: {format_time(v)}")

    return totals


def print_global_summary(
    analysis_timing: Dict[str, float],
    output_timing: Dict[str, float],
    phase_times: Optional[Dict[str, float]] = None,
) -> None:
    """Print global timing summary."""
    # Calculate total elapsed time
    total_elapsed = 0.0
    if phase_times:
        total_elapsed = (
            phase_times.get("phase1_sec", 0.0) +
            phase_times.get("phase2_sec", 0.0) +
            phase_times.get("phase3_sec", 0.0)
        )

    print("\n" + "=" * 50)
    print("GLOBAL TIMING SUMMARY")
    print("=" * 50)

    if total_elapsed > 0:
        print(f"\n  TOTAL ELAPSED: {format_time(total_elapsed)}")

    print("\n[ANALYSIS PHASE]")
    for k, v in sorted(analysis_timing.items()):
        if k in ("n_frames", "n_cines"):
            print(f"  {k}: {int(v)}")
        else:
            print(f"  {k}: {format_time(v)}")

    print("\n[OUTPUT PHASE]")
    for k, v in sorted(output_timing.items()):
        if k in ("n_outputs",):
            print(f"  {k}: {int(v)}")
        else:
            print(f"  {k}: {format_time(v)}")

    if phase_times:
        print("\n[PHASE TOTALS]")
        if "phase1_sec" in phase_times:
            print(f"  Phase 1 (analysis):    {format_time(phase_times['phase1_sec'])}")
        if "phase2_sec" in phase_times:
            print(f"  Phase 2 (calibration): {format_time(phase_times['phase2_sec'])}")
        if "phase3_sec" in phase_times:
            print(f"  Phase 3 (outputs):     {format_time(phase_times['phase3_sec'])}")


def save_profile_json(
    output_root: Union[str, Path],
    filename: str,
    data: Dict[str, Any],
) -> None:
    """Save profiling data to JSON file."""
    prof_path = Path(output_root) / filename
    with open(prof_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[PROFILE] Saved → {prof_path}")


def init_output_timing() -> Dict[str, float]:
    """Return fresh output timing dict."""
    return {
        "crop": 0.0,
        "darkness_plot": 0.0,
        "overlay_plot": 0.0,
        "imwrite": 0.0,
    }


def init_output_timing_with_count() -> Dict[str, float]:
    """Return fresh output timing dict with counter."""
    return {
        "crop": 0.0,
        "darkness_plot": 0.0,
        "overlay_plot": 0.0,
        "imwrite": 0.0,
        "n_outputs": 0,
    }


def accumulate_timings(
    target: Dict[str, float],
    source: Dict[str, float],
) -> None:
    """Accumulate source values into target dict (only keys that exist in target)."""
    for k, v in source.items():
        if k in target:
            target[k] += v
