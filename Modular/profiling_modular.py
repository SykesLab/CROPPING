# profiling_modular.py
#
# Timing aggregation and profiling output utilities.
# Used by both per-folder and global pipelines.

import json
from pathlib import Path


def aggregate_timings(timing_list, label=""):
    """
    Aggregate timing dicts from multiple workers and print summary.
    
    Args:
        timing_list: List of timing dicts from workers
        label: Label for print output
        
    Returns:
        dict with summed totals
    """
    if not timing_list:
        return {}
    
    totals = {}
    for t in timing_list:
        for k, v in t.items():
            totals[k] = totals.get(k, 0.0) + v
    
    print(f"\n  [TIMING {label}]")
    for k, v in sorted(totals.items()):
        if k in ("n_frames", "n_cines", "n_outputs"):
            print(f"    {k}: {int(v)}")
        else:
            print(f"    {k}: {v:.2f}s")
    
    return totals


def print_global_summary(analysis_timing, output_timing, phase_times=None):
    """
    Print global timing summary at end of pipeline.
    
    Args:
        analysis_timing: dict of analysis phase timings
        output_timing: dict of output phase timings
        phase_times: optional dict with phase1_sec, phase2_sec, phase3_sec
    """
    print("\n" + "=" * 50)
    print("GLOBAL TIMING SUMMARY")
    print("=" * 50)
    
    print("\n[ANALYSIS PHASE]")
    for k, v in sorted(analysis_timing.items()):
        if k in ("n_frames", "n_cines"):
            print(f"  {k}: {int(v)}")
        else:
            print(f"  {k}: {v:.2f}s")
    
    print("\n[OUTPUT PHASE]")
    for k, v in sorted(output_timing.items()):
        if k in ("n_outputs",):
            print(f"  {k}: {int(v)}")
        else:
            print(f"  {k}: {v:.2f}s")
    
    if phase_times:
        print(f"\n[PHASE TOTALS]")
        if "phase1_sec" in phase_times:
            print(f"  Phase 1 (analysis):    {phase_times['phase1_sec']:.2f}s")
        if "phase2_sec" in phase_times:
            print(f"  Phase 2 (calibration): {phase_times['phase2_sec']:.3f}s")
        if "phase3_sec" in phase_times:
            print(f"  Phase 3 (outputs):     {phase_times['phase3_sec']:.2f}s")


def save_profile_json(output_root, filename, data):
    """
    Save profiling data to JSON file.
    
    Args:
        output_root: Output directory path
        filename: JSON filename (e.g., "profiling_global.json")
        data: dict to serialize
    """
    prof_path = Path(output_root) / filename
    with open(prof_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[PROFILE] Saved → {prof_path}")


def init_output_timing():
    """Return a fresh output timing dict."""
    return {
        "crop": 0.0,
        "darkness_plot": 0.0,
        "overlay_plot": 0.0,
        "imwrite": 0.0,
    }


def init_output_timing_with_count():
    """Return a fresh output timing dict with n_outputs counter."""
    return {
        "crop": 0.0,
        "darkness_plot": 0.0,
        "overlay_plot": 0.0,
        "imwrite": 0.0,
        "n_outputs": 0,
    }


def accumulate_timings(target, source):
    """
    Add source timing values to target dict.
    Only adds keys that exist in target.
    
    Args:
        target: dict to accumulate into
        source: dict with values to add
    """
    for k, v in source.items():
        if k in target:
            target[k] += v
