"""
Parallel processing utilities with progress bars.
"""

import os
import time
from multiprocessing import Pool
from typing import Any, Callable, List, Optional

from tqdm import tqdm


def _worker_init() -> None:
    """Initialize worker process with suppressed stdout/stderr.

    This runs once per worker before any tasks, preventing the
    pyphantom SDK banner from printing on each worker spawn.
    """
    import sys
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    sys.stderr = devnull

    # Import pyphantom while suppressed to trigger its banner silently
    try:
        import pyphantom  # noqa: F401
    except ImportError:
        pass

    # Restore stdout/stderr for actual work
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    devnull.close()


def run_parallel(
    func: Callable[[Any], Any],
    items: List[Any],
    desc: str = "Processing",
    processes: Optional[int] = None,
    safe_mode: bool = False,
    gui_mode: bool = False,
) -> List[Any]:
    """
    Execute function over items with optional parallelisation.

    safe_mode: Run sequentially (single process) for debugging.
    gui_mode: Emit progress markers for GUI instead of tqdm bars.
    """
    total = len(items)
    if total == 0:
        return []

    if gui_mode:
        results = []
        start_time = time.time()
        last_print_pct = -10

        if safe_mode:
            for i, item in enumerate(items):
                results.append(func(item))
                _emit_gui_progress(i + 1, total, desc, start_time, last_print_pct)
                last_print_pct = _update_last_print_pct(i + 1, total, last_print_pct)
        else:
            with Pool(processes=processes, initializer=_worker_init) as pool:
                for i, result in enumerate(pool.imap(func, items)):
                    results.append(result)
                    _emit_gui_progress(i + 1, total, desc, start_time, last_print_pct)
                    last_print_pct = _update_last_print_pct(i + 1, total, last_print_pct)
        return results

    if safe_mode:
        results = []
        for item in tqdm(items, total=total, desc=f"{desc} (safe)", dynamic_ncols=True, smoothing=0.1, unit="item"):
            results.append(func(item))
        return results

    with Pool(processes=processes, initializer=_worker_init) as pool:
        results = list(tqdm(pool.imap(func, items), total=total, desc=desc, dynamic_ncols=True, smoothing=0.1, unit="item"))

    return results


def _emit_gui_progress(current: int, total: int, desc: str, start_time: float, last_print_pct: int) -> None:
    """Emit progress for GUI and print log updates every 10%."""
    progress_pct = int(current / total * 100)
    elapsed = time.time() - start_time

    print(f"__PROGRESS__:{current}:{total}:{desc}")

    if progress_pct >= last_print_pct + 10 or current == total:
        avg_time = elapsed / current
        remaining = int(avg_time * (total - current))
        print(f"  {desc}: {current}/{total} ({progress_pct}%) - {remaining}s remaining")


def _update_last_print_pct(current: int, total: int, last_print_pct: int) -> int:
    """Update the last print percentage threshold."""
    progress_pct = int(current / total * 100)
    if progress_pct >= last_print_pct + 10 or current == total:
        return (progress_pct // 10) * 10
    return last_print_pct
