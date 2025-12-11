"""Parallel processing utilities with progress bars."""

import time
from multiprocessing import Pool
from typing import Any, Callable, List, Optional


from tqdm import tqdm


def run_parallel(
    func: Callable[[Any], Any],
    items: List[Any],
    desc: str = "Processing",
    processes: Optional[int] = None,
    safe_mode: bool = False,
    gui_mode: bool = False,
) -> List[Any]:
    """Execute function over items with optional parallelisation.

    Args:
        func: Worker function taking a single argument.
        items: List of items to process.
        desc: Description for progress bar.
        processes: Number of worker processes (None = CPU count).
        safe_mode: If True, run sequentially for debugging.
        gui_mode: If True, print progress updates instead of tqdm bar.

    Returns:
        List of results in same order as items.

    Example:
        >>> results = run_parallel(process_item, items, desc="Analysing")
    """
    total = len(items)
    if total == 0:
        return []

    if gui_mode:
        # GUI mode: emit progress for every item, print log every 10%
        results = []
        start_time = time.time()
        last_print_pct = -10  # So first print happens at 0%
        
        if safe_mode:
            for i, item in enumerate(items):
                results.append(func(item))
                current = i + 1
                progress_pct = int(current / total * 100)
                elapsed = time.time() - start_time
                
                # Emit progress marker (hidden from log, detected by GUI)
                print(f"__PROGRESS__:{current}:{total}:{desc}")
                
                # Print to log every 10%
                if progress_pct >= last_print_pct + 10 or current == total:
                    avg_time = elapsed / current
                    remaining = int(avg_time * (total - current))
                    print(f"  {desc}: {current}/{total} ({progress_pct}%) - {remaining}s remaining")
                    last_print_pct = (progress_pct // 10) * 10
        else:
            with Pool(processes=processes) as pool:
                for i, result in enumerate(pool.imap(func, items)):
                    results.append(result)
                    current = i + 1
                    progress_pct = int(current / total * 100)
                    elapsed = time.time() - start_time
                    
                    # Emit progress marker (hidden from log, detected by GUI)
                    print(f"__PROGRESS__:{current}:{total}:{desc}")
                    
                    # Print to log every 10%
                    if progress_pct >= last_print_pct + 10 or current == total:
                        avg_time = elapsed / current
                        remaining = int(avg_time * (total - current))
                        print(f"  {desc}: {current}/{total} ({progress_pct}%) - {remaining}s remaining")
                        last_print_pct = (progress_pct // 10) * 10
        return results

    if safe_mode:
        # Single-process mode for debugging
        results = []
        for item in tqdm(
            items,
            total=total,
            desc=f"{desc} (safe)",
            dynamic_ncols=True,
            smoothing=0.1,
            unit="item",
        ):
            results.append(func(item))
        return results

    # Multiprocessing mode
    with Pool(processes=processes) as pool:
        results = list(
            tqdm(
                pool.imap(func, items),
                total=total,
                desc=desc,
                dynamic_ncols=True,
                smoothing=0.1,
                unit="item",
            )
        )

    return results
