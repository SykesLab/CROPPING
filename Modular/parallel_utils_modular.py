"""Parallel processing utilities with progress bars."""

from multiprocessing import Pool
from typing import Any, Callable, List, Optional

from tqdm import tqdm


def run_parallel(
    func: Callable[[Any], Any],
    items: List[Any],
    desc: str = "Processing",
    processes: Optional[int] = None,
    safe_mode: bool = False,
) -> List[Any]:
    """Execute function over items with optional parallelisation.

    Args:
        func: Worker function taking a single argument.
        items: List of items to process.
        desc: Description for progress bar.
        processes: Number of worker processes (None = CPU count).
        safe_mode: If True, run sequentially for debugging.

    Returns:
        List of results in same order as items.

    Example:
        >>> results = run_parallel(process_item, items, desc="Analysing")
    """
    total = len(items)
    if total == 0:
        return []

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
