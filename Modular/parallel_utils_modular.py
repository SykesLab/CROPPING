# parallel_utils_modular.py
from multiprocessing import Pool
from tqdm import tqdm


def run_parallel(func, items, desc="Processing", processes=None, safe_mode=False):
    """
    Parallel (or safe single-process) wrapper with a clean TQDM progress bar.

    Parameters
    ----------
    func : callable
        Worker function taking a single argument.
    items : list
        Items to map over.
    desc : str
        Text displayed on the progress bar.
    processes : int or None
        Number of worker processes for multiprocessing.Pool.
        Ignored in safe_mode.
    safe_mode : bool
        If True → run sequentially in the main process (nice for debugging,
        stack traces, and profiling). If False → use multiprocessing.
    """
    total = len(items)
    if total == 0:
        return []

    # -----------------------------
    # SAFE MODE: single-process loop
    # -----------------------------
    if safe_mode:
        results = []
        for item in tqdm(
            items,
            total=total,
            desc=f"{desc} (safe)",
            dynamic_ncols=True,
            smoothing=0.1,
        ):
            results.append(func(item))
        return results

    # -----------------------------
    # FAST MODE: multiprocessing
    # -----------------------------
    with Pool(processes=processes) as pool:
        results = list(
            tqdm(
                pool.imap(func, items),
                total=total,
                desc=desc,
                dynamic_ncols=True,
                smoothing=0.1,
            )
        )

    return results
