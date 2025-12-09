# parallel_utils_modular.py
from multiprocessing import Pool, cpu_count


def run_parallel(func, items, max_workers=None):
    """
    Apply func(item) in parallel over items.
    Returns list of results in the same order.

    max_workers defaults to all CPU cores.
    """
    if max_workers is None:
        max_workers = cpu_count()

    with Pool(processes=max_workers) as pool:
        results = pool.map(func, items)

    return results
