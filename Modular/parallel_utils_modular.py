# parallel_utils_modular.py
from multiprocessing import Pool
from tqdm import tqdm

def run_parallel(func, items, desc="Processing", processes=None):
    """
    Multiprocessing wrapper with a clean TQDM progress bar.
    - func: worker function
    - items: list of items for mapping
    - desc: text displayed on the progress bar
    - processes: optional override for number of worker processes
    """
    total = len(items)
    if total == 0:
        return []

    with Pool(processes=processes) as pool:
        # imap yields results lazily and preserves order → works perfectly with tqdm
        results = list(
            tqdm(
                pool.imap(func, items),
                total=total,
                desc=desc,
                dynamic_ncols=True,
                smoothing=0.1
            )
        )

    return results
