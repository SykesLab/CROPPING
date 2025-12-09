# timing_utils_modular.py
import time


class Timer:
    """
    Simple wall-clock timer.

    - .seconds → raw float seconds since last reset
    - .elapsed → nicely formatted string (s / min / hr)
    """

    def __init__(self):
        self.start_time = time.time()

    def reset(self):
        self.start_time = time.time()

    @property
    def seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def elapsed(self) -> str:
        sec = self.seconds
        if sec < 60:
            return f"{sec:.1f}s"
        elif sec < 3600:
            return f"{sec/60:.1f} min"
        else:
            return f"{sec/3600:.1f} hr"
