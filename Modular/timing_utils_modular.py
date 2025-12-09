# timing_utils_modular.py
import time

class Timer:
    def __init__(self):
        self.start_time = time.time()

    def reset(self):
        self.start_time = time.time()

    @property
    def elapsed(self):
        sec = time.time() - self.start_time
        if sec < 60:
            return f"{sec:.1f}s"
        elif sec < 3600:
            return f"{sec/60:.1f} min"
        else:
            return f"{sec/3600:.1f} hr"
