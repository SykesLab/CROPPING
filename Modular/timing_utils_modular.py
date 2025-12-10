"""Simple wall-clock timer utility."""

import time


class Timer:
    """Wall-clock timer with formatted output.

    Attributes:
        start_time: Unix timestamp when timer was created/reset.

    Example:
        >>> timer = Timer()
        >>> # ... do work ...
        >>> print(f"Elapsed: {timer.elapsed}")
    """

    def __init__(self) -> None:
        """Initialise timer with current time."""
        self.start_time: float = time.time()

    def reset(self) -> None:
        """Reset timer to current time."""
        self.start_time = time.time()

    @property
    def seconds(self) -> float:
        """Return elapsed seconds since last reset."""
        return time.time() - self.start_time

    @property
    def elapsed(self) -> str:
        """Return formatted elapsed time string (s/min/hr)."""
        sec = self.seconds
        if sec < 60:
            return f"{sec:.1f}s"
        elif sec < 3600:
            return f"{sec / 60:.1f} min"
        else:
            return f"{sec / 3600:.1f} hr"
