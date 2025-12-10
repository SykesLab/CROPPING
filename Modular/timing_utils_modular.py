"""Simple wall-clock timer utility."""

import time


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time string.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted string like "2h 15m 30s" or "45m 12s" or "8.5s".
    """
    if seconds < 60:
        return f"{seconds:.1f}s"

    total_secs = int(seconds)
    hours = total_secs // 3600
    mins = (total_secs % 3600) // 60
    secs = total_secs % 60

    if hours > 0:
        return f"{hours}h {mins}m {secs}s"
    else:
        return f"{mins}m {secs}s"


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
        """Return formatted elapsed time string (e.g. '2h 15m 30s')."""
        return format_time(self.seconds)
