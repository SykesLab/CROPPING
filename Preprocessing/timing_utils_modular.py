"""Simple wall-clock timer utility."""

import time


def format_time(seconds: float) -> str:
    """Format seconds as human-readable string (e.g. '2h 15m 30s')."""
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
    """Wall-clock timer with formatted output."""

    def __init__(self) -> None:
        self.start_time: float = time.time()

    def reset(self) -> None:
        self.start_time = time.time()

    @property
    def seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def elapsed(self) -> str:
        return format_time(self.seconds)
