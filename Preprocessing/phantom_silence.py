"""Silent importer for pyphantom modules.

Centralises all pyphantom imports and suppresses the SDK banner that would
otherwise print on every multiprocessing worker spawn.

All other modules should import cine/utils from here, never directly from pyphantom.

Exports:
    cine: Module for loading .cine files.
    utils: Module containing FrameRange and other utilities.
    ph: Phantom camera object (may be None if no camera connected).
"""

import os
from contextlib import contextmanager
from typing import Any, Generator, Optional


@contextmanager
def suppress_all_output() -> Generator[None, None, None]:
    """Context manager to silence stdout and stderr at the OS level.

    Uses low-level file descriptor manipulation to suppress output from
    C extensions that bypass Python's sys.stdout/stderr.

    Yields:
        None
    """
    # Save original file descriptors
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    # Redirect stdout/stderr to devnull
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)

    try:
        yield
    finally:
        # Restore original file descriptors
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(devnull_fd)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)


# Load pyphantom components silently
with suppress_all_output():
    from pyphantom import cine, utils

    ph: Optional[Any] = None
    try:
        from pyphantom import Phantom
        ph = Phantom(init_camera=False)
    except Exception:
        try:
            ph = Phantom()
        except Exception:
            ph = None
