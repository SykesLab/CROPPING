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
import sys
from contextlib import contextmanager
from typing import Any, Generator, Optional


@contextmanager
def suppress_all_output() -> Generator[None, None, None]:
    """Context manager to silence stdout and stderr.

    Yields:
        None
    """
    saved_out = sys.stdout
    saved_err = sys.stderr
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    sys.stderr = devnull
    try:
        yield
    finally:
        sys.stdout = saved_out
        sys.stderr = saved_err
        devnull.close()


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
