"""
Shared pyphantom utilities.

Provides silent output suppression and SDK initialisation used by both
the preprocessing cine_io module and the calibration cine_loader.
"""

import os
from contextlib import contextmanager
from typing import Any, Generator, Optional


@contextmanager
def suppress_output() -> Generator[None, None, None]:
    """Suppress stdout/stderr at OS level for C extensions.

    Uses file descriptor manipulation to silence output from pyphantom's
    SDK banner and Phantom() constructor, which print directly from C
    and bypass Python's sys.stdout/stderr.
    """
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)

    try:
        yield
    finally:
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(devnull_fd)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)


def init_pyphantom() -> tuple:
    """Silently import pyphantom and initialise the Phantom SDK.

    Returns:
        (cine_module, utils_module, phantom_instance, available, initialized)
    """
    cine_mod: Optional[Any] = None
    utils_mod: Optional[Any] = None
    phantom: Optional[Any] = None
    available = False
    initialized = False

    try:
        with suppress_output():
            from pyphantom import cine as _cine, utils as _utils
            cine_mod = _cine
            utils_mod = _utils
            available = True

            try:
                from pyphantom import Phantom
                phantom = Phantom(init_camera=False)
                initialized = phantom is not None
            except Exception:
                try:
                    phantom = Phantom()
                    initialized = phantom is not None
                except Exception:
                    phantom = None
    except ImportError:
        pass

    return cine_mod, utils_mod, phantom, available, initialized
