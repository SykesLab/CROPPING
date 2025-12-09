# phantom_silence_modular.py
#
# A single, centralised, fully silent importer for all pyphantom modules.
# Every other file must import cine, utils, and ph ONLY from this file.
#
# This prevents the "phpy: version ..." banner from printing in multiprocessing.

import os
import sys
from contextlib import contextmanager


@contextmanager
def suppress_all_output():
    """Silence ALL stdout/stderr for the duration of the block."""
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


# ==========================================================
# Load ALL pyphantom components silently in ONE place
# ==========================================================
with suppress_all_output():
    from pyphantom import cine, utils
    try:
        from pyphantom import Phantom
        ph = Phantom(init_camera=False)
    except Exception:
        try:
            ph = Phantom()
        except Exception:
            ph = None

# EXPORTED SYMBOLS:
#   cine  → for loading cine files
#   utils → for FrameRange(), etc.
#   ph    → Phantom object (may be None if no camera is connected)
