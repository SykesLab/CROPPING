"""Launch Preprocessing GUI via: python -m Preprocessing"""
import sys
from pathlib import Path

_dir = str(Path(__file__).resolve().parent)
if _dir not in sys.path:
    sys.path.insert(0, _dir)


def run_gui():
    from gui import run_gui as _run
    _run()


if __name__ == "__main__":
    run_gui()
