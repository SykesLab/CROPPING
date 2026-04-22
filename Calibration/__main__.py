"""Launch Calibration GUI via: python -m Calibration"""
import sys
from pathlib import Path

_dir = str(Path(__file__).resolve().parent)
_repo = str(Path(__file__).resolve().parent.parent)
for p in (_dir, _repo, str(Path(_repo) / "Preprocessing")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _run():
    from calibration_gui import main
    main()


if __name__ == "__main__":
    _run()
