"""Launch Training GUI via: python -m Training"""
import sys
from pathlib import Path

_dir = str(Path(__file__).resolve().parent)
_repo = str(Path(__file__).resolve().parent.parent)
for p in (_dir, _repo, str(Path(_repo) / "Preprocessing"), str(Path(_repo) / "Calibration")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _run():
    from training_gui import TrainingGUI
    app = TrainingGUI()
    app.run()


if __name__ == "__main__":
    _run()
