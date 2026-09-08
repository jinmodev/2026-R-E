"""Run the Robot Path Lab desktop application from the repository root."""

from __future__ import annotations

import sys
from pathlib import Path


SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robot_planner.gui import main  # noqa: E402


if __name__ == "__main__":
    main()

