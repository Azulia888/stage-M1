"""
main.py — entry point for the AFC desktop GUI.

Run from the project root with:
    python gui/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root (parent of this gui/ folder) importable, so that
# `from vision_module import VisionModule` etc. resolve regardless of the
# current working directory the script was launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtWidgets import QApplication
from launch_window import LaunchWindow


def main() -> None:
    app = QApplication(sys.argv)
    window = LaunchWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()