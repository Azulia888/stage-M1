"""
launch_window.py — first screen: choose a URL (yt-dlp) or a local file,
and whether it is a video or an image.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QRadioButton, QFileDialog, QMessageBox,
)

from run_window import RunWindow

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


class LaunchWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AFC — New analysis")
        self.resize(520, 280)
        self._run_window = None

        self.url_radio = QRadioButton("From a URL (YouTube, TikTok, ...)")
        self.local_radio = QRadioButton("From a local file")
        self.url_radio.setChecked(True)

        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("https://www.youtube.com/watch?v=...")

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("/path/to/video.mp4")
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse)

        self.video_radio = QRadioButton("Video")
        self.image_radio = QRadioButton("Image")
        self.video_radio.setChecked(True)

        run_btn = QPushButton("Run analysis")
        run_btn.clicked.connect(self._run)

        layout = QVBoxLayout(self)
        layout.addWidget(self.url_radio)
        layout.addWidget(self.url_edit)
        layout.addWidget(self.local_radio)

        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit)
        path_row.addWidget(browse_btn)
        layout.addLayout(path_row)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Media type:"))
        type_row.addWidget(self.video_radio)
        type_row.addWidget(self.image_radio)
        layout.addLayout(type_row)

        layout.addStretch()
        layout.addWidget(run_btn)

        self.url_radio.toggled.connect(self._sync_enabled)
        self._sync_enabled()

    def _sync_enabled(self) -> None:
        is_url = self.url_radio.isChecked()
        self.url_edit.setEnabled(is_url)
        self.path_edit.setEnabled(not is_url)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select media file")
        if not path:
            return
        self.path_edit.setText(path)
        self.local_radio.setChecked(True)
        if Path(path).suffix.lower() in _IMAGE_EXTS:
            self.image_radio.setChecked(True)
        else:
            self.video_radio.setChecked(True)

    def _run(self) -> None:
        is_video = self.video_radio.isChecked()

        if self.url_radio.isChecked():
            value = self.url_edit.text().strip()
            mode = "url"
            if not value:
                QMessageBox.warning(self, "Missing URL", "Please enter a URL.")
                return
        else:
            value = self.path_edit.text().strip()
            mode = "local"
            if not value or not Path(value).exists():
                QMessageBox.warning(self, "Invalid path", "Please select a valid local file.")
                return

        self._run_window = RunWindow(mode, value, is_video)
        self._run_window.show()
        self.close()