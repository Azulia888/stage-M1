"""
run_window.py — "please wait" screen with a live console log while the
pipeline runs in the background, then hands off to the graph window.
"""

from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar,
    QMessageBox, QFileDialog,
)
from PySide6.QtGui import QTextCursor

from pipeline_worker import PipelineWorker
from graph_window import GraphWindow

# Where the per-run console log .txt files are written.
_EXPORTS_DIR = Path(__file__).resolve().parent.parent 


class RunWindow(QWidget):
    def __init__(self, mode: str, value: str, is_video: bool):
        super().__init__()
        self.setWindowTitle("AFC — Running analysis...")
        self.resize(820, 560)
        self._graph_window = None

        layout = QVBoxLayout(self)

        # Top row: stretch + total-run-timer label, pinned to the top right.
        # Stays blank while the pipeline is running and is filled in once
        # the run finishes.
        top_row = QHBoxLayout()
        top_row.addStretch()
        self.timer_label = QLabel("")
        self.timer_label.setStyleSheet("font-size: 11pt; color: #888888;")
        top_row.addWidget(self.timer_label)
        layout.addLayout(top_row)

        self.status_label = QLabel("Please wait, the analysis is running...")
        self.status_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate
        layout.addWidget(self.progress)

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet(
            "background-color: #1d1f21; color: #d0d0d0; font-family: monospace;"
        )
        layout.addWidget(self.console)

        self.worker = PipelineWorker(mode, value, is_video)
        self.worker.log_line.connect(self._append_log)
        self.worker.total_time.connect(self._set_total_time)
        self.worker.finished_ok.connect(self._on_success)
        self.worker.finished_error.connect(self._on_error)
        self.worker.start()

    def _append_log(self, text: str) -> None:
        self.console.moveCursor(QTextCursor.End)
        self.console.insertPlainText(text)
        self.console.moveCursor(QTextCursor.End)

    def _set_total_time(self, text: str) -> None:
        self.timer_label.setText(f"Total time: {text}")

    def _on_success(self, data) -> None:
        self.status_label.setText("Analysis complete.")
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self._export_console_log(self._video_name(data))
        self._offer_export(data)
        self._graph_window = GraphWindow(data)
        self._graph_window.show()
        self.close()

    def _offer_export(self, data) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save analysis (optional)", filter="Pickle (*.pkl)"
        )
        if path:
            data.save(path)

    def _on_error(self, message: str) -> None:
        self.progress.setRange(0, 1)
        self._export_console_log(self._video_name())
        QMessageBox.critical(self, "Analysis failed", message)
        self.close()

    def _video_name(self, data=None) -> str:
        """Best-effort video name to use for the console log filename."""
        if data is not None and getattr(data, "originalMedia", None):
            return Path(data.originalMedia).stem or "video"
        raw = self.worker.value
        if self.worker.mode == "local":
            return Path(raw).stem or "video"
        # URL mode with no media resolved yet (e.g. download failed):
        # fall back to a filesystem-safe slug of the URL.
        slug = re.sub(r"[^\w-]+", "_", raw).strip("_")[:60]
        return slug or "video"

    def _export_console_log(self, video_name: str) -> None:
        try:
            _EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = _EXPORTS_DIR / f"{video_name}_console_log.txt"
            out_path.write_text(self.console.toPlainText(), encoding="utf-8")
        except OSError as e:
            # Exporting the log is best-effort and must never crash the run.
            self._append_log(f"[console log export failed: {e}]\n")
