"""
run_window.py — "please wait" screen with a live console log while the
pipeline runs in the background, then hands off to the graph window.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPlainTextEdit, QProgressBar, QMessageBox
from PySide6.QtGui import QTextCursor

from pipeline_worker import PipelineWorker
from graph_window import GraphWindow


class RunWindow(QWidget):
    def __init__(self, mode: str, value: str, is_video: bool):
        super().__init__()
        self.setWindowTitle("AFC — Running analysis...")
        self.resize(820, 560)
        self._graph_window = None

        layout = QVBoxLayout(self)

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
        self.worker.finished_ok.connect(self._on_success)
        self.worker.finished_error.connect(self._on_error)
        self.worker.start()

    def _append_log(self, text: str) -> None:
        self.console.moveCursor(QTextCursor.End)
        self.console.insertPlainText(text)
        self.console.moveCursor(QTextCursor.End)

    def _on_success(self, data) -> None:
        self.status_label.setText("Analysis complete.")
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self._graph_window = GraphWindow(data)
        self._graph_window.show()
        self.close()

    def _on_error(self, message: str) -> None:
        self.progress.setRange(0, 1)
        QMessageBox.critical(self, "Analysis failed", message)
        self.close()