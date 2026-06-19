"""
pipeline_worker.py — runs VisionModule.run / runURL on a background QThread
and forwards every print() the tools make to the GUI as a Qt signal, so the
console log box updates live while the pipeline runs.
"""

from __future__ import annotations

import sys
import traceback

from PySide6.QtCore import QThread, Signal

from vision_module import VisionModule


class _EmitStream:
    """A minimal stdout/stderr replacement that forwards writes to a signal."""

    def __init__(self, signal):
        self._signal = signal

    def write(self, text: str) -> None:
        if text:
            self._signal.emit(text)

    def flush(self) -> None:
        pass


class PipelineWorker(QThread):
    log_line = Signal(str)
    finished_ok = Signal(object)   # emits the finished DataManager
    finished_error = Signal(str)

    def __init__(self, mode: str, value: str, is_video: bool):
        """
        mode: "url" or "local"
        value: the URL or the local file path
        """
        super().__init__()
        self.mode = mode
        self.value = value
        self.is_video = is_video

    def run(self) -> None:
        old_out, old_err = sys.stdout, sys.stderr
        stream = _EmitStream(self.log_line)
        sys.stdout = stream
        sys.stderr = stream
        try:
            module = VisionModule()
            if self.mode == "url":
                module.runURL(self.value, self.is_video)
            else:
                module.run(self.value, "", self.is_video)
            self.finished_ok.emit(module.data)
        except Exception as exc:
            self.log_line.emit(traceback.format_exc())
            self.finished_error.emit(str(exc))
        finally:
            sys.stdout, sys.stderr = old_out, old_err