"""
pipeline_worker.py — runs VisionModule.run / runURL on a background QThread
and forwards every print() the tools make to the GUI as a Qt signal, so the
console log box updates live while the pipeline runs.

Every forwarded line is timestamped with the time elapsed since the
pipeline started, and the worker reports the total run time when it's done.
"""

from __future__ import annotations

import sys
import traceback

from PySide6.QtCore import QThread, Signal

from pipeline_timer import PipelineTimer
from vision_module import VisionModule


class _EmitStream:
    """A minimal stdout/stderr replacement that timestamps each complete
    line (time elapsed since the pipeline started) and forwards it to a
    signal, so the console log box updates live while the pipeline runs.

    print() can call write() more than once per logical message (e.g. once
    for the text, once for the newline), so writes are buffered until a
    full line is available before a timestamp is attached and emitted.
    """

    def __init__(self, signal):
        self._signal = signal
        self._buffer = ""

    def write(self, text: str) -> None:
        if not text:
            return
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit_line(line)

    def flush(self) -> None:
        if self._buffer:
            self._emit_line(self._buffer)
            self._buffer = ""

    def _emit_line(self, line: str) -> None:
        self._signal.emit(f"[{PipelineTimer.elapsed_str()}] {line}\n")


class PipelineWorker(QThread):
    log_line = Signal(str)
    total_time = Signal(str)  # emitted with the formatted total run time
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
        PipelineTimer.start()
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
            total = PipelineTimer.elapsed_str()
            print(f"Pipeline finished in {total}.")
            stream.flush()
            self.total_time.emit(total)
            self.finished_ok.emit(module.data)
        except Exception as exc:
            self.log_line.emit(traceback.format_exc())
            total = PipelineTimer.elapsed_str()
            print(f"Pipeline aborted after {total}.")
            stream.flush()
            self.total_time.emit(total)
            self.finished_error.emit(str(exc))
        finally:
            stream.flush()
            sys.stdout, sys.stderr = old_out, old_err
