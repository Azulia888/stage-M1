"""
pipeline_timer.py — shared wall-clock anchor for a pipeline run.

`PipelineTimer` is reset once, right when the worker thread launches the
pipeline. From that moment on, the console log (which timestamps every
line with the time elapsed since launch) and anything else that wants to
report timing read the same clock, so the numbers reported across a run
stay consistent with each other.
"""

from __future__ import annotations

import time


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as `H:MM:SS` (or `M:SS` under an hour)."""
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class PipelineTimer:
    """Process-wide elapsed-time clock for the currently running pipeline."""

    _start: float | None = None

    @classmethod
    def start(cls) -> None:
        """Reset the clock to zero. Call once, right as the pipeline launches."""
        cls._start = time.monotonic()

    @classmethod
    def elapsed(cls) -> float:
        """Seconds elapsed since `start()` was called (0 if never started)."""
        if cls._start is None:
            return 0.0
        return time.monotonic() - cls._start

    @classmethod
    def elapsed_str(cls) -> str:
        return format_duration(cls.elapsed())
