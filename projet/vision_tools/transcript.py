"""
transcript.py — TranscriptTool: audio transcription via OpenAI Whisper.
"""

from __future__ import annotations

import sys
from pathlib import Path

from data_manager import DataManager
from vision_tools.base import VisionTool, _make_tool_json, WHISPER_MODEL


def _format_srt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


class TranscriptTool(VisionTool):
    TOOL_NAME = "Transcript"
    INPUTS = ["Video"]

    def __init__(self, model: str = WHISPER_MODEL):
        self.model = model

    def run(self, data: DataManager) -> dict | None:
        if not data.isVideo:
            return None

        try:
            import whisper
            import torch
        except ImportError:
            print("TranscriptTool: openai-whisper not installed. Skipping.", file=sys.stderr)
            return None

        try:
            print(f"TranscriptTool: loading Whisper model '{self.model}'...", file=sys.stderr)
            model = whisper.load_model(self.model)
            model = model.to(torch.float32)

            print(f"TranscriptTool: transcribing '{data.originalMedia}'...", file=sys.stderr)
            result = model.transcribe(data.originalMedia, fp16=False)
            transcript = result["text"].strip()

            srt_path = Path(data.originalMedia).with_suffix(".srt")
            segments = result.get("segments", [])
            if segments:
                with open(srt_path, "w", encoding="utf-8") as f:
                    for i, seg in enumerate(segments, 1):
                        f.write(
                            f"{i}\n"
                            f"{_format_srt_ts(seg['start'])} --> {_format_srt_ts(seg['end'])}\n"
                            f"{seg['text'].strip()}\n\n"
                        )

            return _make_tool_json(self.TOOL_NAME, self.INPUTS, transcript)

        except Exception as e:
            print(f"TranscriptTool error: {e}", file=sys.stderr)
            return _make_tool_json(self.TOOL_NAME, self.INPUTS, None, explanation=str(e), has_run=0)