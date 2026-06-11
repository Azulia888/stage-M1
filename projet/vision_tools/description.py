"""
description.py — DescriptionTool: image/video description via Ollama vision model.
"""

from __future__ import annotations

import base64
import bisect
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from data_manager import DataManager
from vision_tools.base import (
    VisionTool, _make_tool_json, _ollama_post, _ollama_response,
    OLLAMA_HOST, OLLAMA_VISION_MODEL, OLLAMA_SYNTH_MODEL,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYNTH_DESC_LIMIT = 300
_SUBTITLE_SUMMARIZE_THRESHOLD = 5000


# ---------------------------------------------------------------------------
# SRT helpers
# ---------------------------------------------------------------------------

@dataclass
class _Frame:
    timestamp_ms: int
    jpeg_bytes: bytes
    subtitle: str


def _parse_srt(srt_path: str) -> list[tuple[int, int, str]]:
    text = Path(srt_path).read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\n{2,}", text.strip())
    subs = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            int(lines[0].strip())
        except ValueError:
            continue
        times = lines[1].split("-->")
        if len(times) != 2:
            continue
        try:
            start_ms = _ts_to_ms(times[0])
            end_ms = _ts_to_ms(times[1])
        except Exception:
            continue
        body = " ".join(l.strip() for l in lines[2:] if l.strip())
        body = re.sub(r"<[^>]+>", "", body).strip()
        subs.append((start_ms, end_ms, body))
    return subs


def _ts_to_ms(ts: str) -> int:
    ts = ts.strip().replace(",", ".")
    h, m, rest = ts.split(":")
    s, ms_raw = rest.split(".")
    ms = int(ms_raw.ljust(3, "0")[:3])
    return int(h) * 3_600_000 + int(m) * 60_000 + int(s) * 1_000 + ms


def _sub_at_ms(subs: list[tuple[int, int, str]], starts: list[int], ms: int) -> str:
    i = bisect.bisect_right(starts, ms) - 1
    if i >= 0 and subs[i][0] <= ms <= subs[i][1]:
        return subs[i][2]
    return ""


def _ms_to_hms(ms: int) -> str:
    s = ms // 1000
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class DescriptionTool(VisionTool):
    TOOL_NAME = "Description"
    INPUTS = ["Image", "Video"]

    def __init__(
        self,
        model: str = OLLAMA_VISION_MODEL,
        synth_model: str = OLLAMA_SYNTH_MODEL,
        host: str = OLLAMA_HOST,
        num_frames: int = 20,
        frame_timeout: int = 180,
        synth_timeout: int = 1000,
    ):
        self.model = model
        self.synth_model = synth_model
        self.host = host
        self.num_frames = num_frames
        self.frame_timeout = frame_timeout
        self.synth_timeout = synth_timeout

    def _describe_image_path(self, image_path: str, prompt: str | None = None) -> str:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        prompt = prompt or "Describe this image in detail."
        result = _ollama_post(self.host, {
            "model": self.model,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
        }, timeout=self.frame_timeout)
        return _ollama_response(result)

    def _extract_frames(self, video_path: str, subs: list, starts: list) -> list[_Frame]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if total_frames <= 0:
            raise RuntimeError(f"Cannot determine frame count for: {video_path}")

        indices = np.linspace(0, total_frames - 1, num=self.num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, bgr = cap.read()
            if not ret:
                continue
            ts_ms = int(idx / video_fps * 1000)
            h, w = bgr.shape[:2]
            if w > 640:
                scale = 640 / w
                bgr = cv2.resize(bgr, (640, int(h * scale)))
            _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
            sub_text = _sub_at_ms(subs, starts, ts_ms) if subs else ""
            frames.append(_Frame(ts_ms, buf.tobytes(), sub_text))

        cap.release()
        return frames

    def _describe_frame(self, frame: _Frame) -> str:
        b64 = base64.b64encode(frame.jpeg_bytes).decode()
        prompt = "Describe what is happening in this video frame concisely."
        if frame.subtitle:
            prompt += f' The subtitle at this moment reads: "{frame.subtitle}"'
        result = _ollama_post(self.host, {
            "model": self.model,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
        }, timeout=self.frame_timeout)
        return _ollama_response(result)

    def _synthesize(self, frame_descs: list[dict], subs: list) -> str:
        lines = []
        for d in frame_descs:
            desc = d["description"]
            if len(desc) > _SYNTH_DESC_LIMIT:
                desc = desc[:_SYNTH_DESC_LIMIT].rsplit(" ", 1)[0] + "..."
            lines.append(f"[{d['timestamp']}] {desc}")
        combined = "\n".join(lines)

        subtitle_section = ""
        if subs:
            seen: set[str] = set()
            transcript_lines = []
            for _, _, text in subs:
                if text and text not in seen:
                    seen.add(text)
                    transcript_lines.append(text)
            transcript = " ".join(transcript_lines)
            if transcript:
                if len(transcript) > _SUBTITLE_SUMMARIZE_THRESHOLD:
                    result = _ollama_post(self.host, {
                        "model": self.synth_model,
                        "prompt": (
                            "Summarize these video subtitles concisely in a few sentences:\n\n"
                            + transcript
                        ),
                        "stream": False,
                    }, timeout=self.synth_timeout)
                    transcript = _ollama_response(result)
                    label = "Subtitle summary"
                else:
                    label = "Subtitle transcript"
                subtitle_section = f"\n\n{label}:\n{transcript}"

        prompt = (
            "You are given timestamped visual descriptions of frames from a video"
            + (", along with its subtitle content" if subtitle_section else "")
            + ". Write a coherent, fluent textual summary of the full video content. "
            "This summary must stand on its own. Be specific and informative.\n\n"
            f"Frame descriptions:\n{combined}"
            f"{subtitle_section}"
        )
        result = _ollama_post(self.host, {
            "model": self.synth_model,
            "prompt": prompt,
            "stream": False,
        }, timeout=self.synth_timeout)
        return _ollama_response(result)

    def run(self, data: DataManager) -> dict | None:
        try:
            if not data.isVideo:
                desc = self._describe_image_path(data.originalMedia)
                return _make_tool_json(self.TOOL_NAME, self.INPUTS, desc)

            srt_path = Path(data.originalMedia).with_suffix(".srt")
            subs: list[tuple[int, int, str]] = []
            starts: list[int] = []
            if srt_path.exists():
                subs = _parse_srt(str(srt_path))
                starts = [s[0] for s in subs]

            frames = self._extract_frames(data.originalMedia, subs, starts)
            print(f"DescriptionTool: {len(frames)} frames to describe.", file=sys.stderr)

            frame_descs = []
            for i, frame in enumerate(frames, 1):
                ts = _ms_to_hms(frame.timestamp_ms)
                print(f"  [{i}/{len(frames)}] {ts} ...", file=sys.stderr)
                try:
                    desc = self._describe_frame(frame)
                except RuntimeError as e:
                    desc = "[description unavailable]"
                    print(f"    WARNING: {e}", file=sys.stderr)
                frame_descs.append({"timestamp": ts, "description": desc, "subtitle": frame.subtitle})

            if not frame_descs:
                raise RuntimeError("No frames could be described.")

            print("DescriptionTool: synthesizing...", file=sys.stderr)
            output = self._synthesize(frame_descs, subs)
            return _make_tool_json(self.TOOL_NAME, self.INPUTS, output)

        except Exception as e:
            print(f"DescriptionTool error: {e}", file=sys.stderr)
            return _make_tool_json(self.TOOL_NAME, self.INPUTS, None, explanation=str(e), has_run=0)