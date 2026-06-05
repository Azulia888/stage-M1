"""
vision_tool.py — Implementations of all VisionTool subclasses.

Tools backed by existing scripts
---------------------------------
TranscriptTool      → transcribe.py logic (OpenAI Whisper)
KeyFrameExtractionTool → extract_keyframes.py logic (OpenCV)
OCRTool             → ocr.py logic (Ollama vision model)
DescriptionTool     → describe_image.py / describe_video.py logic (Ollama)
MetadataTool        → ffprobe / exiftool / stat fallback

Stub tools (no backing script yet — return None with hasRun=0)
---------------------------------------------------------------
AiDetectionTool, DeepFakeDetectionTool, LipSyncDetectionTool,
WeatherDetectionTool, GeolocationTool, FacialRecognitionTool, NERTool
"""

from __future__ import annotations

import base64
import bisect
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from data_manager import DataManager


# ---------------------------------------------------------------------------
# Helpers shared across tools
# ---------------------------------------------------------------------------

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "qwen3.5:4b")
OLLAMA_SYNTH_MODEL = os.environ.get("OLLAMA_SYNTH_MODEL", OLLAMA_VISION_MODEL)
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")


def _make_tool_json(
    tool_name: str,
    inputs: list[str],
    output: object,
    explanation: str = "",
    confidence: int = 0,
    confidence_explanation: str = "",
    corroborating_tools: list = None,
    has_run: int = 1,
) -> dict:
    return {
        "ToolName": tool_name,
        "Input": inputs,
        "hasRun": has_run,
        "Output": output,
        "Explanation": explanation,
        "Confidence": confidence,
        "ConfidenceExplanation": confidence_explanation,
        "CorroboratingTools": corroborating_tools or [],
    }


def _ollama_post(host: str, payload: dict, timeout: int = 120, think: bool = False) -> dict:
    if not think:
        payload = {**payload, "think": False}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"Ollama HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama connection error: {e.reason}") from e


def _ollama_response(result: dict) -> str:
    text = result.get("response", "").strip()
    if text:
        return text
    thinking = result.get("thinking", "").strip()
    if thinking:
        print("WARNING: 'response' empty, using 'thinking' field.", file=sys.stderr)
        return thinking
    return ""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class VisionTool:
    TOOL_NAME: str = "BaseTool"
    INPUTS: list[str] = []

    def run(self, data: DataManager) -> dict | None:
        """Execute the tool and return a tool-result dict, or None on skip."""
        return None

    def addData(self, data: DataManager) -> None:
        result = self.run(data)
        if result is not None:
            data.addToolResult(result)


# ---------------------------------------------------------------------------
# TranscriptTool
# ---------------------------------------------------------------------------

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

            # Save SRT alongside the media file
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


def _format_srt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# KeyFrameExtractionTool
# ---------------------------------------------------------------------------

class KeyFrameExtractionTool(VisionTool):
    TOOL_NAME = "Keyframes"
    INPUTS = ["Video"]

    def __init__(
        self,
        strategy: str = "both",
        count: int = 10,
        threshold: float = 30.0,
        output_dir: str | None = None,
    ):
        self.strategy = strategy
        self.count = count
        self.threshold = threshold
        self.output_dir = output_dir  # None → auto-derive from media path

    def run(self, data: DataManager) -> dict | None:
        if not data.isVideo:
            return None

        out_dir = self.output_dir or str(
            Path(data.originalMedia).parent / (Path(data.originalMedia).stem + "_keyframes")
        )
        os.makedirs(out_dir, exist_ok=True)

        try:
            cap = cv2.VideoCapture(data.originalMedia)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {data.originalMedia}")

            if self.strategy == "uniform":
                saved = _kf_uniform(cap, self.count, out_dir)
            elif self.strategy == "scene":
                saved = _kf_scene(cap, self.threshold, out_dir)
            else:  # "both"
                saved = _kf_both(cap, self.count, self.threshold, out_dir)

            cap.release()
            print(f"KeyFrameExtractionTool: saved {len(saved)} frames to {out_dir}", file=sys.stderr)
            return _make_tool_json(self.TOOL_NAME, self.INPUTS, saved)

        except Exception as e:
            print(f"KeyFrameExtractionTool error: {e}", file=sys.stderr)
            return _make_tool_json(self.TOOL_NAME, self.INPUTS, None, explanation=str(e), has_run=0)


def _save_frame(frame: np.ndarray, output_dir: str, index: int) -> str:
    path = os.path.join(output_dir, f"frame_{index:05d}.jpg")
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


def _kf_uniform(cap: cv2.VideoCapture, num_frames: int, output_dir: str) -> list[str]:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError("Could not determine frame count.")
    indices = np.linspace(0, total - 1, num=num_frames, dtype=int)
    saved = []
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            saved.append(_save_frame(frame, output_dir, i))
    return saved


def _kf_scene(cap: cv2.VideoCapture, threshold: float, output_dir: str) -> list[str]:
    saved, prev_gray, save_idx = [], None, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = np.mean(np.abs(gray.astype(float) - prev_gray.astype(float)))
            if diff >= threshold:
                saved.append(_save_frame(frame, output_dir, save_idx))
                save_idx += 1
        prev_gray = gray
    if not saved:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if ret:
            saved.insert(0, _save_frame(frame, output_dir, 0))
    return saved


def _kf_both(cap: cv2.VideoCapture, num_frames: int, threshold: float, output_dir: str) -> list[str]:
    scene_frames, prev_gray = [0], None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = np.mean(np.abs(gray.astype(float) - prev_gray.astype(float)))
            if diff >= threshold:
                scene_frames.append(idx)
        prev_gray = gray
        idx += 1
    if len(scene_frames) <= num_frames:
        chosen = scene_frames
    else:
        indices = np.linspace(0, len(scene_frames) - 1, num=num_frames, dtype=int)
        chosen = [scene_frames[i] for i in indices]
    saved = []
    for save_idx, frame_pos in enumerate(chosen):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if ret:
            saved.append(_save_frame(frame, output_dir, save_idx))
    return saved


# ---------------------------------------------------------------------------
# MetadataTool
# ---------------------------------------------------------------------------

class MetadataTool(VisionTool):
    TOOL_NAME = "Metadata Gatherer"
    INPUTS = ["Image", "Video"]

    def run(self, data: DataManager) -> dict | None:
        path = Path(data.originalMedia)
        metadata_lines = []

        # 1. Try ffprobe (videos)
        if data.isVideo:
            try:
                proc = subprocess.run(
                    [
                        "ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_format", "-show_streams", str(path),
                    ],
                    capture_output=True, text=True, timeout=30,
                )
                if proc.returncode == 0:
                    info = json.loads(proc.stdout)
                    fmt = info.get("format", {})
                    tags = fmt.get("tags", {})
                    duration = float(fmt.get("duration", 0))
                    metadata_lines.append(f"Duration: {duration:.1f}s")
                    if tags.get("creation_time"):
                        metadata_lines.append(f"Creation time: {tags['creation_time']}")
                    for stream in info.get("streams", []):
                        if stream.get("codec_type") == "video":
                            metadata_lines.append(
                                f"Video: {stream.get('codec_name')} "
                                f"{stream.get('width')}x{stream.get('height')} "
                                f"@ {stream.get('r_frame_rate')} fps"
                            )
                        elif stream.get("codec_type") == "audio":
                            metadata_lines.append(
                                f"Audio: {stream.get('codec_name')} "
                                f"{stream.get('sample_rate')}Hz "
                                f"{stream.get('channels')}ch"
                            )
            except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
                pass

        # 2. Try exiftool (images and fallback for video)
        try:
            proc = subprocess.run(
                ["exiftool", "-json", str(path)],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode == 0:
                exif = json.loads(proc.stdout)
                if exif:
                    for key in ("DateTimeOriginal", "CreateDate", "ModifyDate",
                                "GPSLatitude", "GPSLongitude", "Make", "Model",
                                "ImageWidth", "ImageHeight", "MIMEType"):
                        val = exif[0].get(key)
                        if val:
                            metadata_lines.append(f"{key}: {val}")
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
            pass

        # 3. Always add file stats
        try:
            stat = path.stat()
            metadata_lines.append(f"File size: {stat.st_size} bytes")
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
            metadata_lines.append(f"Modified: {mtime}")
        except OSError:
            pass

        # 4. Load side-car metadata file if provided
        if data.metadata_path:
            try:
                sidecar = Path(data.metadata_path).read_text(encoding="utf-8", errors="replace").strip()
                if sidecar:
                    metadata_lines.append(f"Sidecar: {sidecar}")
            except OSError:
                pass

        output = "\n".join(metadata_lines) if metadata_lines else "No metadata found."
        return _make_tool_json(self.TOOL_NAME, self.INPUTS, output)


# ---------------------------------------------------------------------------
# OCRTool
# ---------------------------------------------------------------------------

class OCRTool(VisionTool):
    TOOL_NAME = "OCR"
    INPUTS = ["Image", "Video", "Keyframes"]

    def __init__(self, model: str = OLLAMA_VISION_MODEL, host: str = OLLAMA_HOST):
        self.model = model
        self.host = host

    def _ocr_image_path(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        prompt = (
            "Extract all the text from this image. If the text isn't in English, translate it first. "
            "Output ONLY the extracted text. Do not include any introductory phrases, "
            "commentary, or markdown formatting."
        )
        result = _ollama_post(self.host, {
            "model": self.model,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
        }, timeout=120)
        text = _ollama_response(result).strip()
        # Strip accidental markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
        return text

    def run(self, data: DataManager) -> dict | None:
        # Collect images to OCR: keyframes (for video) or the media itself (for images)
        sources: list[str] = []
        if data.isVideo:
            sources = data.keyframes or []
        else:
            sources = [data.originalMedia]

        if not sources:
            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS, None,
                explanation="No images available for OCR.", has_run=0,
            )

        seen: set[str] = set()
        unique_lines: list[str] = []

        for img_path in sources:
            try:
                text = self._ocr_image_path(img_path)
                for line in text.splitlines():
                    line = line.strip()
                    if line and line not in seen:
                        seen.add(line)
                        unique_lines.append(line)
            except Exception as e:
                print(f"OCRTool: error on {img_path}: {e}", file=sys.stderr)

        output = "\n".join(unique_lines) if unique_lines else "No text detected."
        return _make_tool_json(self.TOOL_NAME, self.INPUTS, output)


# ---------------------------------------------------------------------------
# DescriptionTool
# ---------------------------------------------------------------------------

_SYNTH_DESC_LIMIT = 300
_SUBTITLE_SUMMARIZE_THRESHOLD = 5000


@dataclass
class _Frame:
    timestamp_ms: int
    jpeg_bytes: bytes
    subtitle: str


def _parse_srt(srt_path: str) -> list[tuple[int, int, str]]:
    """Return list of (start_ms, end_ms, text)."""
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

    # --- image path ---

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

    # --- video helpers (adapted from describe_video.py) ---

    def _extract_frames(self, video_path: str, subs: list, starts: list) -> list[_Frame]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if total_frames <= 0:
            raise RuntimeError(f"Cannot determine frame count for: {video_path}")

        # Evenly spaced frame indices across the full video
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

    # --- public run ---

    def run(self, data: DataManager) -> dict | None:
        try:
            if not data.isVideo:
                desc = self._describe_image_path(data.originalMedia)
                return _make_tool_json(self.TOOL_NAME, self.INPUTS, desc)

            # For video: look for a matching SRT next to the video file
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


# ---------------------------------------------------------------------------
# Stub detection tools
# ---------------------------------------------------------------------------
# These tools have no backing implementation yet.
# They return hasRun=0 and Output=None so downstream consumers can detect them.
# Replace the `run` method body when a real model or API is available.

class _StubTool(VisionTool):
    """Base for tools that are not yet implemented."""

    def run(self, data: DataManager) -> dict | None:
        return _make_tool_json(
            self.TOOL_NAME,
            self.INPUTS,
            output=None,
            explanation="Not implemented.",
            has_run=0,
        )


class AiDetectionTool(_StubTool):
    TOOL_NAME = "AI Detection"
    INPUTS = ["Image", "Video"]


class DeepFakeDetectionTool(_StubTool):
    TOOL_NAME = "Deepfake Detection"
    INPUTS = ["Video"]


class LipSyncDetectionTool(_StubTool):
    TOOL_NAME = "Lipsync Detection"
    INPUTS = ["Video"]


class WeatherDetectionTool(_StubTool):
    TOOL_NAME = "Weather Detection"
    INPUTS = ["Image", "Video", "Keyframes"]


class GeolocationTool(_StubTool):
    TOOL_NAME = "Geolocation"
    INPUTS = ["Image", "Keyframes"]


class FacialRecognitionTool(_StubTool):
    TOOL_NAME = "Facial Recognition"
    INPUTS = ["Image", "Video", "Keyframes"]


class NERTool(_StubTool):
    TOOL_NAME = "NER"
    INPUTS = ["Image", "Video", "Keyframes", "Transcript", "Description", "OCR", "Metadata"]