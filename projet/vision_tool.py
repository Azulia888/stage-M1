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
import glob
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional



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



"""
vision_tool_additions.py — Implementations for NERTool, WeatherDetectionTool,
GeolocationTool, and LipSyncDetectionTool.
 
Drop these classes into vision_tool.py, replacing the corresponding _StubTool
subclasses. All shared helpers (_ollama_post, _ollama_response, _make_tool_json,
OLLAMA_HOST, OLLAMA_VISION_MODEL, OLLAMA_SYNTH_MODEL) are assumed to be
already defined there.
 
NOTE: There is a pre-existing bug in DataManager.addToolResult — the branch
`toolJson["ToolName"] == "Metadata"` never fires because MetadataTool.TOOL_NAME
is "Metadata Gatherer". This file does not fix it, but callers that rely on
data.metadata should be aware.
"""
 
 
# ---------------------------------------------------------------------------
# NERTool
# ---------------------------------------------------------------------------
 
# Entity types the model is asked to extract.
_NER_ENTITY_TYPES = (
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "DATE",
    "EVENT",
    "WORK_OF_ART",
    "LAW",
    "PRODUCT",
    "LANGUAGE",
    "NATIONALITY",
)
 
_NER_PROMPT_TEMPLATE = """\
You are a named-entity recognition (NER) system for a fact-checking newsroom.
Extract all named entities from the text below and return ONLY a valid JSON object.
The JSON must have one key per entity type and an array of unique string values.
Use only these entity types: {types}.
If no entities of a type are found, omit that key.
Output ONLY the JSON object — no explanation, no markdown fences.
 
Text:
{text}"""
 
 
class NERTool(VisionTool):
    TOOL_NAME = "NER"
    INPUTS = ["Image", "Video", "Keyframes", "Transcript", "Description", "OCR", "Metadata"]
 
    def __init__(self, model: str = OLLAMA_SYNTH_MODEL, host: str = OLLAMA_HOST):
        self.model = model
        self.host = host
 
    def _collect_text(self, data) -> str:
        """Aggregate all available text fields into a single string."""
        parts: list[str] = []
        if data.transcript:
            parts.append(f"[Transcript]\n{data.transcript}")
        if data.description:
            parts.append(f"[Description]\n{data.description}")
        if data.ocr:
            parts.append(f"[OCR]\n{data.ocr}")
        if data.metadata:
            parts.append(f"[Metadata]\n{data.metadata}")
        # Also pull from the sidecar metadata file if it has a title/description
        # that hasn't been captured yet.
        if data.metadata_path:
            try:
                raw = Path(data.metadata_path).read_text(encoding="utf-8", errors="replace")
                sidecar = json.loads(raw)
                for key in ("title", "description", "uploader", "channel", "tags"):
                    val = sidecar.get(key)
                    if val and isinstance(val, (str, list)):
                        if isinstance(val, list):
                            val = ", ".join(str(v) for v in val)
                        parts.append(f"[Sidecar:{key}]\n{val}")
            except Exception:
                pass
        return "\n\n".join(parts)
 
    def run(self, data) -> dict | None:
        text = self._collect_text(data)
        if not text.strip():
            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS, None,
                explanation="No text available for NER.",
                has_run=0,
            )
 
        # Truncate to avoid hitting context limits on large transcripts.
        max_chars = 12_000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[...truncated]"
 
        prompt = _NER_PROMPT_TEMPLATE.format(
            types=", ".join(_NER_ENTITY_TYPES),
            text=text,
        )
 
        try:
            result = _ollama_post(self.host, {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }, timeout=120)
            raw = _ollama_response(result).strip()
 
            # Strip accidental markdown fences
            if raw.startswith("```"):
                lines = raw.split("\n")
                inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                raw = "\n".join(inner).strip()
 
            entities = json.loads(raw)
 
            # Normalise: ensure all values are lists of strings
            for k in list(entities.keys()):
                if not isinstance(entities[k], list):
                    entities[k] = [str(entities[k])]
                entities[k] = [str(v).strip() for v in entities[k] if str(v).strip()]
                if not entities[k]:
                    del entities[k]
 
            total = sum(len(v) for v in entities.values())
            explanation = (
                f"Extracted {total} named entities across "
                f"{len(entities)} type(s) from aggregated text fields."
            )
            confidence = 70
            confidence_explanation = (
                "NER accuracy depends on the quality of upstream text fields. "
                "Transcript and OCR errors propagate directly. "
                "LLM-based extraction can hallucinate entities not present in the text; "
                "all results should be verified against the source."
            )
 
            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS,
                output=entities,
                explanation=explanation,
                confidence=confidence,
                confidence_explanation=confidence_explanation,
                corroborating_tools=["Transcript", "Description", "OCR", "Metadata Gatherer"],
            )
 
        except json.JSONDecodeError as e:
            print(f"NERTool: model did not return valid JSON — {e}", file=sys.stderr)
            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS, None,
                explanation=f"Model returned non-JSON output: {e}",
                has_run=0,
            )
        except Exception as e:
            print(f"NERTool error: {e}", file=sys.stderr)
            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS, None,
                explanation=str(e),
                has_run=0,
            )
 
 
# ---------------------------------------------------------------------------
# WeatherDetectionTool
# ---------------------------------------------------------------------------
 
_WEATHER_PROMPT = """\
You are an expert analyst examining video or image frames for a fact-checking newsroom.
Analyse the visual evidence in this image and provide your best assessment of:
1. Weather conditions (e.g. sunny, overcast, rainy, snowy, foggy, night-time).
2. Approximate time of day (e.g. dawn, morning, midday, afternoon, dusk, night).
3. Season, if inferable (e.g. summer, winter — only if strong visual evidence exists).
4. Any notable atmospheric or lighting anomalies.
 
Return ONLY a valid JSON object with keys: "weather", "time_of_day", "season", "notes".
Use null for fields you cannot determine. No explanation outside the JSON, no markdown fences."""
 
_WEATHER_SYNTH_PROMPT = """\
You are summarising weather observations across multiple frames of the same video
for a fact-checking newsroom.
Below are per-frame JSON observations. Produce a single consolidated JSON summary
with keys: "weather", "time_of_day", "season", "notes", "consistency".
"consistency" should flag any contradictions between frames (e.g. lighting shifts
that suggest different times of day) that might indicate editing or context manipulation.
Output ONLY the JSON object, no markdown fences.
 
Frame observations:
{observations}"""
 
 
class WeatherDetectionTool(VisionTool):
    TOOL_NAME = "Weather Detection"
    INPUTS = ["Image", "Video", "Keyframes"]
 
    def __init__(
        self,
        model: str = OLLAMA_VISION_MODEL,
        synth_model: str = OLLAMA_SYNTH_MODEL,
        host: str = OLLAMA_HOST,
        frame_timeout: int = 120,
        synth_timeout: int = 180,
    ):
        self.model = model
        self.synth_model = synth_model
        self.host = host
        self.frame_timeout = frame_timeout
        self.synth_timeout = synth_timeout
 
    def _analyse_frame(self, image_path: str) -> dict:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        result = _ollama_post(self.host, {
            "model": self.model,
            "prompt": _WEATHER_PROMPT,
            "images": [b64],
            "stream": False,
        }, timeout=self.frame_timeout)
        raw = _ollama_response(result).strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}
 
    def _synthesise(self, observations: list[dict]) -> dict:
        obs_text = "\n".join(
            f"Frame {i + 1}: {json.dumps(o)}" for i, o in enumerate(observations)
        )
        result = _ollama_post(self.host, {
            "model": self.synth_model,
            "prompt": _WEATHER_SYNTH_PROMPT.format(observations=obs_text),
            "stream": False,
        }, timeout=self.synth_timeout)
        raw = _ollama_response(result).strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}
 
    def run(self, data) -> dict | None:
        sources: list[str] = []
        if data.isVideo:
            sources = data.keyframes or []
        else:
            sources = [data.originalMedia]
 
        if not sources:
            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS, None,
                explanation="No images available for weather analysis.",
                has_run=0,
            )
 
        observations: list[dict] = []
        for img_path in sources:
            try:
                obs = self._analyse_frame(img_path)
                observations.append(obs)
            except Exception as e:
                print(f"WeatherDetectionTool: error on {img_path}: {e}", file=sys.stderr)
 
        if not observations:
            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS, None,
                explanation="All frame analyses failed.",
                has_run=0,
            )
 
        if len(observations) == 1:
            output = observations[0]
        else:
            output = self._synthesise(observations)
 
        # Flag consistency issues for fact-checking relevance
        consistency_note = output.get("consistency", "")
        has_inconsistency = bool(
            consistency_note and consistency_note.lower() not in ("none", "consistent", "")
        )
        confidence = 55 if has_inconsistency else 70
        confidence_explanation = (
            "Weather and lighting analysis from a vision model is indicative, not authoritative. "
            "Confidence is reduced when frame-to-frame inconsistencies are detected, "
            "as these may indicate edited or spliced footage."
            + (f" Inconsistency noted: {consistency_note}" if has_inconsistency else "")
        )
 
        return _make_tool_json(
            self.TOOL_NAME, self.INPUTS,
            output=output,
            explanation=(
                f"Weather conditions analysed across {len(observations)} frame(s). "
                + ("Inconsistencies between frames detected — may indicate editing." if has_inconsistency else "")
            ),
            confidence=confidence,
            confidence_explanation=confidence_explanation,
            corroborating_tools=["Geolocation", "Description", "Keyframes"],
        )
 
 
# ---------------------------------------------------------------------------
# GeolocationTool
# ---------------------------------------------------------------------------
 
_GEO_PROMPT = """\
You are an expert visual geolocation analyst working for a fact-checking newsroom.
Examine this image carefully for any location cues:
- Text on signs, shop fronts, street signs, licence plates, banners.
- Recognisable landmarks, monuments, architecture, urban layout.
- Vegetation, terrain, and landscape features.
- Flags, uniforms, or cultural markers.
 
Return ONLY a valid JSON object with these keys:
  "region": broadest confident region (continent, sub-region, or null),
  "country": ISO country name or null,
  "city_or_area": city or district if identifiable, otherwise null,
  "landmark": specific named landmark if identified, otherwise null,
  "visual_cues": array of strings describing the evidence used,
  "caveat": brief note on confidence limitations.
 
Do not guess beyond what the visual evidence supports.
Output ONLY the JSON object, no markdown fences."""
 
_GEO_SYNTH_PROMPT = """\
You are consolidating geolocation observations from multiple frames of the same video
for a fact-checking newsroom.
Below are per-frame JSON observations. Produce a single consolidated JSON summary
using the same keys: "region", "country", "city_or_area", "landmark",
"visual_cues", "caveat", "consistency".
"consistency" should flag any contradictions that suggest different filming locations
within the same purported video.
Output ONLY the JSON object, no markdown fences.
 
Frame observations:
{observations}"""
 
_GEO_ETHICAL_CAVEAT = (
    "IMPORTANT: Geolocation output is indicative only and must not be used to identify "
    "individuals or their whereabouts. Results are based solely on visual scene analysis "
    "and carry significant uncertainty. Do not publish geolocation findings without "
    "independent corroboration from primary sources."
)
 
 
class GeolocationTool(VisionTool):
    TOOL_NAME = "Geolocation"
    INPUTS = ["Image", "Keyframes"]
 
    def __init__(
        self,
        model: str = OLLAMA_VISION_MODEL,
        synth_model: str = OLLAMA_SYNTH_MODEL,
        host: str = OLLAMA_HOST,
        frame_timeout: int = 120,
        synth_timeout: int = 180,
    ):
        self.model = model
        self.synth_model = synth_model
        self.host = host
        self.frame_timeout = frame_timeout
        self.synth_timeout = synth_timeout
 
    def _analyse_frame(self, image_path: str) -> dict:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        result = _ollama_post(self.host, {
            "model": self.model,
            "prompt": _GEO_PROMPT,
            "images": [b64],
            "stream": False,
        }, timeout=self.frame_timeout)
        raw = _ollama_response(result).strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}
 
    def _synthesise(self, observations: list[dict]) -> dict:
        obs_text = "\n".join(
            f"Frame {i + 1}: {json.dumps(o)}" for i, o in enumerate(observations)
        )
        result = _ollama_post(self.host, {
            "model": self.synth_model,
            "prompt": _GEO_SYNTH_PROMPT.format(observations=obs_text),
            "stream": False,
        }, timeout=self.synth_timeout)
        raw = _ollama_response(result).strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}
 
    def run(self, data) -> dict | None:
        sources: list[str] = []
        if data.isVideo:
            sources = data.keyframes or []
        else:
            sources = [data.originalMedia]
 
        if not sources:
            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS, None,
                explanation="No images available for geolocation.",
                has_run=0,
            )
 
        observations: list[dict] = []
        for img_path in sources:
            try:
                obs = self._analyse_frame(img_path)
                observations.append(obs)
            except Exception as e:
                print(f"GeolocationTool: error on {img_path}: {e}", file=sys.stderr)
 
        if not observations:
            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS, None,
                explanation="All frame analyses failed.",
                has_run=0,
            )
 
        output = observations[0] if len(observations) == 1 else self._synthesise(observations)
 
        # Assess how specific the result is to calibrate confidence
        country = output.get("country")
        city = output.get("city_or_area")
        landmark = output.get("landmark")
        cues = output.get("visual_cues") or []
 
        if landmark:
            confidence = 65
        elif city:
            confidence = 50
        elif country:
            confidence = 40
        else:
            confidence = 20
 
        confidence_explanation = (
            "Visual geolocation via a general-purpose vision model is inherently uncertain. "
            f"Confidence is based on specificity of identified cues ({len(cues)} cue(s) found). "
            "Results must be corroborated via reverse image search or primary source verification "
            "before any editorial use. " + _GEO_ETHICAL_CAVEAT
        )
 
        consistency_note = output.get("consistency", "")
        has_inconsistency = bool(
            consistency_note and consistency_note.lower() not in ("none", "consistent", "")
        )
        if has_inconsistency:
            confidence = max(confidence - 15, 10)
 
        return _make_tool_json(
            self.TOOL_NAME, self.INPUTS,
            output=output,
            explanation=(
                f"Location cues analysed across {len(observations)} frame(s). "
                + ("Inconsistent locations across frames detected." if has_inconsistency else "")
                + " " + _GEO_ETHICAL_CAVEAT
            ),
            confidence=confidence,
            confidence_explanation=confidence_explanation,
            corroborating_tools=["Weather Detection", "Description", "Keyframes", "Metadata Gatherer"],
        )


SYNCNET_DIR = os.environ.get("SYNCNET_DIR", str(Path(".") / "syncnet_python"))
SYNCNET_MODEL = os.environ.get(
    "SYNCNET_MODEL",
    str(Path(SYNCNET_DIR) / "data" / "syncnet_v2.model"),
)
 
# Thresholds
_CONF_LOW_SIGNAL = 3.0   # syncnet confidence below this = unreliable / no face
_OFFSET_MINOR    = 2     # |offset| <= this → minor drift
_OFFSET_SUSPECT  = 5     # |offset| <= this → suspicious; above → strong desync
 
# Regex patterns for parsing SyncNet log lines (emitted via Python logging)
_RE_OFFSET     = re.compile(r"AV offset:\s*([-\d]+)")
_RE_MIN_DIST   = re.compile(r"Min dist:\s*([\d.]+)")
_RE_CONFIDENCE = re.compile(r"Confidence:\s*([\d.]+)")
 
 
# ---------------------------------------------------------------------------
# LipSyncDetectionTool
# ---------------------------------------------------------------------------
 
class LipSyncDetectionTool(VisionTool):
    TOOL_NAME = "Lipsync Detection"
    INPUTS = ["Video"]
 
    def __init__(
        self,
        syncnet_dir: str = SYNCNET_DIR,
        model_path: str = SYNCNET_MODEL,
        timeout: int = 600,
    ):
        """
        Parameters
        ----------
        syncnet_dir:
            Path to the cloned syncnet_python repository.
            run_pipeline.py and run_syncnet.py must exist there.
        model_path:
            Path to the syncnet_v2.model weights file.
        timeout:
            Per-subprocess timeout in seconds. Face detection on long videos
            can be slow; 600 s is conservative for a CPU-only machine.
        """
        self.syncnet_dir = str(Path(syncnet_dir).resolve())
        self.model_path = str(Path(model_path).resolve())
        self.timeout = timeout
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
 
    def _check_install(self) -> str | None:
        """Return an error string if SyncNet is not usable, else None."""
        if not Path(self.syncnet_dir).is_dir():
            return (
                f"SYNCNET_DIR '{self.syncnet_dir}' not found. "
                "Clone https://github.com/joonson/syncnet_python and set "
                "the SYNCNET_DIR environment variable."
            )
        for script in ("run_pipeline.py", "run_syncnet.py"):
            if not (Path(self.syncnet_dir) / script).exists():
                return f"'{script}' not found in '{self.syncnet_dir}'."
        if not Path(self.model_path).exists():
            return (
                f"SyncNet model not found at '{self.model_path}'. "
                "Run `sh download_model.sh` inside the syncnet_python directory."
            )
        return None
 
    def _run_subprocess(self, args: list[str], label: str) -> tuple[str, str]:
        """Run a subprocess, return (stdout, stderr). Raises on non-zero exit."""
        print(
            f"LipSyncDetectionTool [{label}]: {' '.join(args)}",
            file=sys.stderr,
        )
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            cwd=self.syncnet_dir,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"{label} exited {proc.returncode}.\n"
                f"STDOUT: {proc.stdout[-2000:]}\n"
                f"STDERR: {proc.stderr[-2000:]}"
            )
        return proc.stdout, proc.stderr
 
    @staticmethod
    def _parse_syncnet_output(text: str) -> dict[str, float | None]:
        """
        Parse AV offset, min dist, and confidence from SyncNet log output.
        SyncNet writes these via Python's logging module; they appear in
        stderr when the default basicConfig handler is active.
        Returns a dict with keys 'av_offset', 'min_dist', 'confidence',
        each float or None if not found.
        """
        result: dict[str, float | None] = {
            "av_offset": None,
            "min_dist": None,
            "confidence": None,
        }
        for line in text.splitlines():
            m = _RE_OFFSET.search(line)
            if m:
                result["av_offset"] = float(m.group(1))
            m = _RE_MIN_DIST.search(line)
            if m:
                result["min_dist"] = float(m.group(1))
            m = _RE_CONFIDENCE.search(line)
            if m:
                result["confidence"] = float(m.group(1))
        return result
 
    @staticmethod
    def _assess(
        av_offset: float | None,
        syncnet_conf: float | None,
    ) -> tuple[str, int, str]:
        """
        Return (assessment, tool_confidence_0_100, confidence_explanation).
 
        assessment one of: "in_sync", "minor_drift", "suspicious",
                           "no_signal", "inconclusive"
        """
        if av_offset is None or syncnet_conf is None:
            return (
                "inconclusive",
                15,
                "SyncNet did not produce a parseable result. "
                "The video may lack audio, be too short, or have no detectable face.",
            )
 
        abs_offset = abs(int(av_offset))
 
        if syncnet_conf < _CONF_LOW_SIGNAL:
            return (
                "no_signal",
                20,
                f"SyncNet confidence {syncnet_conf:.3f} is below the low-signal "
                f"threshold ({_CONF_LOW_SIGNAL}). No face track was long enough "
                "to produce a reliable reading. This does NOT rule out manipulation.",
            )
 
        if abs_offset == 0:
            assessment = "in_sync"
            tool_conf = 80
            explanation = (
                f"AV offset is 0 frames with SyncNet confidence {syncnet_conf:.3f}. "
                "Audio and lip movements are consistent."
            )
        elif abs_offset <= _OFFSET_MINOR:
            assessment = "minor_drift"
            tool_conf = 60
            explanation = (
                f"AV offset is {int(av_offset)} frame(s). Minor drift can result "
                "from encoding or re-muxing and does not necessarily indicate "
                f"manipulation. SyncNet confidence: {syncnet_conf:.3f}."
            )
        elif abs_offset <= _OFFSET_SUSPECT:
            assessment = "suspicious"
            tool_conf = 40
            explanation = (
                f"AV offset is {int(av_offset)} frame(s), which exceeds the "
                f"minor-drift threshold ({_OFFSET_MINOR}). This may indicate "
                "audio replacement or lip-sync manipulation. "
                f"SyncNet confidence: {syncnet_conf:.3f}. "
                "Manual review and corroboration are required."
            )
        else:
            assessment = "suspicious"
            tool_conf = 25
            explanation = (
                f"AV offset is {int(av_offset)} frame(s), which is large enough "
                "to be clearly perceptible and strongly suggests audio-visual "
                "desynchronisation. This may indicate deliberate manipulation. "
                f"SyncNet confidence: {syncnet_conf:.3f}. "
                "Manual review and corroboration are strongly recommended."
            )
 
        return assessment, tool_conf, explanation
 
    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
 
    def run(self, data) -> dict | None:
        if not data.isVideo:
            return None
 
        # --- Pre-flight check ---
        install_error = self._check_install()
        if install_error:
            print(f"LipSyncDetectionTool: {install_error}", file=sys.stderr)
            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS, None,
                explanation=install_error,
                has_run=0,
            )
 
        video_path = str(Path(data.originalMedia).resolve())
        reference = f"afc_{uuid.uuid4().hex[:8]}"
 
        with tempfile.TemporaryDirectory(prefix="syncnet_") as tmp_data_dir:
            try:
                # --- Stage 1: face detection, tracking, crop ---
                print(
                    f"LipSyncDetectionTool: running pipeline on '{video_path}' ...",
                    file=sys.stderr,
                )
                self._run_subprocess(
                    [
                        "python", "run_pipeline.py",
                        "--videofile", video_path,
                        "--reference", reference,
                        "--data_dir", tmp_data_dir,
                    ],
                    label="run_pipeline",
                )
 
                # --- Check whether any face tracks were produced ---
                crop_dir = Path(tmp_data_dir) / "pycrop" / reference
                crop_files = sorted(glob.glob(str(crop_dir / "*.avi")))
                if not crop_files:
                    msg = (
                        "SyncNet face-tracking pipeline produced no face tracks. "
                        "The video may contain no visible faces, or all detected "
                        "faces were shorter than the minimum track length (100 frames). "
                        "Lip-sync analysis requires a sustained close-up of a speaking face."
                    )
                    print(f"LipSyncDetectionTool: {msg}", file=sys.stderr)
                    return _make_tool_json(
                        self.TOOL_NAME, self.INPUTS, None,
                        explanation=msg,
                        has_run=0,
                    )
 
                print(
                    f"LipSyncDetectionTool: {len(crop_files)} face track(s) found.",
                    file=sys.stderr,
                )
 
                # --- Stage 2: sync evaluation ---
                # run_syncnet.py evaluates each crop in pycrop/{reference}/*.avi
                # internally; we only need to point it at the original video and
                # data_dir so it can resolve the paths the same way run_pipeline did.
                stdout, stderr = self._run_subprocess(
                    [
                        "python", "run_syncnet.py",
                        "--videofile", video_path,
                        "--reference", reference,
                        "--data_dir", tmp_data_dir,
                        "--initial_model", self.model_path,
                    ],
                    label="run_syncnet",
                )
 
                # SyncNet logs to stderr via Python logging
                combined_output = stdout + "\n" + stderr
                parsed = self._parse_syncnet_output(combined_output)
 
                av_offset   = parsed["av_offset"]
                min_dist    = parsed["min_dist"]
                syncnet_conf = parsed["confidence"]
 
                print(
                    f"LipSyncDetectionTool: offset={av_offset}, "
                    f"min_dist={min_dist}, syncnet_conf={syncnet_conf}",
                    file=sys.stderr,
                )
 
            except subprocess.TimeoutExpired as e:
                msg = f"SyncNet subprocess timed out after {self.timeout}s: {e}"
                print(f"LipSyncDetectionTool: {msg}", file=sys.stderr)
                return _make_tool_json(
                    self.TOOL_NAME, self.INPUTS, None,
                    explanation=msg,
                    has_run=0,
                )
            except RuntimeError as e:
                print(f"LipSyncDetectionTool: {e}", file=sys.stderr)
                return _make_tool_json(
                    self.TOOL_NAME, self.INPUTS, None,
                    explanation=str(e),
                    has_run=0,
                )
 
        # --- Interpret results ---
        assessment, tool_conf, conf_explanation = self._assess(av_offset, syncnet_conf)
 
        output = {
            "assessment": assessment,
            "av_offset_frames": int(av_offset) if av_offset is not None else None,
            "syncnet_confidence": round(syncnet_conf, 3) if syncnet_conf is not None else None,
            "min_dist": round(min_dist, 3) if min_dist is not None else None,
            "face_tracks_analysed": len(crop_files),
        }
 
        return _make_tool_json(
            self.TOOL_NAME, self.INPUTS,
            output=output,
            explanation=(
                f"SyncNet lip-sync analysis on {len(crop_files)} face track(s). "
                f"Assessment: {assessment}. "
                f"AV offset: {av_offset} frame(s). "
                f"SyncNet confidence: {syncnet_conf}."
            ),
            confidence=tool_conf,
            confidence_explanation=conf_explanation,
            corroborating_tools=["Transcript", "Description", "Deepfake Detection"],
        )


# Approximate characters-per-token ratio used for budget estimation.
_CHARS_PER_TOKEN: int = 4
 
# Hard token ceiling per call (matches the assumption in the docstring).
_MAX_TOKENS: int = 4096
 
# Characters reserved for the prompt template boilerplate per call.
_PROMPT_OVERHEAD_CHARS: int = 512 * _CHARS_PER_TOKEN   # 2048
 
# Remaining characters available for raw metadata content per call.
_CHUNK_MAX_CHARS: int = (_MAX_TOKENS * _CHARS_PER_TOKEN) - _PROMPT_OVERHEAD_CHARS
# = 16 384 - 2 048 = 14 336
 
 
# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
 
_SINGLE_PROMPT = """\
You are a media forensics analyst working for a fact-checking newsroom.
Below is the raw technical and contextual metadata extracted from a media file.
Produce a concise, structured plain-text summary of the most relevant facts.
Focus on: provenance (origin, uploader, platform), timestamps and dates, \
technical properties (codec, resolution, duration), and any fields that could \
help assess authenticity or context.
Flag any anomalies (e.g. creation date after upload date, mismatched codecs, \
missing expected fields).
Output only the summary — no preamble, no markdown.
 
RAW METADATA:
{metadata}"""
 
_CHUNK_PROMPT = """\
You are a media forensics analyst working for a fact-checking newsroom.
Below is a PARTIAL excerpt of raw metadata extracted from a media file \
(chunk {chunk_index} of {total_chunks}).
Summarise the key facts present in this excerpt only.
Be concise. Do not invent information not present in the excerpt.
Output only the summary — no preamble, no markdown.
 
METADATA EXCERPT:
{metadata}"""
 
_SYNTHESIS_PROMPT = """\
You are a media forensics analyst working for a fact-checking newsroom.
Below are partial summaries produced from sequential excerpts of the raw \
metadata of a single media file.
Merge them into one coherent, non-redundant summary.
Focus on: provenance, timestamps, technical properties, and authenticity signals.
Flag any anomalies or contradictions across the partial summaries.
Output only the final summary — no preamble, no markdown.
 
PARTIAL SUMMARIES:
{summaries}"""
 
 
# ---------------------------------------------------------------------------
# MetadataAnalyzerTool
# ---------------------------------------------------------------------------
 
class MetadataAnalyzerTool(VisionTool):
    TOOL_NAME = "Metadata Analyzer"
    INPUTS = ["Metadata"]
 
    def __init__(
        self,
        model: str = OLLAMA_SYNTH_MODEL,
        host: str = OLLAMA_HOST,
        timeout: int = 120,
    ):
        self.model = model
        self.host = host
        self.timeout = timeout
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
 
    @staticmethod
    def _split_into_chunks(text: str, max_chars: int) -> list[str]:
        """
        Split text into chunks of at most max_chars characters, breaking
        only on line boundaries to preserve metadata field integrity.
        """
        lines = text.splitlines(keepends=True)
        chunks: list[str] = []
        current: list[str] = []
        current_len: int = 0
 
        for line in lines:
            # If a single line exceeds the limit, hard-cut it.
            if len(line) > max_chars:
                if current:
                    chunks.append("".join(current))
                    current, current_len = [], 0
                # Break the oversized line into max_chars slices.
                for i in range(0, len(line), max_chars):
                    chunks.append(line[i : i + max_chars])
                continue
 
            if current_len + len(line) > max_chars:
                chunks.append("".join(current))
                current, current_len = [], 0
 
            current.append(line)
            current_len += len(line)
 
        if current:
            chunks.append("".join(current))
 
        return chunks
 
    def _call_ollama(self, prompt: str) -> str:
        """Send a prompt to Ollama and return the response text."""
        result = _ollama_post(
            self.host,
            {"model": self.model, "prompt": prompt, "stream": False},
            timeout=self.timeout,
        )
        return _ollama_response(result).strip()
 
    def _summarise_chunk(self, chunk: str, index: int, total: int) -> str:
        prompt = _CHUNK_PROMPT.format(
            chunk_index=index,
            total_chunks=total,
            metadata=chunk,
        )
        return self._call_ollama(prompt)
 
    def _synthesise(self, partial_summaries: list[str]) -> str:
        joined = "\n\n---\n\n".join(
            f"[Chunk {i + 1}]\n{s}" for i, s in enumerate(partial_summaries)
        )
        prompt = _SYNTHESIS_PROMPT.format(summaries=joined)
        return self._call_ollama(prompt)
 
    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
 
    def run(self, data) -> Optional[dict]:
        raw = data.raw_metadata
        if not raw or not raw.strip():
            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS, None,
                explanation=(
                    "No raw metadata available. "
                    "MetadataTool must run before MetadataAnalyzerTool."
                ),
                has_run=0,
            )
 
        try:
            if len(raw) <= _CHUNK_MAX_CHARS:
                # Single-call path
                print(
                    "MetadataAnalyzerTool: single-call path "
                    f"({len(raw)} chars).",
                    file=sys.stderr,
                )
                prompt = _SINGLE_PROMPT.format(metadata=raw)
                summary = self._call_ollama(prompt)
                partial_summaries = None
                n_chunks = 1
 
            else:
                # Multi-chunk path
                chunks = self._split_into_chunks(raw, _CHUNK_MAX_CHARS)
                n_chunks = len(chunks)
                print(
                    f"MetadataAnalyzerTool: multi-chunk path — "
                    f"{len(raw)} chars split into {n_chunks} chunk(s).",
                    file=sys.stderr,
                )
                partial_summaries = []
                for i, chunk in enumerate(chunks, start=1):
                    print(
                        f"  MetadataAnalyzerTool: summarising chunk "
                        f"{i}/{n_chunks} ({len(chunk)} chars) ...",
                        file=sys.stderr,
                    )
                    partial_summaries.append(
                        self._summarise_chunk(chunk, i, n_chunks)
                    )
 
                print(
                    "MetadataAnalyzerTool: synthesising partial summaries ...",
                    file=sys.stderr,
                )
                summary = self._synthesise(partial_summaries)
 
            output = {
                "summary": summary,
                "chunks_used": n_chunks,
                # Partial summaries are included for transparency / audit trail.
                # None when the single-call path was used.
                "partial_summaries": partial_summaries,
            }
 
            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS,
                output=output,
                explanation=(
                    f"Metadata synthesised from {n_chunks} chunk(s). "
                    + (
                        "Single-call path (metadata fits within context window)."
                        if n_chunks == 1
                        else f"Multi-chunk path: {n_chunks} partial summaries merged."
                    )
                ),
                confidence=75,
                confidence_explanation=(
                    "LLM-generated summary of structured metadata. "
                    "Factual accuracy is high for fields present verbatim in the raw data. "
                    "The model may misinterpret ambiguous field names or codec strings. "
                    "Always cross-check against data.raw_metadata for precision."
                ),
                corroborating_tools=["Metadata Gatherer", "Description", "NER"],
            )
 
        except Exception as e:
            print(f"MetadataAnalyzerTool error: {e}", file=sys.stderr)
            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS, None,
                explanation=str(e),
                has_run=0,
            )
 
 
# ---------------------------------------------------------------------------
# Patched DataManager
# Replace the contents of data_manager.py with this class.
#
# Changes vs. original:
#   - Added raw_metadata field (set by MetadataTool / "Metadata Gatherer").
#   - Fixed addToolResult branch: "Metadata Gatherer" → raw_metadata
#     (was broken: checked for "Metadata" which never matched).
#   - Added branch: "Metadata Analyzer" → metadata (the synthesised summary).
# ---------------------------------------------------------------------------
 
class DataManager:
    originalMedia: str
    keyframes: list[str]
    description: str
    metadata: str          # Synthesised summary from MetadataAnalyzerTool
    raw_metadata: str      # Raw text from MetadataTool
    metadata_path: str
    ocr: str
    transcript: str
    isVideo: bool
    toolResult: dict
 
    def __init__(self, originalMedia: str, metadata_path: str, isVideo: bool):
        self.originalMedia = originalMedia
        self.metadata_path = metadata_path
        self.isVideo = isVideo
        self.keyframes = []
        self.description = None
        self.metadata = None
        self.raw_metadata = None
        self.ocr = None
        self.transcript = None
        self.toolResult = {}
 
    def addToolResult(self, toolJson: dict):
        if "ToolName" not in toolJson:
            return
 
        self.toolResult[toolJson["ToolName"]] = toolJson
 
        name = toolJson["ToolName"]
 
        if name == "Description":
            self.description = toolJson["Output"]
        elif name == "Metadata Gatherer":
            self.raw_metadata = toolJson["Output"]   # raw text; was broken before
        elif name == "Metadata Analyzer":
            # Output is a dict; expose only the final summary string on data.metadata
            output = toolJson["Output"]
            if isinstance(output, dict):
                self.metadata = output.get("summary")
            else:
                self.metadata = output
        elif name == "OCR":
            self.ocr = toolJson["Output"]
        elif name == "Keyframes":
            self.keyframes = toolJson["Output"]
        elif name == "Transcript":
            self.transcript = toolJson["Output"]
 




class AiDetectionTool(_StubTool):
    TOOL_NAME = "AI Detection"
    INPUTS = ["Image", "Video"]


class DeepFakeDetectionTool(_StubTool):
    TOOL_NAME = "Deepfake Detection"
    INPUTS = ["Video"]


class FacialRecognitionTool(_StubTool):
    TOOL_NAME = "Facial Recognition"
    INPUTS = ["Image", "Video", "Keyframes"]

