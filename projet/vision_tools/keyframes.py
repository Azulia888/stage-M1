"""
keyframes.py — KeyFrameExtractionTool: adaptive keyframe extraction via OpenCV.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

from data_manager import DataManager
from vision_tools.base import VisionTool, _make_tool_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _duration_to_count(duration_s: float) -> int:
    if duration_s < 30:
        return 5
    elif duration_s < 90:
        return 10
    elif duration_s < 180:
        return 15
    elif duration_s < 300:
        return 20
    else:
        return 30


def _ms_to_filename_ts(ms: int) -> str:
    h = ms // 3_600_000
    ms -= h * 3_600_000
    m = ms // 60_000
    ms -= m * 60_000
    s = ms // 1_000
    ms -= s * 1_000
    return f"{h:02d}h{m:02d}m{s:02d}s{ms:03d}ms"


def _save_frame(frame: np.ndarray, output_dir: str, index: int, timestamp_ms: int) -> str:
    ts_str = _ms_to_filename_ts(timestamp_ms)
    path = os.path.join(output_dir, f"frame_{index:05d}_{ts_str}.jpg")
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


def _kf_uniform(cap: cv2.VideoCapture, num_frames: int, output_dir: str, fps: float) -> list[str]:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError("Could not determine frame count.")
    indices = np.linspace(0, total - 1, num=num_frames, dtype=int)
    saved = []
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            ts_ms = int(idx / fps * 1000) if fps > 0 else 0
            saved.append(_save_frame(frame, output_dir, i, ts_ms))
    return saved


def _kf_scene(cap: cv2.VideoCapture, threshold: float, output_dir: str, fps: float) -> list[str]:
    saved, prev_gray, save_idx, frame_idx = [], None, 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = np.mean(np.abs(gray.astype(float) - prev_gray.astype(float)))
            if diff >= threshold:
                ts_ms = int(frame_idx / fps * 1000) if fps > 0 else 0
                saved.append(_save_frame(frame, output_dir, save_idx, ts_ms))
                save_idx += 1
        prev_gray = gray
        frame_idx += 1
    if not saved:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if ret:
            saved.insert(0, _save_frame(frame, output_dir, 0, 0))
    return saved


def _kf_both(
    cap: cv2.VideoCapture, num_frames: int, threshold: float, output_dir: str, fps: float
) -> list[str]:
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
            ts_ms = int(frame_pos / fps * 1000) if fps > 0 else 0
            saved.append(_save_frame(frame, output_dir, save_idx, ts_ms))
    return saved


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class KeyFrameExtractionTool(VisionTool):
    TOOL_NAME = "Keyframes"
    INPUTS = ["Video"]

    def __init__(
        self,
        strategy: str = "both",
        count: int | None = None,
        threshold: float = 30.0,
        output_dir: str | None = None,
    ):
        self.strategy = strategy
        self.count = count
        self.threshold = threshold
        self.output_dir = output_dir

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

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_s = total_frames / fps if fps > 0 else 0.0
            count = self.count if self.count is not None else _duration_to_count(duration_s)

            if self.strategy == "uniform":
                saved = _kf_uniform(cap, count, out_dir, fps)
            elif self.strategy == "scene":
                saved = _kf_scene(cap, self.threshold, out_dir, fps)
            else:
                saved = _kf_both(cap, count, self.threshold, out_dir, fps)

            cap.release()
            print(
                f"KeyFrameExtractionTool: saved {len(saved)} frames to {out_dir} "
                f"(duration={duration_s:.1f}s, count={count})",
                file=sys.stderr,
            )
            return _make_tool_json(self.TOOL_NAME, self.INPUTS, saved)

        except Exception as e:
            print(f"KeyFrameExtractionTool error: {e}", file=sys.stderr)
            return _make_tool_json(self.TOOL_NAME, self.INPUTS, None, explanation=str(e), has_run=0)