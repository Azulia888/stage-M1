#!/usr/bin/env python3
"""
Describe a video using a locally running vLLM server with Qwen2.5-VL-7B-Instruct.

Start the server first:
    vllm serve Qwen/Qwen2.5-VL-7B-Instruct

Then run:
    python describe_video.py <video_path>
"""

import argparse
import base64
import sys
from pathlib import Path

import cv2
import requests

# --- Config ---
VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_FRAMES = 16          # number of frames to sample from the video
FRAME_QUALITY = 85       # JPEG quality for encoded frames
PROMPT = "Describe this video in detail. What is happening, who or what is present, and what is the overall context or story?"


def sample_frames(video_path: str, n: int) -> list[str]:
    """Extract n evenly-spaced frames from the video and return as base64 JPEG strings."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError("Could not determine frame count.")

    indices = [int(i * (total - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
    frames_b64 = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        ret2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, FRAME_QUALITY])
        if ret2:
            frames_b64.append(base64.b64encode(buf.tobytes()).decode("utf-8"))

    cap.release()
    return frames_b64


def describe_video(video_path: str, url: str, max_frames: int, prompt: str) -> str:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"Sampling {max_frames} frames from: {path.name}")
    frames = sample_frames(str(path), max_frames)
    print(f"Sampled {len(frames)} frames. Sending to vLLM...")

    # Build content: interleave frames then the text prompt
    content = []
    for b64 in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1024,
        "temperature": 0.2,
    }

    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description="Describe a video using Qwen2.5-VL via vLLM.")
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("--url", default=VLLM_URL, help=f"vLLM endpoint (default: {VLLM_URL})")
    parser.add_argument("--frames", type=int, default=MAX_FRAMES, help=f"Frames to sample (default: {MAX_FRAMES})")
    parser.add_argument("--prompt", default=PROMPT, help="Prompt to send alongside the frames")
    args = parser.parse_args()

    description = describe_video(args.video, args.url, args.frames, args.prompt)
    print("\n--- Video Description ---")
    print(description)


if __name__ == "__main__":
    main()