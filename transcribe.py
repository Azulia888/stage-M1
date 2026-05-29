#!/usr/bin/env python3
"""
Transcribe a local video file using OpenAI Whisper.

Usage:
    python transcribe.py <video_file> [--model base] [--output transcript.txt] [--language en]

Models (speed vs accuracy tradeoff):
    tiny, base, small, medium, large
    Default: base (good balance for most use cases)

Install dependencies:
    pip install openai-whisper
    # ffmpeg must also be installed: https://ffmpeg.org/download.html
"""

import argparse
import sys
import time
from pathlib import Path


def transcribe(video_path: str, model_name: str, output_path: str | None, language: str | None):
    try:
        import whisper
    except ImportError:
        print("Error: openai-whisper is not installed.")
        print("Run: pip install openai-whisper")
        sys.exit(1)

    import torch

    video = Path(video_path)
    if not video.exists():
        print(f"Error: file not found: {video_path}")
        sys.exit(1)

    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)
    # Force fp32 — fp16 produces NaN logits on some CUDA GPUs
    model = model.to(torch.float32)

    print(f"Transcribing '{video.name}'...")
    start = time.time()

    options = {"fp16": False}
    if language:
        options["language"] = language

    result = model.transcribe(str(video), **options)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s — detected language: {result.get('language', 'unknown')}")

    transcript = result["text"].strip()

    # Determine output path
    if output_path:
        out = Path(output_path)
    else:
        out = video.with_suffix(".txt")

    out.write_text(transcript, encoding="utf-8")
    print(f"Transcript saved to: {out}")

    # Also print a preview
    preview = transcript[:500]
    print(f"\n--- Preview ---\n{preview}{'...' if len(transcript) > 500 else ''}")

    # Optionally save SRT subtitles
    srt_path = video.with_suffix(".srt")
    segments = result.get("segments", [])
    if segments:
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                start_ts = format_timestamp(seg["start"])
                end_ts = format_timestamp(seg["end"])
                f.write(f"{i}\n{start_ts} --> {end_ts}\n{seg['text'].strip()}\n\n")
        print(f"SRT subtitles saved to: {srt_path}")


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    parser = argparse.ArgumentParser(description="Transcribe a video file using Whisper.")
    parser.add_argument("video", help="Path to the video file (MP4, MKV, etc.)")
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model to use (default: base)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .txt file path (default: same name as video with .txt extension)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code to force (e.g. 'en', 'fr'). Auto-detected if omitted.",
    )
    args = parser.parse_args()
    transcribe(args.video, args.model, args.output, args.language)


if __name__ == "__main__":
    main()