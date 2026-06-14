#!/usr/bin/env python3
"""
Describe a video using an ollama image-to-text model and an SRT subtitle file.

Usage:
    python describe_video.py <video_file> <srt_file> [options]

Options:
    --model MODEL         Ollama vision model name (default: llava)
    --synth-model MODEL   Ollama model for synthesis step; falls back to --model
    --fps FLOAT           Frames per second to sample (default: 0.5)
    --host URL            Ollama host (default: http://localhost:11434)
    --output FILE         Write description to file instead of stdout
    --no-synthesis        Skip synthesis; print per-frame descriptions only
"""

import argparse
import base64
import bisect
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2


# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------

@dataclass
class Subtitle:
    index: int
    start_ms: int
    end_ms: int
    text: str


def _ts_to_ms(ts: str) -> int:
    """Convert SRT timestamp (HH:MM:SS,mmm) to milliseconds.

    Handles sub-spec files that use fewer than 3 ms digits by padding right,
    so '5' -> 500, '05' -> 50, '005' -> 5.
    """
    ts = ts.strip().replace(",", ".")
    h, m, rest = ts.split(":")
    s, ms_raw = rest.split(".")
    ms = int(ms_raw.ljust(3, "0")[:3])   # pad/truncate to exactly 3 digits
    return int(h) * 3_600_000 + int(m) * 60_000 + int(s) * 1_000 + ms


def parse_srt(path: str) -> list[Subtitle]:
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\n{2,}", text.strip())
    subs = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue
        times = lines[1].split("-->")
        if len(times) != 2:
            continue
        try:
            start_ms = _ts_to_ms(times[0])
            end_ms = _ts_to_ms(times[1])
        except (ValueError, AttributeError):
            continue
        text_body = " ".join(l.strip() for l in lines[2:] if l.strip())
        text_body = re.sub(r"<[^>]+>", "", text_body).strip()
        subs.append(Subtitle(idx, start_ms, end_ms, text_body))
    return subs


def subtitle_at_ms(subs: list[Subtitle], starts: list[int], ms: int) -> str:
    """Return subtitle text active at `ms` using binary search, or empty string.

    `starts` must be a pre-built sorted list of sub.start_ms values
    (same length and order as `subs`).
    """
    i = bisect.bisect_right(starts, ms) - 1
    if i >= 0 and subs[i].start_ms <= ms <= subs[i].end_ms:
        return subs[i].text
    return ""


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    timestamp_ms: int
    jpeg_bytes: bytes
    subtitle: str


def get_duration_ms(cap: cv2.VideoCapture) -> int:
    """Seek to end to get true duration; works for VFR and CFR video."""
    # CAP_PROP_POS_AVI_RATIO = 1.0 seeks to the last frame
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
    duration_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    cap.set(cv2.CAP_PROP_POS_MSEC, 0)  # rewind
    return duration_ms


def extract_frames(video_path: str, subs: list[Subtitle], fps: float) -> list[Frame]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    duration_ms = get_duration_ms(cap)
    if duration_ms == 0:
        # Fallback for containers that don't support AVI_RATIO seeking
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_ms = int(total_frames / video_fps * 1000)

    starts = [s.start_ms for s in subs]
    interval_ms = int(1000 / fps)
    frames = []
    ts_ms = 0

    while ts_ms <= duration_ms:
        # Seek by time (ms) — more reliable than frame number across codecs
        cap.set(cv2.CAP_PROP_POS_MSEC, ts_ms)
        ret, bgr = cap.read()
        if not ret:
            ts_ms += interval_ms
            continue

        h, w = bgr.shape[:2]
        if w > 640:
            scale = 640 / w
            bgr = cv2.resize(bgr, (640, int(h * scale)))

        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
        sub_text = subtitle_at_ms(subs, starts, ts_ms)
        frames.append(Frame(ts_ms, buf.tobytes(), sub_text))
        ts_ms += interval_ms

    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Ollama calls
# ---------------------------------------------------------------------------

def _ms_to_hms(ms: int) -> str:
    s = ms // 1000
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _ollama_post(host: str, payload: dict, timeout: int = 120, think: bool = True) -> dict:
    """POST to ollama /api/generate and return parsed JSON. Raises on HTTP errors.

    When think=False, passes think:false to ollama so reasoning models skip the
    internal chain-of-thought and write directly to response. Faster and avoids
    the empty-response issue on models like qwen3.5.
    """
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


def _extract_response(result: dict, label: str) -> str:
    """Return the model's response text, falling back to 'thinking' if 'response' is empty.

    Some reasoning models (e.g. qwen3) put their entire output in 'thinking'
    and leave 'response' empty. When that happens we use 'thinking' instead
    and log a warning so the user knows.
    """
    text = result.get("response", "").strip()
    if text:
        return text
    thinking = result.get("thinking", "").strip()
    if thinking:
        print(
            f"  WARNING ({label}): 'response' is empty but 'thinking' has content. "
            f"Using 'thinking' field. Consider passing --no-think to disable reasoning mode.",
            file=sys.stderr,
        )
        return thinking
    print(f"  WARNING ({label}): model returned empty response and empty thinking. Raw reply: {result}", file=sys.stderr)
    return ""


def describe_frame(host: str, model: str, frame: Frame, timeout: int = 180, think: bool = True) -> str:
    b64 = base64.b64encode(frame.jpeg_bytes).decode()
    prompt = "Describe what is happening in this video frame concisely."
    if frame.subtitle:
        prompt += f' The subtitle at this moment reads: "{frame.subtitle}"'

    result = _ollama_post(host, {
        "model": model,
        "prompt": prompt,
        "images": [b64],
        "stream": False,
    }, timeout=timeout, think=think)
    return _extract_response(result, "frame")


# Characters per frame description kept in the synthesis prompt.
# Keeps total prompt size bounded regardless of how verbose the vision model was.
_SYNTH_DESC_LIMIT = 300

# If the full subtitle transcript exceeds this many characters, summarize it first.
_SUBTITLE_SUMMARIZE_THRESHOLD = 5000


def extract_subtitle_transcript(subs: list[Subtitle]) -> str:
    """Return all subtitle lines as a single transcript, deduplicated and in order."""
    seen: set[str] = set()
    lines = []
    for sub in subs:
        if sub.text and sub.text not in seen:
            seen.add(sub.text)
            lines.append(sub.text)
    return " ".join(lines)


def summarize_subtitles(host: str, model: str, transcript: str, timeout: int = 1000, think: bool = True) -> str:
    """Ask the model to condense the subtitle transcript into a shorter summary."""
    print("  Subtitle transcript too long; summarizing subtitles first ...", file=sys.stderr)
    prompt = (
        "The following is a transcript of subtitles from a video. "
        "Summarize the spoken content concisely in a few sentences, "
        "preserving the key information and topics covered.\n\n"
        f"{transcript}"
    )
    result = _ollama_post(host, {"model": model, "prompt": prompt, "stream": False}, timeout=timeout, think=think)
    return _extract_response(result, "subtitle summarization")


def synthesize_description(host: str, model: str, frame_descriptions: list[dict],
                           subs: list[Subtitle], timeout: int = 1000, think: bool = True) -> str:
    """Synthesize a narrative from per-frame descriptions and subtitle transcript.

    If the subtitle transcript is short enough it is included verbatim;
    otherwise it is summarized first via a separate model call.
    Each frame description is truncated to _SYNTH_DESC_LIMIT characters so the
    combined prompt stays within the model's context window.
    """
    lines = []
    for d in frame_descriptions:
        desc = d["description"]
        if len(desc) > _SYNTH_DESC_LIMIT:
            desc = desc[:_SYNTH_DESC_LIMIT].rsplit(" ", 1)[0] + "..."
        lines.append(f"[{d['timestamp']}] {desc}")
    combined = "\n".join(lines)

    subtitle_section = ""
    if subs:
        transcript = extract_subtitle_transcript(subs)
        if transcript:
            if len(transcript) > _SUBTITLE_SUMMARIZE_THRESHOLD:
                transcript = summarize_subtitles(host, model, transcript, timeout=timeout, think=think)
                label = "Subtitle summary"
            else:
                label = "Subtitle transcript"
            subtitle_section = f"\n\n{label}:\n{transcript}"

    prompt = (
        "You are given timestamped visual descriptions of frames from a video"
        + (", along with its subtitle content" if subtitle_section else "")
        + ". Write a coherent, fluent textual summary of the full video content. "
        "This resume needs to be able to stand on its own, without added frame description or subtitles. "
        "Be specific and informative.\n\n"
        f"Frame descriptions:\n{combined}"
        f"{subtitle_section}"
    )

    print(f"  Synthesis prompt length: {len(prompt)} characters.", file=sys.stderr)

    result = _ollama_post(host, {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }, timeout=timeout, think=think)
    return _extract_response(result, "synthesis")


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_per_frame(frame_descriptions: list[dict]) -> str:
    blocks = []
    for d in frame_descriptions:
        header = f"[{d['timestamp']}]"
        if d["subtitle"]:
            header += f" subtitle: {d['subtitle']}"
        blocks.append(f"{header}\n{d['description']}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("srt", help="Path to the SRT subtitle file")
    parser.add_argument("--model", default="qwen3.5:2b", help="Ollama vision model (default: qwen3.5:2b)")
    parser.add_argument("--synth-model", default=None,
                        help="Ollama model for synthesis step (default: same as --model)")
    parser.add_argument("--fps", type=float, default=0.5,
                        help="Frames per second to sample (default: 0.5 = one frame every 2 s)")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--output", help="Write final description to this file")
    parser.add_argument("--no-synthesis", action="store_true",
                        help="Skip synthesis; print per-frame descriptions only")
    parser.add_argument("--frame-timeout", type=int, default=180,
                        help="Seconds to wait per frame description call (default: 120)")
    parser.add_argument("--synth-timeout", type=int, default=1000,
                        help="Seconds to wait for synthesis call (default: 600)")
    parser.add_argument("--no-think", action="store_true",
                        help="Pass think:false to ollama (disables chain-of-thought on reasoning "
                             "models like qwen3; faster and avoids empty-response issue)")
    args = parser.parse_args()

    synth_model = args.synth_model or args.model
    think = not args.no_think

    print(f"Parsing subtitles from {args.srt} ...", file=sys.stderr)
    subs = parse_srt(args.srt)
    print(f"  {len(subs)} subtitle entries loaded.", file=sys.stderr)

    print(f"Extracting frames from {args.video} at {args.fps} fps ...", file=sys.stderr)
    frames = extract_frames(args.video, subs, args.fps)
    print(f"  {len(frames)} frames extracted.", file=sys.stderr)

    frame_descriptions = []
    for i, frame in enumerate(frames, 1):
        ts = _ms_to_hms(frame.timestamp_ms)
        print(f"  [{i}/{len(frames)}] Describing frame at {ts} ...", file=sys.stderr)
        try:
            desc = describe_frame(args.host, args.model, frame, timeout=args.frame_timeout, think=think)
        except RuntimeError as e:
            print(f"    WARNING: frame skipped ({e})", file=sys.stderr)
            desc = "[description unavailable]"
        frame_descriptions.append({"timestamp": ts, "description": desc, "subtitle": frame.subtitle})
        print(f"    -> {desc[:80]}{'...' if len(desc) > 80 else ''}", file=sys.stderr)

    if not frame_descriptions:
        print("ERROR: no frames were described. Aborting.", file=sys.stderr)
        sys.exit(1)

    if args.no_synthesis:
        result = format_per_frame(frame_descriptions)
    else:
        print(f"Synthesizing final description with model '{synth_model}' ...", file=sys.stderr)
        try:
            result = synthesize_description(args.host, synth_model, frame_descriptions,
                                            subs=subs, timeout=args.synth_timeout, think=think)
        except RuntimeError as e:
            print(f"WARNING: synthesis failed ({e}). Falling back to per-frame output.", file=sys.stderr)
            result = format_per_frame(frame_descriptions)

    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
        print(f"Description written to {args.output}", file=sys.stderr)
    elif result:
        print(result)
    else:
        print("ERROR: result is empty — nothing to output.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()