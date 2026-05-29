"""
Keyframe extraction from a local video using OpenCV.

Strategies:
  - uniform   : evenly spaced frames across the video (default)
  - scene     : frames at scene changes (via frame diff threshold)
  - both      : scene detection, then uniform-sample within each scene

Usage:
    python extract_keyframes.py --video path/to/video.mp4 --count 10 --output ./keyframes
    python extract_keyframes.py --video path/to/video.mp4 --strategy scene --threshold 30
    python extract_keyframes.py --video path/to/video.mp4 --strategy both --count 5
"""

import argparse
import os
import cv2
import numpy as np


def open_video(path: str) -> cv2.VideoCapture:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open: {path}")
    return cap


def save_frame(frame: np.ndarray, output_dir: str, index: int) -> str:
    out_path = os.path.join(output_dir, f"frame_{index:05d}.jpg")
    cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return out_path


# --- Strategies ---

def extract_uniform(cap: cv2.VideoCapture, num_frames: int, output_dir: str) -> list[str]:
    """Pick N evenly spaced frames across the entire video."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError("Could not determine frame count. Try --strategy scene.")

    indices = np.linspace(0, total - 1, num=num_frames, dtype=int)
    saved = []
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            saved.append(save_frame(frame, output_dir, i))
    return saved


def extract_scene(cap: cv2.VideoCapture, threshold: float, output_dir: str) -> list[str]:
    """
    Save one frame per detected scene change.
    threshold: mean absolute diff between consecutive grayscale frames (0-255).
               Lower = more sensitive. Typical range: 20-40.
    """
    saved = []
    prev_gray = None
    frame_idx = 0
    save_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = np.mean(np.abs(gray.astype(float) - prev_gray.astype(float)))
            if diff >= threshold:
                saved.append(save_frame(frame, output_dir, save_idx))
                save_idx += 1

        prev_gray = gray
        frame_idx += 1

    # Always include the first frame
    if save_idx == 0 or (saved and "frame_00000" not in saved[0]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if ret:
            path = os.path.join(output_dir, "frame_00000.jpg")
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved.insert(0, path)

    return saved


def extract_both(cap: cv2.VideoCapture, num_frames: int, threshold: float, output_dir: str) -> list[str]:
    """
    Detect scene boundaries, then pick `num_frames` uniformly distributed
    across those boundaries for a good spread with temporal relevance.
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scene_frames = [0]  # always include frame 0
    prev_gray = None

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

    # Sample evenly from detected scene-change frame indices
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
            saved.append(save_frame(frame, output_dir, save_idx))

    return saved


# --- Main ---

def run(args: argparse.Namespace) -> None:
    os.makedirs(args.output, exist_ok=True)
    cap = open_video(args.video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps > 0 else 0
    print(f"Video : {args.video}")
    print(f"Frames: {total}  FPS: {fps:.2f}  Duration: {duration:.1f}s")
    print(f"Strategy: {args.strategy}")

    try:
        if args.strategy == "uniform":
            saved = extract_uniform(cap, args.count, args.output)
        elif args.strategy == "scene":
            saved = extract_scene(cap, args.threshold, args.output)
        elif args.strategy == "both":
            saved = extract_both(cap, args.count, args.threshold, args.output)
        else:
            raise ValueError(f"Unknown strategy: {args.strategy}")
    finally:
        cap.release()

    print(f"\nSaved {len(saved)} frame(s) to: {args.output}")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract keyframes from a video using OpenCV.")
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument("--output", default="./keyframes", help="Output directory (default: ./keyframes).")
    parser.add_argument("--strategy", choices=["uniform", "scene", "both"], default="uniform",
                        help="uniform: evenly spaced | scene: on scene changes | both: scene-aware uniform (default: uniform).")
    parser.add_argument("--count", type=int, default=10,
                        help="Number of frames to extract, used by uniform and both (default: 10).")
    parser.add_argument("--threshold", type=float, default=30.0,
                        help="Scene change sensitivity for scene/both strategies, 0-255 (default: 30).")
    args = parser.parse_args()
    run(args)