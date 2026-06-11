"""
lipsync.py — LipSyncDetectionTool: audio-visual sync analysis via SyncNet.
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from data_manager import DataManager
from vision_tools.base import VisionTool, _make_tool_json


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYNCNET_DIR = os.environ.get("SYNCNET_DIR", str(Path(".") / "syncnet_python"))
SYNCNET_MODEL = os.environ.get(
    "SYNCNET_MODEL",
    str(Path(SYNCNET_DIR) / "data" / "syncnet_v2.model"),
)

_CONF_LOW_SIGNAL = 3.0
_OFFSET_MINOR = 2
_OFFSET_SUSPECT = 5

_RE_OFFSET = re.compile(r"AV offset:\s*([-\d]+)")
_RE_MIN_DIST = re.compile(r"Min dist:\s*([\d.]+)")
_RE_CONFIDENCE = re.compile(r"Confidence:\s*([\d.]+)")


# ---------------------------------------------------------------------------
# Tool
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
        self.syncnet_dir = str(Path(syncnet_dir).resolve())
        self.model_path = str(Path(model_path).resolve())
        self.timeout = timeout

    def _check_install(self) -> str | None:
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
        print(f"LipSyncDetectionTool [{label}]: {' '.join(args)}", file=sys.stderr)
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
        result: dict[str, float | None] = {
            "av_offset": None, "min_dist": None, "confidence": None,
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
        if av_offset is None or syncnet_conf is None:
            return (
                "inconclusive", 15,
                "SyncNet did not produce a parseable result. "
                "The video may lack audio, be too short, or have no detectable face.",
            )

        abs_offset = abs(int(av_offset))

        if syncnet_conf < _CONF_LOW_SIGNAL:
            return (
                "no_signal", 20,
                f"SyncNet confidence {syncnet_conf:.3f} is below the low-signal "
                f"threshold ({_CONF_LOW_SIGNAL}). No face track was long enough "
                "to produce a reliable reading. This does NOT rule out manipulation.",
            )

        if abs_offset == 0:
            return (
                "in_sync", 80,
                f"AV offset is 0 frames with SyncNet confidence {syncnet_conf:.3f}. "
                "Audio and lip movements are consistent.",
            )
        elif abs_offset <= _OFFSET_MINOR:
            return (
                "minor_drift", 60,
                f"AV offset is {int(av_offset)} frame(s). Minor drift can result "
                "from encoding or re-muxing and does not necessarily indicate "
                f"manipulation. SyncNet confidence: {syncnet_conf:.3f}.",
            )
        elif abs_offset <= _OFFSET_SUSPECT:
            return (
                "suspicious", 40,
                f"AV offset is {int(av_offset)} frame(s), which exceeds the "
                f"minor-drift threshold ({_OFFSET_MINOR}). This may indicate "
                "audio replacement or lip-sync manipulation. "
                f"SyncNet confidence: {syncnet_conf:.3f}. "
                "Manual review and corroboration are required.",
            )
        else:
            return (
                "suspicious", 25,
                f"AV offset is {int(av_offset)} frame(s), which is large enough "
                "to be clearly perceptible and strongly suggests audio-visual "
                "desynchronisation. This may indicate deliberate manipulation. "
                f"SyncNet confidence: {syncnet_conf:.3f}. "
                "Manual review and corroboration are strongly recommended.",
            )

    def run(self, data: DataManager) -> dict | None:
        if not data.isVideo:
            return None

        install_error = self._check_install()
        if install_error:
            print(f"LipSyncDetectionTool: {install_error}", file=sys.stderr)
            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS, None, explanation=install_error, has_run=0,
            )

        video_path = str(Path(data.originalMedia).resolve())
        reference = f"afc_{uuid.uuid4().hex[:8]}"

        with tempfile.TemporaryDirectory(prefix="syncnet_") as tmp_data_dir:
            try:
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
                        self.TOOL_NAME, self.INPUTS, None, explanation=msg, has_run=0,
                    )

                print(
                    f"LipSyncDetectionTool: {len(crop_files)} face track(s) found.",
                    file=sys.stderr,
                )

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

                combined_output = stdout + "\n" + stderr
                parsed = self._parse_syncnet_output(combined_output)
                av_offset = parsed["av_offset"]
                min_dist = parsed["min_dist"]
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
                    self.TOOL_NAME, self.INPUTS, None, explanation=msg, has_run=0,
                )
            except RuntimeError as e:
                print(f"LipSyncDetectionTool: {e}", file=sys.stderr)
                return _make_tool_json(
                    self.TOOL_NAME, self.INPUTS, None, explanation=str(e), has_run=0,
                )

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