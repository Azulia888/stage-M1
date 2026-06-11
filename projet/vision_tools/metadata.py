"""
metadata.py — MetadataTool: extracts technical and file metadata via ffprobe / exiftool.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from data_manager import DataManager
from vision_tools.base import VisionTool, _make_tool_json


class MetadataTool(VisionTool):
    TOOL_NAME = "Metadata Gatherer"
    INPUTS = ["Image", "Video"]

    def run(self, data: DataManager) -> dict | None:
        path = Path(data.originalMedia)
        metadata_lines = []

        # 1. ffprobe (videos)
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

        # 2. exiftool (images + fallback for video)
        try:
            proc = subprocess.run(
                ["exiftool", "-json", str(path)],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode == 0:
                exif = json.loads(proc.stdout)
                if exif:
                    for key in (
                        "DateTimeOriginal", "CreateDate", "ModifyDate",
                        "GPSLatitude", "GPSLongitude", "Make", "Model",
                        "ImageWidth", "ImageHeight", "MIMEType",
                    ):
                        val = exif[0].get(key)
                        if val:
                            metadata_lines.append(f"{key}: {val}")
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
            pass

        # 3. File stats
        try:
            stat = path.stat()
            metadata_lines.append(f"File size: {stat.st_size} bytes")
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
            metadata_lines.append(f"Modified: {mtime}")
        except OSError:
            pass

        # 4. Sidecar metadata file
        if data.metadata_path:
            try:
                sidecar = Path(data.metadata_path).read_text(encoding="utf-8", errors="replace").strip()
                if sidecar:
                    metadata_lines.append(f"Sidecar: {sidecar}")
            except OSError:
                pass

        output = "\n".join(metadata_lines) if metadata_lines else "No metadata found."
        return _make_tool_json(self.TOOL_NAME, self.INPUTS, output)