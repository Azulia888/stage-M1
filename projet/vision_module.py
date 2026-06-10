import json
import re
import sys
from pathlib import Path

from vision_tool import (
    TranscriptTool, KeyFrameExtractionTool, MetadataTool, OCRTool, DescriptionTool,
    DeepFakeDetectionTool, LipSyncDetectionTool, AiDetectionTool, WeatherDetectionTool,
    GeolocationTool, FacialRecognitionTool, NERTool, MetadataAnalyzerTool
)
from data_manager import DataManager


class VisionModule:
    data: DataManager
    tools: list

    def __init__(self):
        self.data = None
        self.tools = [
            TranscriptTool(),
            KeyFrameExtractionTool(),
            MetadataTool(),
            MetadataAnalyzerTool(),
            OCRTool(),
            DescriptionTool(),
            DeepFakeDetectionTool(),
            LipSyncDetectionTool(),
            AiDetectionTool(),
            WeatherDetectionTool(),
            GeolocationTool(),
            FacialRecognitionTool(),
            NERTool(),
        ]

    def run(self, media_path: str, metadata_path: str, isVideo: bool):
        self.data = DataManager(media_path, metadata_path, isVideo)
        for tool in self.tools:
            tool.addData(self.data)

    def runURL(self, url: str, isVideo: bool, download_dir: str | None = None):
        """Download a URL with yt-dlp, then run the full tool pipeline on it.

        Parameters
        ----------
        url:
            Any URL supported by yt-dlp (YouTube, Twitter/X, Facebook, etc.).
        isVideo:
            Pass True for video content, False for images / audio-only.
        download_dir:
            Directory to download into. Defaults to ./vision_module_<video_title>,
            derived from the video's title after a metadata probe. Pass an
            explicit path to override.
        """
        try:
            import yt_dlp
        except ImportError:
            raise RuntimeError("yt-dlp is not installed. Run: pip install yt-dlp")

        # --- Step 1: probe metadata (no download) to get the title ---
        probe_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        print(f"VisionModule.runURL: probing '{url}' ...", file=sys.stderr)
        with yt_dlp.YoutubeDL(probe_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            info = json.loads(json.dumps(ydl.sanitize_info(info), default=str))

        # --- Step 2: resolve download directory from title ---
        if download_dir is None:
            title = info.get("title") or info.get("id") or "video"
            folder_name = "vision_module_" + _slugify(title)
            download_dir = str(Path(".") / folder_name)

        Path(download_dir).mkdir(parents=True, exist_ok=True)
        print(f"VisionModule.runURL: downloading into '{download_dir}' ...", file=sys.stderr)

        # --- Step 3: download ---
        out_template = str(Path(download_dir) / "%(id)s.%(ext)s")
        metadata_path = str(Path(download_dir) / "metadata.json")

        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": out_template,
            "writeinfojson": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitlesformat": "srt",
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            info = json.loads(json.dumps(ydl.sanitize_info(info), default=str))

        media_path = self._resolve_media_path(info, download_dir)
        print(f"VisionModule.runURL: downloaded to '{media_path}'", file=sys.stderr)

        _write_metadata_sidecar(info, metadata_path)

        self.run(media_path, metadata_path, isVideo)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_media_path(info: dict, download_dir: str) -> str:
        """Return the path to the downloaded media file."""
        requested = info.get("requested_downloads") or []
        if requested and requested[0].get("filepath"):
            path = requested[0]["filepath"]
            if Path(path).exists():
                return path

        video_id = info.get("id", "video")
        ext = info.get("ext", "mp4")
        path = str(Path(download_dir) / f"{video_id}.{ext}")
        if Path(path).exists():
            return path

        for p in sorted(Path(download_dir).iterdir()):
            if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".avi", ".mov",
                                     ".jpg", ".jpeg", ".png", ".webp"}:
                return str(p)

        raise FileNotFoundError(
            f"Could not locate downloaded media in '{download_dir}'. "
            f"yt-dlp info dict keys: {list(info.keys())}"
        )


def _slugify(text: str, max_len: int = 60) -> str:
    """Convert a video title into a safe directory name."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)   # drop punctuation
    text = re.sub(r"[\s_-]+", "_", text)   # spaces/dashes → underscore
    text = text.strip("_")
    return text[:max_len] or "video"


def _write_metadata_sidecar(info: dict, metadata_path: str) -> None:
    """Write a subset of yt-dlp's info dict as a human-readable JSON sidecar."""
    keys = (
        "id", "title", "description", "uploader", "uploader_id", "uploader_url",
        "channel", "channel_id", "channel_url",
        "upload_date", "timestamp", "release_date",
        "duration", "duration_string",
        "view_count", "like_count", "comment_count", "repost_count",
        "webpage_url", "original_url", "extractor", "extractor_key",
        "tags", "categories", "age_limit",
        "width", "height", "fps", "vcodec", "acodec",
        "filesize", "filesize_approx",
        "location", "coordinates",
    )
    subset = {k: info[k] for k in keys if k in info and info[k] is not None}
    Path(metadata_path).write_text(json.dumps(subset, indent=2, default=str), encoding="utf-8")

test = VisionModule()
test.runURL("https://www.youtube.com/watch?v=uqO5Qgi4AcQ", True)
print(test.data.description)
print(test.data.metadata)
print(test.data.transcript)
print(test.data.ocr)
print(test.data.keyframes)
print(test.data.toolResult)


