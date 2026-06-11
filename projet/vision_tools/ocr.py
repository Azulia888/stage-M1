"""
ocr.py — OCRTool: text extraction from images/keyframes via Ollama vision model.
"""

from __future__ import annotations

import base64
import sys

from data_manager import DataManager
from vision_tools.base import (
    VisionTool, _make_tool_json, _ollama_post, _ollama_response,
    OLLAMA_HOST, OLLAMA_VISION_MODEL,
)


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
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
        return text

    def run(self, data: DataManager) -> dict | None:
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