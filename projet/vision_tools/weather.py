"""
weather.py — WeatherDetectionTool: weather/lighting analysis via Ollama vision model.
"""

from __future__ import annotations

import base64
import json
import sys

from data_manager import DataManager
from vision_tools.base import (
    VisionTool, _make_tool_json, _ollama_post, _ollama_response,
    OLLAMA_HOST, OLLAMA_VISION_MODEL, OLLAMA_SYNTH_MODEL,
)


# ---------------------------------------------------------------------------
# Prompts
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


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

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
        print("WeatherTool: analysing frame...", file=sys.stderr)
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
        print("WeatherTool: synthesising...", file=sys.stderr)
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

    def run(self, data: DataManager) -> dict | None:
        sources: list[str] = data.keyframes if data.isVideo else [data.originalMedia]

        if not sources:
            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS, None,
                explanation="No images available for weather analysis.", has_run=0,
            )

        observations: list[dict] = []
        for img_path in sources:
            try:
                observations.append(self._analyse_frame(img_path))
            except Exception as e:
                print(f"WeatherDetectionTool: error on {img_path}: {e}", file=sys.stderr)

        if not observations:
            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS, None,
                explanation="All frame analyses failed.", has_run=0,
            )

        output = observations[0] if len(observations) == 1 else self._synthesise(observations)

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
                + ("Inconsistencies between frames detected — may indicate editing."
                   if has_inconsistency else "")
            ),
            confidence=confidence,
            confidence_explanation=confidence_explanation,
            corroborating_tools=["Geolocation", "Description", "Keyframes"],
        )