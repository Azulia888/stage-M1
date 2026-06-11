"""
geolocation.py — GeolocationTool: visual geolocation analysis via Ollama vision model.
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

_GEO_PROMPT = """\
You are an expert visual geolocation analyst working for a fact-checking newsroom.
Examine this image carefully for any location cues:
- Text on signs, shop fronts, street signs, licence plates, banners.
- Recognisable landmarks, monuments, architecture, urban layout.
- Vegetation, terrain, and landscape features.
- Flags, uniforms, or cultural markers.

Return ONLY a valid JSON object with these keys:
  "region": broadest confident region (continent, sub-region, or null),
  "country": ISO country name or null,
  "city_or_area": city or district if identifiable, otherwise null,
  "landmark": specific named landmark if identified, otherwise null,
  "visual_cues": array of strings describing the evidence used,
  "caveat": brief note on confidence limitations.

Do not guess beyond what the visual evidence supports.
Output ONLY the JSON object, no markdown fences."""

_GEO_SYNTH_PROMPT = """\
You are consolidating geolocation observations from multiple frames of the same video
for a fact-checking newsroom.
Below are per-frame JSON observations. Produce a single consolidated JSON summary
using the same keys: "region", "country", "city_or_area", "landmark",
"visual_cues", "caveat", "consistency".
"consistency" should flag any contradictions that suggest different filming locations
within the same purported video.
Output ONLY the JSON object, no markdown fences.

Frame observations:
{observations}"""

_GEO_ETHICAL_CAVEAT = (
    "IMPORTANT: Geolocation output is indicative only and must not be used to identify "
    "individuals or their whereabouts. Results are based solely on visual scene analysis "
    "and carry significant uncertainty. Do not publish geolocation findings without "
    "independent corroboration from primary sources."
)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class GeolocationTool(VisionTool):
    TOOL_NAME = "Geolocation"
    INPUTS = ["Image", "Keyframes"]

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
        print("GeolocationTool: analysing frame...", file=sys.stderr)
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        result = _ollama_post(self.host, {
            "model": self.model,
            "prompt": _GEO_PROMPT,
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
        print("GeolocationTool: synthesising...", file=sys.stderr)
        obs_text = "\n".join(
            f"Frame {i + 1}: {json.dumps(o)}" for i, o in enumerate(observations)
        )
        result = _ollama_post(self.host, {
            "model": self.synth_model,
            "prompt": _GEO_SYNTH_PROMPT.format(observations=obs_text),
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
                explanation="No images available for geolocation.", has_run=0,
            )

        observations: list[dict] = []
        for img_path in sources:
            try:
                observations.append(self._analyse_frame(img_path))
            except Exception as e:
                print(f"GeolocationTool: error on {img_path}: {e}", file=sys.stderr)

        if not observations:
            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS, None,
                explanation="All frame analyses failed.", has_run=0,
            )

        output = observations[0] if len(observations) == 1 else self._synthesise(observations)

        country = output.get("country")
        city = output.get("city_or_area")
        landmark = output.get("landmark")
        cues = output.get("visual_cues") or []

        if landmark:
            confidence = 65
        elif city:
            confidence = 50
        elif country:
            confidence = 40
        else:
            confidence = 20

        consistency_note = output.get("consistency", "")
        has_inconsistency = bool(
            consistency_note and consistency_note.lower() not in ("none", "consistent", "")
        )
        if has_inconsistency:
            confidence = max(confidence - 15, 10)

        return _make_tool_json(
            self.TOOL_NAME, self.INPUTS,
            output=output,
            explanation=(
                f"Location cues analysed across {len(observations)} frame(s). "
                + ("Inconsistent locations across frames detected." if has_inconsistency else "")
                + " " + _GEO_ETHICAL_CAVEAT
            ),
            confidence=confidence,
            confidence_explanation=(
                "Visual geolocation via a general-purpose vision model is inherently uncertain. "
                f"Confidence is based on specificity of identified cues ({len(cues)} cue(s) found). "
                "Results must be corroborated via reverse image search or primary source verification "
                "before any editorial use. " + _GEO_ETHICAL_CAVEAT
            ),
            corroborating_tools=["Weather Detection", "Description", "Keyframes", "Metadata Gatherer"],
        )