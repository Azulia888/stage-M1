"""
ner.py — NERTool: named-entity recognition over aggregated text fields.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from data_manager import DataManager
from vision_tools.base import (
    VisionTool, _make_tool_json, _ollama_post, _ollama_response,
    OLLAMA_HOST, OLLAMA_SYNTH_MODEL,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_NER_ENTITY_TYPES = (
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "DATE",
    "EVENT",
    "WORK_OF_ART",
    "LAW",
    "PRODUCT",
    "LANGUAGE",
    "NATIONALITY",
)

_NER_PROMPT_TEMPLATE = """\
You are a named-entity recognition (NER) system for a fact-checking newsroom.
Extract all named entities from the text below and return ONLY a valid JSON object.
The JSON must have one key per entity type and an array of unique string values.
Use only these entity types: {types}.
If no entities of a type are found, omit that key.
Output ONLY the JSON object — no explanation, no markdown fences.

Text:
{text}"""


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class NERTool(VisionTool):
    TOOL_NAME = "NER"
    INPUTS = ["Image", "Video", "Keyframes", "Transcript", "Description", "OCR", "Metadata"]

    def __init__(self, model: str = OLLAMA_SYNTH_MODEL, host: str = OLLAMA_HOST):
        self.model = model
        self.host = host

    def _collect_text(self, data: DataManager) -> str:
        parts: list[str] = []
        if data.transcript:
            parts.append(f"[Transcript]\n{data.transcript}")
        if data.description:
            parts.append(f"[Description]\n{data.description}")
        if data.ocr:
            parts.append(f"[OCR]\n{data.ocr}")
        if data.metadata:
            parts.append(f"[Metadata]\n{data.metadata}")
        if data.metadata_path:
            try:
                raw = Path(data.metadata_path).read_text(encoding="utf-8", errors="replace")
                sidecar = json.loads(raw)
                for key in ("title", "description", "uploader", "channel", "tags"):
                    val = sidecar.get(key)
                    if val and isinstance(val, (str, list)):
                        if isinstance(val, list):
                            val = ", ".join(str(v) for v in val)
                        parts.append(f"[Sidecar:{key}]\n{val}")
            except Exception:
                pass
        return "\n\n".join(parts)

    def run(self, data: DataManager) -> dict | None:
        text = self._collect_text(data)
        if not text.strip():
            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS, None,
                explanation="No text available for NER.",
                has_run=0,
            )

        max_chars = 12_000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[...truncated]"

        prompt = _NER_PROMPT_TEMPLATE.format(
            types=", ".join(_NER_ENTITY_TYPES),
            text=text,
        )

        try:
            result = _ollama_post(self.host, {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }, timeout=120)
            raw = _ollama_response(result).strip()

            if raw.startswith("```"):
                lines = raw.split("\n")
                inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                raw = "\n".join(inner).strip()

            entities = json.loads(raw)

            for k in list(entities.keys()):
                if not isinstance(entities[k], list):
                    entities[k] = [str(entities[k])]
                entities[k] = [str(v).strip() for v in entities[k] if str(v).strip()]
                if not entities[k]:
                    del entities[k]

            total = sum(len(v) for v in entities.values())
            explanation = (
                f"Extracted {total} named entities across "
                f"{len(entities)} type(s) from aggregated text fields."
            )

            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS,
                output=entities,
                explanation=explanation,
                confidence=70,
                confidence_explanation=(
                    "NER accuracy depends on the quality of upstream text fields. "
                    "Transcript and OCR errors propagate directly. "
                    "LLM-based extraction can hallucinate entities not present in the text; "
                    "all results should be verified against the source."
                ),
                corroborating_tools=["Transcript", "Description", "OCR", "Metadata Gatherer"],
            )

        except json.JSONDecodeError as e:
            print(f"NERTool: model did not return valid JSON — {e}", file=sys.stderr)
            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS, None,
                explanation=f"Model returned non-JSON output: {e}",
                has_run=0,
            )
        except Exception as e:
            print(f"NERTool error: {e}", file=sys.stderr)
            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS, None, explanation=str(e), has_run=0,
            )