"""
ner.py — NERTool: named-entity recognition over aggregated text fields,
augmented with a visual NER pass over keyframes.

Backends
--------
"llm"      — LLM-only via Ollama (original behaviour)
"nltk"     — NLTK ne_chunk + averaged_perceptron_tagger
"spacy"    — spaCy (model resolved from SPACY_MODEL env var, default en_core_web_sm)
"ensemble" — NLTK + spaCy results fed to an LLM together with the source text
             for a final consolidated extraction

Visual pass
-----------
Before the text-based backend runs, each keyframe (or the single image) is
sent to the vision model with a prompt restricted to entities visible in the
scene itself (people, organisations, locations, events, etc. — not text in
the image, which OCR already covers). The merged per-frame results are
injected as an extra section into the text fed to the final backend, so the
LLM (or NLTK/spaCy, as plain proper nouns) has access to both textual and
visual evidence.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from typing import Literal

from data_manager import DataManager
from vision_tools.base import (
    VisionTool, _make_tool_json, _ollama_post, _ollama_response,
    OLLAMA_HOST, OLLAMA_SYNTH_MODEL, OLLAMA_VISION_MODEL,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SPACY_MODEL = os.environ.get("SPACY_MODEL", "en_core_web_sm")

Backend = Literal["llm", "nltk", "spacy", "ensemble"]

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

# NLTK tag → canonical type
_NLTK_TAG_MAP: dict[str, str] = {
    "PERSON":       "PERSON",
    "ORGANIZATION": "ORGANIZATION",
    "GPE":          "LOCATION",
    "LOCATION":     "LOCATION",
    "FACILITY":     "LOCATION",
    "GSP":          "LOCATION",
}

# spaCy label → canonical type (only labels we care about)
_SPACY_LABEL_MAP: dict[str, str] = {
    "PERSON":      "PERSON",
    "ORG":         "ORGANIZATION",
    "GPE":         "LOCATION",
    "LOC":         "LOCATION",
    "FAC":         "LOCATION",
    "DATE":        "DATE",
    "TIME":        "DATE",
    "EVENT":       "EVENT",
    "WORK_OF_ART": "WORK_OF_ART",
    "LAW":         "LAW",
    "PRODUCT":     "PRODUCT",
    "LANGUAGE":    "LANGUAGE",
    "NORP":        "NATIONALITY",
}


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_LLM_PROMPT = """\
You are a named-entity recognition (NER) system for a fact-checking newsroom.
Extract all named entities from the text below and return ONLY a valid JSON object.
The JSON must have one key per entity type and an array of unique string values.
Use only these entity types: {types}.
If no entities of a type are found, omit that key.
Omit very common entities, such as days of the week or the current time.
Only report entities that could be interesting to know about in a fact-checking environment.
Output ONLY the JSON object — no explanation, no markdown fences.

Text:
{text}"""

_ENSEMBLE_PROMPT = """\
You are a named-entity recognition (NER) system for a fact-checking newsroom.
You are given:
1. Source text (transcript, description, OCR, metadata, visual entities).
2. NER results from NLTK.
3. NER results from spaCy.

Your task: produce a single, deduplicated, corrected JSON object of named entities.
Use ONLY these entity types: {types}.
Fix any obvious errors from the automated tools (wrong type, split names, artefacts).
Add any clearly missed entities you can see in the source text.
Remove any hallucinations or garbage strings.
If no entities of a type are found, omit that key.
Output ONLY the JSON object — no explanation, no markdown fences.

SOURCE TEXT:
{text}

NLTK RESULTS:
{nltk_result}

SPACY RESULTS:
{spacy_result}"""

_VISUAL_NER_PROMPT = """\
You are a named-entity recognition (NER) system for a fact-checking newsroom.
Examine this image frame and identify named entities visible in the scene itself \
— recognisable people, organisations (logos, uniforms, liveries), locations or \
landmarks, named events, products, and flags or symbols indicating nationality.
Do not report text written in the image (signs, captions, subtitles); that is \
handled separately by OCR.
Return ONLY a valid JSON object with one key per entity type and an array of \
unique string values. Use only these entity types: {types}.
If no entities of a type are found, omit that key.
Do not guess beyond what the visual evidence supports.
Output ONLY the JSON object — no explanation, no markdown fences."""


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def _dedupe(entities: dict[str, list[str]]) -> dict[str, list[str]]:
    return {k: sorted(set(v)) for k, v in entities.items() if v}


def _run_visual_frame(
    image_path: str, host: str, model: str, timeout: int = 120
) -> dict[str, list[str]]:
    """Run visual-only NER on a single image/keyframe via the vision model."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    prompt = _VISUAL_NER_PROMPT.format(types=", ".join(_NER_ENTITY_TYPES))
    result = _ollama_post(host, {
        "model": model,
        "prompt": prompt,
        "images": [b64],
        "stream": False,
    }, timeout=timeout)
    raw = _ollama_response(result).strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
    try:
        entities: dict = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    cleaned: dict[str, list[str]] = {}
    for k, v in entities.items():
        if not isinstance(v, list):
            v = [str(v)]
        vals = [str(x).strip() for x in v if str(x).strip()]
        if vals:
            cleaned[k] = vals
    return _dedupe(cleaned)


def _run_nltk(text: str) -> dict[str, list[str]]:
    try:
        import nltk
        from nltk import ne_chunk, pos_tag, word_tokenize
        from nltk.tree import Tree

        for resource in (
            "punkt", "punkt_tab", "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng", "maxent_ne_chunker",
            "maxent_ne_chunker_tab", "words",
        ):
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass

        entities: dict[str, list[str]] = {}
        chunked = ne_chunk(pos_tag(word_tokenize(text)))
        for subtree in chunked:
            if isinstance(subtree, Tree):
                label = _NLTK_TAG_MAP.get(subtree.label())
                if label:
                    name = " ".join(token for token, _ in subtree.leaves())
                    entities.setdefault(label, []).append(name)
        return _dedupe(entities)

    except ImportError:
        raise RuntimeError("nltk is not installed. Run: pip install nltk")


def _run_spacy(text: str) -> dict[str, list[str]]:
    try:
        import spacy
    except ImportError:
        raise RuntimeError("spacy is not installed. Run: pip install spacy")

    try:
        nlp = spacy.load(SPACY_MODEL)
    except OSError:
        raise RuntimeError(
            f"spaCy model '{SPACY_MODEL}' not found. "
            f"Run: python -m spacy download {SPACY_MODEL}"
        )

    doc = nlp(text[:1_000_000])
    entities: dict[str, list[str]] = {}
    for ent in doc.ents:
        label = _SPACY_LABEL_MAP.get(ent.label_)
        if label:
            entities.setdefault(label, []).append(ent.text.strip())
    return _dedupe(entities)


def _run_llm(
    text: str,
    host: str,
    model: str,
    prompt_template: str = _LLM_PROMPT,
    extra_kwargs: dict | None = None,
) -> dict[str, list[str]]:
    fmt_kwargs = {"types": ", ".join(_NER_ENTITY_TYPES), "text": text}
    if extra_kwargs:
        fmt_kwargs.update(extra_kwargs)
    prompt = prompt_template.format(**fmt_kwargs)
    result = _ollama_post(host, {"model": model, "prompt": prompt, "stream": False}, timeout=120)
    raw = _ollama_response(result).strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
    entities: dict = json.loads(raw)
    cleaned: dict[str, list[str]] = {}
    for k, v in entities.items():
        if not isinstance(v, list):
            v = [str(v)]
        vals = [str(x).strip() for x in v if str(x).strip()]
        if vals:
            cleaned[k] = vals
    return _dedupe(cleaned)


def _run_ensemble(
    text: str,
    host: str,
    model: str,
) -> tuple[dict, dict, dict]:
    """Returns (nltk_result, spacy_result, llm_final)."""
    nltk_result = _run_nltk(text)
    spacy_result = _run_spacy(text)
    final = _run_llm(
        text,
        host,
        model,
        prompt_template=_ENSEMBLE_PROMPT,
        extra_kwargs={
            "nltk_result":  json.dumps(nltk_result,  ensure_ascii=False),
            "spacy_result": json.dumps(spacy_result, ensure_ascii=False),
        },
    )
    return nltk_result, spacy_result, final


# ---------------------------------------------------------------------------
# Confidence helpers
# ---------------------------------------------------------------------------

_BACKEND_CONFIDENCE: dict[str, int] = {
    "llm":      70,
    "nltk":     60,
    "spacy":    65,
    "ensemble": 80,
}

_BACKEND_CONFIDENCE_EXPLANATION: dict[str, str] = {
    "llm": (
        "LLM-based extraction can hallucinate entities not present in the text; "
        "all results should be verified against the source."
    ),
    "nltk": (
        "NLTK rule-based chunker has limited entity type coverage and struggles "
        "with unusual names or non-English text. Results should be verified."
    ),
    "spacy": (
        "spaCy statistical NER is generally reliable for common entity types "
        "but may miss rare names or produce label errors. Verify against source text."
    ),
    "ensemble": (
        "NLTK and spaCy results were merged and corrected by an LLM. "
        "This reduces false positives and missed entities relative to any single backend, "
        "but LLM correction may introduce new errors. Verify all results."
    ),
}


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class NERTool(VisionTool):
    TOOL_NAME = "NER"
    INPUTS = ["Image", "Video", "Keyframes", "Transcript", "Description", "OCR", "Metadata"]

    def __init__(
        self,
        model: str = OLLAMA_SYNTH_MODEL,
        vision_model: str = OLLAMA_VISION_MODEL,
        host: str = OLLAMA_HOST,
        backend: Backend = "llm",
        include_visual_entities: bool = True,
        visual_frame_timeout: int = 120,
    ):
        self.model = model
        self.vision_model = vision_model
        self.host = host
        self.backend: Backend = backend
        self.include_visual_entities = include_visual_entities
        self.visual_frame_timeout = visual_frame_timeout

    def _collect_visual_entities(self, data: DataManager) -> dict[str, list[str]]:
        """Run visual-only NER on each keyframe (or the single image) and merge results."""
        sources: list[str] = data.keyframes if data.isVideo else [data.originalMedia]
        if not sources:
            return {}

        merged: dict[str, list[str]] = {}
        for i, img_path in enumerate(sources, 1):
            print(
                f"NERTool: visual NER on frame {i}/{len(sources)} "
                f"({Path(img_path).name}) ...",
                file=sys.stderr,
            )
            try:
                frame_entities = _run_visual_frame(
                    img_path, self.host, self.vision_model, self.visual_frame_timeout
                )
            except Exception as e:
                print(f"NERTool: visual NER error on {img_path}: {e}", file=sys.stderr)
                continue
            for k, v in frame_entities.items():
                merged.setdefault(k, []).extend(v)

        return _dedupe(merged)

    def _collect_text(
        self, data: DataManager, visual_entities: dict[str, list[str]] | None = None
    ) -> str:
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
        if visual_entities:
            parts.append(
                "[Visual entities detected in keyframes]\n"
                + json.dumps(visual_entities, ensure_ascii=False)
            )
        return "\n\n".join(parts)

    def run(self, data: DataManager) -> dict | None:
        visual_entities: dict[str, list[str]] = {}
        if self.include_visual_entities:
            visual_entities = self._collect_visual_entities(data)

        text = self._collect_text(data, visual_entities)
        if not text.strip():
            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS, None,
                explanation="No text available for NER.",
                has_run=0,
            )

        max_chars = 12_000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[...truncated]"

        try:
            extra_output: dict = {}
            if visual_entities:
                extra_output["visual_intermediate"] = visual_entities

            if self.backend == "llm":
                entities = _run_llm(text, self.host, self.model)

            elif self.backend == "nltk":
                entities = _run_nltk(text)

            elif self.backend == "spacy":
                entities = _run_spacy(text)

            elif self.backend == "ensemble":
                nltk_result, spacy_result, entities = _run_ensemble(text, self.host, self.model)
                extra_output["nltk_intermediate"] = nltk_result
                extra_output["spacy_intermediate"] = spacy_result
            else:
                raise ValueError(f"Unknown NER backend: {self.backend!r}")

            total = sum(len(v) for v in entities.values())
            explanation = (
                f"Extracted {total} named entities across {len(entities)} type(s) "
                f"from aggregated text fields using backend '{self.backend}'."
                + (
                    " Visual NER was run on keyframes and merged into the input text."
                    if visual_entities else ""
                )
            )

            output = {"entities": entities, **extra_output}

            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS,
                output=output,
                explanation=explanation,
                confidence=_BACKEND_CONFIDENCE[self.backend],
                confidence_explanation=_BACKEND_CONFIDENCE_EXPLANATION[self.backend],
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