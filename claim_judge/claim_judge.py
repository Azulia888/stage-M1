#!/usr/bin/env python3
"""
claim_judge.py — LLM-based claim judge for the AFC pipeline.

Takes a DataManager .pkl (produced by projet/vision_module.py) and a textual
claim about the media it describes, then asks a local Ollama model to rate
the claim on three axes:

    Authenticity     -1 (AI-generated / manipulated) .. +1 (authentic)
    Context Coverage -1 (claimed context contradicted) .. +1 (context matches)
    Veracity         -1 (claim is false) .. +1 (claim is true)

For each axis the model must explain its reasoning, name which tool(s) in
the DataManager informed the decision, and give a separate confidence score
(0-100) for that specific rating.

Usage
-----
    python claim_judge.py <datamanager.pkl> "<claim text>" <output_path>

    <output_path> is a base path; the report is written to
    "<output_path>.md" (human-readable) and "<output_path>.json" (machine
    readable). Any extension you give is stripped and replaced.

Ablation testing
-----------------
See ENABLED_TOOLS below. Comment out (or remove) any tool name from that
list and re-run the judge on the same DataManager/claim pair to see how
much a given tool's evidence actually moves the ratings — the tool's data
is simply left out of what the LLM is shown, nothing else changes.

Environment variables
----------------------
    OLLAMA_HOST         default: http://localhost:11434
    OLLAMA_JUDGE_MODEL   model used for judging (falls back to
                         OLLAMA_SYNTH_MODEL, then a hardcoded default)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Make the sibling "projet" package importable, so we can unpickle
# DataManager instances that were pickled from that module path.
# ---------------------------------------------------------------------------

_PROJET_DIR = Path(__file__).resolve().parent.parent / "projet"
if str(_PROJET_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJET_DIR))

try:
    from data_manager import DataManager  # noqa: E402
except ImportError as e:  # pragma: no cover
    sys.exit(
        "Could not import DataManager. Expected to find 'data_manager.py' "
        f"in '{_PROJET_DIR}'. Original error: {e}"
    )


# ---------------------------------------------------------------------------
# Ollama config
# ---------------------------------------------------------------------------

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_JUDGE_MODEL = os.environ.get(
    "OLLAMA_JUDGE_MODEL",
    os.environ.get("OLLAMA_SYNTH_MODEL", "qwen3.5:2b"),
)


# ---------------------------------------------------------------------------
# TOOL LIST — this is the ablation switchboard.
#
# ALL_TOOLS documents every tool the vision pipeline can populate a
# DataManager with (see projet/vision_tools/). ENABLED_TOOLS is the subset
# the judge is actually allowed to see evidence from. To test whether a
# given tool matters to the judge's decision, comment its line out of
# ENABLED_TOOLS and re-run the judge on the same DataManager + claim: its
# evidence block will simply be omitted from what the LLM is shown.
#
# Tools that are commented out here are excluded even if the DataManager
# happens to contain data for them.
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    "Transcript",
    "Keyframes",
    "Metadata Gatherer",
    "Metadata Analyzer",
    "OCR",
    "Description",
    "Lipsync Detection",
    "AI Detection",          # stub in the current pipeline — no real output yet
    "Deepfake Detection",    # stub in the current pipeline — no real output yet
    "Weather Detection",
    "Geolocation",
    "Facial Recognition",    # stub in the current pipeline — no real output yet
    "NER",
    "Reverse Image Search",
    "Knowledge Graph",
]

ENABLED_TOOLS = [
    "Transcript",
    "Keyframes",
    "Metadata Gatherer",
    "Metadata Analyzer",
    "OCR",
    "Description",
    "Lipsync Detection",
    # "AI Detection",        # stub — currently never runs, nothing to show
    # "Deepfake Detection",  # stub — currently never runs, nothing to show
    "Weather Detection",
    "Geolocation",
    # "Facial Recognition",  # stub — currently never runs, nothing to show
    "NER",
    "Reverse Image Search",
    "Knowledge Graph",
]

# Short human-readable reminder of what each tool actually measures, given
# to the LLM alongside its output so it can reason about *why* a tool's
# finding is or isn't relevant to a given axis.
TOOL_ROLE_HINTS: dict[str, str] = {
    "Transcript": "Speech-to-text transcript of the media's audio track.",
    "Keyframes": "Representative still frames sampled from the video.",
    "Metadata Gatherer": "Raw technical/file metadata (codec, timestamps, EXIF/GPS, sidecar upload info).",
    "Metadata Analyzer": "LLM summary of the raw metadata, flagging provenance/timestamp anomalies.",
    "OCR": "Text extracted from on-screen signs, captions, or overlays.",
    "Description": "Vision-model description of what is visually happening in the media.",
    "Lipsync Detection": "SyncNet audio-visual sync analysis; flags dubbed/manipulated speech.",
    "AI Detection": "Detector for AI-generated imagery (not yet implemented).",
    "Deepfake Detection": "Deepfake face-swap detector (not yet implemented).",
    "Weather Detection": "Vision-model estimate of weather, time of day, and season shown.",
    "Geolocation": "Vision-model estimate of the filming location from visual cues.",
    "Facial Recognition": "Facial identification against known individuals (not yet implemented).",
    "NER": "Named entities (people, places, organizations, etc.) found in text and visuals.",
    "Reverse Image Search": "Web search for prior appearances of the top keyframe (via SerpApi).",
    "Knowledge Graph": "Wikidata enrichment of the named entities found by NER.",
}

MAX_TOOL_OUTPUT_CHARS = 1200
MAX_EXPLANATION_CHARS = 500


# ---------------------------------------------------------------------------
# Ollama HTTP helper (self-contained; mirrors projet/vision_tools/base.py)
# ---------------------------------------------------------------------------

def _ollama_generate(
    host: str,
    model: str,
    prompt: str,
    timeout: int = 300,
    think: bool = False,
) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False, "think": think}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    result_container: list = [None]
    error_container: list = [None]

    def do_request():
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result_container[0] = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            error_container[0] = RuntimeError(f"Ollama HTTP {e.code}: {body}")
        except urllib.error.URLError as e:
            error_container[0] = RuntimeError(
                f"Could not reach Ollama at '{host}': {e.reason}. "
                "Is 'ollama serve' running?"
            )
        except Exception as e:
            error_container[0] = e

    thread = threading.Thread(target=do_request, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise RuntimeError(
            f"Ollama request timed out after {timeout}s (model={model})."
        )
    if error_container[0] is not None:
        raise error_container[0]

    result = result_container[0] or {}
    text = (result.get("response") or "").strip()
    if text:
        return text
    thinking = (result.get("thinking") or "").strip()
    if thinking:
        print("WARNING: 'response' was empty, falling back to 'thinking' field.", file=sys.stderr)
        return thinking
    return ""


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
    return text


def _extract_json_object(text: str) -> str:
    """Best-effort extraction of the outermost {...} block from model output."""
    text = _strip_code_fences(text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


# ---------------------------------------------------------------------------
# Evidence extraction from the DataManager
# ---------------------------------------------------------------------------

def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ...[truncated]"


def _summarise_output(tool_name: str, output) -> str:
    """Render a tool's raw Output field into compact text for the prompt."""
    if output is None:
        return "(no output)"

    # Keyframes: a list of local file paths is noise, not evidence.
    if tool_name == "Keyframes" and isinstance(output, list):
        return f"{len(output)} keyframe(s) extracted from the video."

    if isinstance(output, str):
        text = output.strip() or "(empty)"
    else:
        try:
            text = json.dumps(output, ensure_ascii=False, indent=None, default=str)
        except TypeError:
            text = str(output)

    return _truncate(text, MAX_TOOL_OUTPUT_CHARS)


@dataclass
class EvidenceBundle:
    included_tools: list[str] = field(default_factory=list)
    unavailable_tools: list[str] = field(default_factory=list)
    evidence_text: str = ""
    per_tool_summary: dict[str, str] = field(default_factory=dict)


def build_evidence(data: DataManager, enabled_tools: list[str]) -> EvidenceBundle:
    bundle = EvidenceBundle()
    blocks: list[str] = []

    for tool_name in enabled_tools:
        tool_json = data.toolResult.get(tool_name)

        if tool_json is None:
            bundle.unavailable_tools.append(f"{tool_name} (no data in this DataManager)")
            continue

        if not tool_json.get("hasRun", 0):
            reason = tool_json.get("Explanation") or "did not run"
            bundle.unavailable_tools.append(f"{tool_name} ({reason})")
            continue

        output_text = _summarise_output(tool_name, tool_json.get("Output"))
        explanation = _truncate((tool_json.get("Explanation") or "").strip(), MAX_EXPLANATION_CHARS)
        confidence = tool_json.get("Confidence", 0)
        confidence_explanation = _truncate(
            (tool_json.get("ConfidenceExplanation") or "").strip(), MAX_EXPLANATION_CHARS
        )
        role_hint = TOOL_ROLE_HINTS.get(tool_name, "")

        block = (
            f"### Tool: {tool_name}\n"
            f"What it measures: {role_hint}\n"
            f"Output: {output_text}\n"
            f"Tool's own explanation: {explanation or '(none provided)'}\n"
            f"Tool's own confidence: {confidence}/100"
            + (f" — {confidence_explanation}" if confidence_explanation else "")
        )
        blocks.append(block)
        bundle.included_tools.append(tool_name)
        bundle.per_tool_summary[tool_name] = output_text

    bundle.evidence_text = "\n\n".join(blocks) if blocks else "(no tool evidence available)"
    return bundle


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_JSON_SCHEMA_EXAMPLE = """{
  "authenticity": {
    "rating": 0.0,
    "explanation": "2-4 sentences citing which tool(s) support this and how.",
    "contributing_tools": ["Geolocation", "Metadata Analyzer"],
    "confidence": 50,
    "confidence_explanation": "Why you are or are not confident in this specific rating."
  },
  "context_coverage": {
    "rating": 0.0,
    "explanation": "...",
    "contributing_tools": [],
    "confidence": 50,
    "confidence_explanation": "..."
  },
  "veracity": {
    "rating": 0.0,
    "explanation": "...",
    "contributing_tools": [],
    "confidence": 50,
    "confidence_explanation": "..."
  }
}"""


def build_prompt(claim: str, media_info: str, evidence: EvidenceBundle) -> str:
    unavailable_text = (
        "\n".join(f"- {t}" for t in evidence.unavailable_tools)
        if evidence.unavailable_tools
        else "(none — all configured tools provided data)"
    )

    return f"""\
You are an expert fact-checking judge working for a newsroom. You will be shown \
a CLAIM made about a piece of media, along with forensic evidence gathered by a \
set of automated analysis tools that already examined that media.

MEDIA:
{media_info}

CLAIM TO JUDGE:
"{claim}"

EVIDENCE FROM ANALYSIS TOOLS:
{evidence.evidence_text}

TOOLS WITH NO USABLE DATA FOR THIS RUN (do not invent findings from these; treat \
them as simply absent):
{unavailable_text}

Rate the claim on three independent axes. Each rating is a float from -1.0 to \
1.0 (decimals allowed):

1. authenticity: -1.0 means the media itself appears AI-generated, synthetic, or \
manipulated; +1.0 means the media appears authentic and unaltered; 0.0 means the \
evidence is insufficient or contradictory.
2. context_coverage: -1.0 means the context asserted by the claim (who/what/where/ \
when/why) is contradicted by what the tools found in the media; +1.0 means the \
claimed context is fully consistent with the media; 0.0 means there isn't enough \
evidence to judge context.
3. veracity: -1.0 means the factual assertion in the claim is false; +1.0 means it \
is true; 0.0 means it cannot be determined from the available evidence.

For each axis, provide:
- "rating": the float score described above.
- "explanation": 2-4 sentences. You MUST explicitly name which tool(s), if any, \
informed this rating and describe how (e.g. "the Geolocation tool identified \
landmarks consistent with Paris, which supports the claimed location"). If no \
tool evidence was relevant or available, say so plainly instead of guessing.
- "contributing_tools": a JSON array with the exact tool names (from the EVIDENCE \
section above) that most influenced this specific rating. Use an empty array if \
none did.
- "confidence": an integer 0-100 reflecting how confident you are in THIS rating \
specifically (not the claim overall).
- "confidence_explanation": a short reason for that confidence level (e.g. \
limited evidence, tools disagreed, strong corroboration across multiple tools).

Base your judgment only on the evidence provided above — do not use outside \
knowledge about the claim's subject matter unless it is common, uncontested \
knowledge needed to interpret the evidence itself. Do not reference or credit any \
tool that is not listed in the EVIDENCE section.

Respond with ONLY a single valid JSON object, no markdown fences, no commentary \
before or after it, using exactly this schema:

{_JSON_SCHEMA_EXAMPLE}"""


# ---------------------------------------------------------------------------
# Response parsing / validation
# ---------------------------------------------------------------------------

class JudgeParseError(Exception):
    def __init__(self, message: str, raw_response: str):
        super().__init__(message)
        self.raw_response = raw_response


_AXES = ("authenticity", "context_coverage", "veracity")


def _clamp(value, lo, hi):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, value))


def parse_judge_response(raw: str, known_tools: list[str]) -> dict:
    candidate = _extract_json_object(raw)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise JudgeParseError(f"Model did not return valid JSON: {e}", raw)

    if not isinstance(parsed, dict):
        raise JudgeParseError("Model JSON was not an object.", raw)

    result: dict = {}
    for axis in _AXES:
        axis_data = parsed.get(axis)
        if not isinstance(axis_data, dict):
            axis_data = {}

        rating = _clamp(axis_data.get("rating", 0.0), -1.0, 1.0)
        confidence = int(_clamp(axis_data.get("confidence", 0), 0, 100))
        explanation = str(axis_data.get("explanation") or "").strip() or "(model gave no explanation)"
        confidence_explanation = str(axis_data.get("confidence_explanation") or "").strip()

        raw_contrib = axis_data.get("contributing_tools", [])
        if not isinstance(raw_contrib, list):
            raw_contrib = [raw_contrib] if raw_contrib else []
        contributing = [str(t).strip() for t in raw_contrib if str(t).strip()]

        # Flag tools the model claims helped but that had no data this run —
        # useful both as a hallucination check and as an ablation sanity check.
        valid_contrib = [t for t in contributing if t in known_tools]
        hallucinated = [t for t in contributing if t not in known_tools]

        result[axis] = {
            "rating": round(rating, 3),
            "explanation": explanation,
            "contributing_tools": valid_contrib,
            "hallucinated_tool_references": hallucinated,
            "confidence": confidence,
            "confidence_explanation": confidence_explanation,
        }

    return result


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _axis_label(axis: str, rating: float) -> str:
    labels = {
        "authenticity": [
            (0.6, "Likely authentic"),
            (0.2, "Leans authentic"),
            (-0.2, "Inconclusive"),
            (-0.6, "Leans AI-generated / manipulated"),
            (float("-inf"), "Likely AI-generated / manipulated"),
        ],
        "context_coverage": [
            (0.6, "Context strongly supported by the media"),
            (0.2, "Context mostly supported"),
            (-0.2, "Inconclusive"),
            (-0.6, "Context mostly contradicted"),
            (float("-inf"), "Context strongly contradicted by the media"),
        ],
        "veracity": [
            (0.6, "Likely true"),
            (0.2, "Leans true"),
            (-0.2, "Inconclusive"),
            (-0.6, "Leans false"),
            (float("-inf"), "Likely false"),
        ],
    }
    for threshold, label in labels[axis]:
        if rating >= threshold:
            return label
    return "Inconclusive"


_AXIS_TITLES = {
    "authenticity": "Authenticity",
    "context_coverage": "Context Coverage",
    "veracity": "Veracity",
}


def render_markdown_report(
    claim: str,
    media_info: str,
    model: str,
    evidence: EvidenceBundle,
    ratings: dict,
    timestamp: str,
) -> str:
    lines: list[str] = []
    lines.append("# Claim Judge Report")
    lines.append("")
    lines.append(f"**Claim:** {claim}")
    lines.append("")
    lines.append(f"**Media:** {media_info}")
    lines.append(f"**Judge model:** {model} (via Ollama)")
    lines.append(f"**Generated:** {timestamp}")
    lines.append("")

    for axis in _AXES:
        r = ratings[axis]
        lines.append(f"## {_AXIS_TITLES[axis]}: {r['rating']:+.2f}  —  {_axis_label(axis, r['rating'])}")
        lines.append("")
        lines.append(f"**Explanation:** {r['explanation']}")
        lines.append("")
        contrib = r["contributing_tools"] or ["(none identified)"]
        lines.append(f"**Tools that informed this rating:** {', '.join(contrib)}")
        if r["hallucinated_tool_references"]:
            lines.append(
                "> ⚠ The model also referenced tool(s) with no data available this run: "
                + ", ".join(r["hallucinated_tool_references"])
                + ". Ignored."
            )
        lines.append("")
        lines.append(f"**Confidence:** {r['confidence']}/100")
        if r["confidence_explanation"]:
            lines.append(f"**Why:** {r['confidence_explanation']}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Evidence Considered")
    lines.append("")
    lines.append(
        f"Tools included ({len(evidence.included_tools)}): "
        + (", ".join(evidence.included_tools) if evidence.included_tools else "(none)")
    )
    lines.append("")
    lines.append("Tools excluded or unavailable for this run:")
    if evidence.unavailable_tools:
        for t in evidence.unavailable_tools:
            lines.append(f"- {t}")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append(
        "_Note: any tool commented out of `ENABLED_TOOLS` in claim_judge.py will "
        "show up here as excluded, even if the DataManager contains data for it. "
        "This is the mechanism for ablation testing — re-run with a tool removed "
        "from `ENABLED_TOOLS` and compare the ratings above._"
    )
    lines.append("")

    return "\n".join(lines)


def render_json_report(
    dm_path: str,
    claim: str,
    media_info: str,
    model: str,
    evidence: EvidenceBundle,
    ratings: dict,
    timestamp: str,
) -> dict:
    return {
        "timestamp": timestamp,
        "datamanager_path": dm_path,
        "media_info": media_info,
        "judge_model": model,
        "claim": claim,
        "ratings": ratings,
        "tools_included": evidence.included_tools,
        "tools_excluded_or_unavailable": evidence.unavailable_tools,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def describe_media(data: DataManager) -> str:
    kind = "video" if data.isVideo else "image"
    name = Path(data.originalMedia).name if data.originalMedia else "(unknown file)"
    parts = [f"{kind.capitalize()} file: {name}"]
    if data.isVideo and data.keyframes:
        parts.append(f"{len(data.keyframes)} keyframe(s) available")
    return " — ".join(parts)


def resolve_output_paths(output_arg: str) -> tuple[Path, Path]:
    p = Path(output_arg)
    base = p.with_suffix("") if p.suffix.lower() in (".md", ".json", ".txt") else p
    return base.with_suffix(".md"), base.with_suffix(".json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Judge a claim about a media file using its DataManager evidence and a local Ollama LLM."
    )
    parser.add_argument("datamanager", help="Path to a DataManager .pkl file produced by the vision pipeline.")
    parser.add_argument("claim", help="The claim to judge, as a string.")
    parser.add_argument(
        "output",
        help="Base path to save the report to. Writes '<output>.md' and '<output>.json'.",
    )
    parser.add_argument("--host", default=OLLAMA_HOST, help=f"Ollama host (default: {OLLAMA_HOST})")
    parser.add_argument("--model", default=OLLAMA_JUDGE_MODEL, help=f"Ollama model (default: {OLLAMA_JUDGE_MODEL})")
    parser.add_argument("--timeout", type=int, default=300, help="Ollama request timeout in seconds (default: 300).")
    parser.add_argument("--think", action="store_true", help="Allow the model's extended-thinking mode, if supported.")
    args = parser.parse_args()

    dm_path = Path(args.datamanager)
    if not dm_path.exists():
        print(f"error: DataManager file not found: {dm_path}", file=sys.stderr)
        return 1

    try:
        data = DataManager.load(str(dm_path))
    except Exception as e:
        print(f"error: could not load DataManager from '{dm_path}': {e}", file=sys.stderr)
        return 1

    media_info = describe_media(data)
    evidence = build_evidence(data, ENABLED_TOOLS)
    prompt = build_prompt(args.claim, media_info, evidence)

    print(f"Judging claim with model '{args.model}' at '{args.host}' ...", file=sys.stderr)
    print(f"Tools considered: {', '.join(evidence.included_tools) or '(none)'}", file=sys.stderr)

    try:
        raw = _ollama_generate(args.host, args.model, prompt, timeout=args.timeout, think=args.think)
    except Exception as e:
        print(f"error: Ollama call failed: {e}", file=sys.stderr)
        return 1

    try:
        ratings = parse_judge_response(raw, evidence.included_tools)
    except JudgeParseError as e:
        print(f"error: {e}", file=sys.stderr)
        print("----- raw model output -----", file=sys.stderr)
        print(e.raw_response, file=sys.stderr)
        return 1

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    markdown = render_markdown_report(args.claim, media_info, args.model, evidence, ratings, timestamp)
    json_report = render_json_report(str(dm_path), args.claim, media_info, args.model, evidence, ratings, timestamp)

    print(markdown)

    md_path, json_path = resolve_output_paths(args.output)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(json_report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved report to '{md_path}' and '{json_path}'.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
