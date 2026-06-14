"""
reverse_image_search.py — ReverseImageSearchTool: LLM-scored keyframe selection
followed by SerpApi reverse image search.

Flow:
  1. Score every keyframe (or the single image) via the vision LLM (no-think).
  2. If the highest score < 6, skip RIS entirely.
  3. Otherwise, run SerpApi reverse image search on the top-scoring frame only.
  4. Optionally dump scores + results to a .txt sidecar.

Environment variables:
  SERPAPI_KEY          — required for actual RIS (no default).
  OLLAMA_HOST          — inherited from base (default http://localhost:11434).
  OLLAMA_VISION_MODEL  — inherited from base.
  RIS_SAVE_TXT         — set to "1" to always write the sidecar .txt file.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from data_manager import DataManager
from vision_tools.base import (
    VisionTool,
    _make_tool_json,
    _ollama_post,
    _ollama_response,
    OLLAMA_HOST,
    OLLAMA_VISION_MODEL,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SERPAPI_KEY: str = "c13af0f324b60d48b08d6293d13dcf71c5a11b312e5dd722a185a46c07f6790e"
SERPAPI_BASE: str = "https://serpapi.com/search.json"
_MIN_SCORE_THRESHOLD: int = 6
_RIS_SAVE_TXT: bool = os.environ.get("RIS_SAVE_TXT", "0") == "1"

_SCORE_PROMPT = """\
You are assisting a fact-checking newsroom with reverse image search triage.
Rate this image frame from 0 to 10 for how useful a reverse image search would be.

High scores (7-10): distinctive visual content — recognisable landmarks, faces,
logos, unique events, specific objects, text overlays, or scenes likely to appear
in published media.

Low scores (0-5): generic content — plain backgrounds, blurry frames, very dark
or overexposed frames, talking heads with no visual context, abstract scenes.

Return ONLY a valid JSON object with two keys:
  "score": integer 0-10,
  "reason": one sentence explaining the rating.
No markdown, no preamble."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_frame(image_path: str, model: str, host: str, timeout: int) -> dict:
    """Ask the vision LLM to rate a single frame for RIS relevancy."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    result = _ollama_post(
        host,
        {
            "model": model,
            "prompt": _SCORE_PROMPT,
            "images": [b64],
            "stream": False,
            "think": False,  # explicit no-think
        },
        timeout=timeout,
        think=False,
    )
    raw = _ollama_response(result).strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()

    try:
        parsed = json.loads(raw)
        score = int(parsed.get("score", 0))
        reason = str(parsed.get("reason", ""))
        return {"score": max(0, min(10, score)), "reason": reason}
    except (json.JSONDecodeError, ValueError):
        # Graceful degradation: treat unparseable response as score 0
        return {"score": 0, "reason": f"LLM returned non-JSON: {raw[:120]}"}


def _serpapi_reverse_search(image_path: str, api_key: str, timeout: int) -> dict:
    """
    Upload the image to SerpApi's Google Reverse Image Search endpoint.
    SerpApi accepts a local file via base64 in the `image_base64` parameter,
    or a public URL via `image_url`. We use base64 to avoid needing a
    publicly reachable host.
    """
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    params = {
        "engine": "google_reverse_image",
        "image_base64": b64,
        "api_key": api_key,
        "hl": "en",
        "gl": "us",
    }
    encoded = urllib.parse.urlencode(params)
    url = f"{SERPAPI_BASE}?{encoded}"

    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"SerpApi HTTP {e.code}: {body[:500]}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"SerpApi connection error: {e.reason}")


def _extract_ris_summary(serpapi_response: dict) -> dict:
    """
    Pull the most useful fields from SerpApi's verbose response dict.
    Returns a compact summary suitable for downstream LLM consumption.
    """
    summary: dict = {}

    # Image results (visually similar images)
    image_results = serpapi_response.get("image_results", [])
    if image_results:
        summary["image_results"] = [
            {
                "title": r.get("title", ""),
                "source": r.get("source", ""),
                "link": r.get("link", ""),
                "thumbnail": r.get("thumbnail", ""),
            }
            for r in image_results[:10]  # cap at 10
        ]

    # Knowledge graph (if SerpApi identifies a known entity)
    kg = serpapi_response.get("knowledge_graph")
    if kg:
        summary["knowledge_graph"] = {
            "title": kg.get("title"),
            "type": kg.get("type"),
            "description": kg.get("description"),
            "source": kg.get("source", {}).get("name"),
        }

    # Pages that include matching images
    pages = serpapi_response.get("pages_with_matching_images", [])
    if pages:
        summary["pages_with_matching_images"] = [
            {
                "title": p.get("page_title", ""),
                "link": p.get("link", ""),
                "snippet": p.get("snippet", ""),
            }
            for p in pages[:10]
        ]

    # Inline images (similar)
    inline = serpapi_response.get("inline_images", [])
    if inline:
        summary["inline_images"] = [
            {"source": i.get("source", ""), "link": i.get("link", "")}
            for i in inline[:5]
        ]

    if not summary:
        summary["raw_keys"] = list(serpapi_response.keys())

    return summary


def _write_txt_sidecar(
    path: str,
    scores: list[dict],
    chosen_frame: Optional[str],
    ris_summary: Optional[dict],
    skip_reason: Optional[str],
) -> None:
    lines = ["=== Reverse Image Search Report ===\n"]

    lines.append("--- Keyframe Scores ---")
    for entry in scores:
        lines.append(
            f"  [{entry['score']:>2}/10]  {Path(entry['path']).name}  —  {entry['reason']}"
        )

    lines.append("")
    if skip_reason:
        lines.append(f"RIS skipped: {skip_reason}")
    else:
        lines.append(f"RIS performed on: {chosen_frame}")
        lines.append("")
        lines.append("--- SerpApi Summary ---")
        lines.append(json.dumps(ris_summary, indent=2, ensure_ascii=False))

    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"ReverseImageSearchTool: sidecar written to '{path}'", file=sys.stderr)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class ReverseImageSearchTool(VisionTool):
    TOOL_NAME = "Reverse Image Search"
    INPUTS = ["Image", "Video", "Keyframes"]

    def __init__(
        self,
        api_key: str = SERPAPI_KEY,
        model: str = OLLAMA_VISION_MODEL,
        host: str = OLLAMA_HOST,
        score_timeout: int = 60,
        ris_timeout: int = 30,
        save_txt: bool = _RIS_SAVE_TXT,
        txt_path: Optional[str] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.host = host
        self.score_timeout = score_timeout
        self.ris_timeout = ris_timeout
        self.save_txt = save_txt
        self.txt_path = txt_path  # None → auto-derive next to source media

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_txt_path(self, media_path: str) -> str:
        if self.txt_path:
            return self.txt_path
        return str(Path(media_path).with_suffix(".ris.txt"))

    def _score_all(self, sources: list[str]) -> list[dict]:
        """Return a list of {path, score, reason} dicts, one per source frame."""
        scored = []
        total = len(sources)
        for i, img_path in enumerate(sources, 1):
            print(
                f"ReverseImageSearchTool: scoring frame {i}/{total} "
                f"({Path(img_path).name}) ...",
                file=sys.stderr,
            )
            try:
                result = _score_frame(img_path, self.model, self.host, self.score_timeout)
                scored.append({"path": img_path, **result})
            except Exception as e:
                print(
                    f"ReverseImageSearchTool: scoring error on {img_path}: {e}",
                    file=sys.stderr,
                )
                scored.append({"path": img_path, "score": 0, "reason": f"Scoring error: {e}"})
        return scored

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------

    def run(self, data: DataManager) -> Optional[dict]:
        sources: list[str] = data.keyframes if data.isVideo else [data.originalMedia]

        if not sources:
            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS, None,
                explanation="No images available for reverse image search.",
                has_run=0,
            )

        # --- Step 1: score all frames ---
        scored = self._score_all(sources)
        best = max(scored, key=lambda x: x["score"])

        print(
            f"ReverseImageSearchTool: best frame score = {best['score']}/10 "
            f"({Path(best['path']).name})",
            file=sys.stderr,
        )

        # --- Step 2: gate on threshold ---
        if best["score"] < _MIN_SCORE_THRESHOLD:
            skip_reason = (
                f"Highest keyframe score ({best['score']}/10) is below the threshold "
                f"of {_MIN_SCORE_THRESHOLD}. No reverse image search performed."
            )
            print(f"ReverseImageSearchTool: {skip_reason}", file=sys.stderr)

            if self.save_txt:
                _write_txt_sidecar(
                    self._resolve_txt_path(data.originalMedia),
                    scored,
                    chosen_frame=None,
                    ris_summary=None,
                    skip_reason=skip_reason,
                )

            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS,
                output={
                    "performed": False,
                    "scores": scored,
                    "skip_reason": skip_reason,
                },
                explanation=skip_reason,
                confidence=-1,
                confidence_explanation="No search performed; threshold not met.",
                corroborating_tools=["Geolocation", "Description", "Keyframes"],
            )

        # --- Step 3: API key check ---
        if not self.api_key:
            msg = (
                "SERPAPI_KEY is not set. Set the environment variable to enable "
                "reverse image search."
            )
            print(f"ReverseImageSearchTool: {msg}", file=sys.stderr)
            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS, None, explanation=msg, has_run=0,
            )

        # --- Step 4: RIS on best frame ---
        chosen_frame = best["path"]
        print(
            f"ReverseImageSearchTool: running SerpApi RIS on '{chosen_frame}' ...",
            file=sys.stderr,
        )

        try:
            raw_response = _serpapi_reverse_search(
                chosen_frame, self.api_key, self.ris_timeout
            )
            ris_summary = _extract_ris_summary(raw_response)
        except Exception as e:
            print(f"ReverseImageSearchTool: SerpApi error: {e}", file=sys.stderr)
            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS, None, explanation=str(e), has_run=0,
            )

        # --- Step 5: optional txt sidecar ---
        if self.save_txt:
            _write_txt_sidecar(
                self._resolve_txt_path(data.originalMedia),
                scored,
                chosen_frame=chosen_frame,
                ris_summary=ris_summary,
                skip_reason=None,
            )

        # --- Confidence heuristic ---
        n_pages = len(ris_summary.get("pages_with_matching_images", []))
        n_images = len(ris_summary.get("image_results", []))
        has_kg = "knowledge_graph" in ris_summary

        if has_kg:
            confidence = 80
            conf_note = "A knowledge graph entity was identified, indicating a well-known subject."
        elif n_pages >= 5:
            confidence = 65
            conf_note = f"{n_pages} pages with matching images found."
        elif n_pages >= 1 or n_images >= 3:
            confidence = 50
            conf_note = f"{n_pages} matching page(s) and {n_images} similar image(s) found."
        else:
            confidence = 30
            conf_note = "Few or no matching results returned by SerpApi."

        output = {
            "performed": True,
            "chosen_frame": chosen_frame,
            "chosen_frame_score": best["score"],
            "chosen_frame_score_reason": best["reason"],
            "all_scores": scored,
            "ris_summary": ris_summary,
        }

        return _make_tool_json(
            self.TOOL_NAME, self.INPUTS,
            output=output,
            explanation=(
                f"Reverse image search performed on the highest-scoring keyframe "
                f"(score {best['score']}/10: {best['reason']}). "
                f"SerpApi returned {n_images} similar image(s) and "
                f"{n_pages} page(s) with matching images."
                + (" Knowledge graph entity identified." if has_kg else "")
            ),
            confidence=confidence,
            confidence_explanation=conf_note,
            corroborating_tools=["Geolocation", "Description", "NER", "Metadata Gatherer"],
        )