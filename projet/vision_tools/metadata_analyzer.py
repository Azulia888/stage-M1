"""
metadata_analyzer.py — MetadataAnalyzerTool: LLM-based synthesis of raw metadata.
"""

from __future__ import annotations

import sys
from typing import Optional

from data_manager import DataManager
from vision_tools.base import (
    VisionTool, _make_tool_json, _ollama_post, _ollama_response,
    OLLAMA_HOST, OLLAMA_SYNTH_MODEL,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN: int = 4
_MAX_TOKENS: int = 4096
_PROMPT_OVERHEAD_CHARS: int = 512 * _CHARS_PER_TOKEN   # 2048
_CHUNK_MAX_CHARS: int = (_MAX_TOKENS * _CHARS_PER_TOKEN) - _PROMPT_OVERHEAD_CHARS  # 14336


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SINGLE_PROMPT = """\
You are a media forensics analyst working for a fact-checking newsroom.
Below is the raw technical and contextual metadata extracted from a media file.
Produce a concise, structured plain-text summary of the most relevant facts.
Focus on: provenance (origin, uploader, platform), timestamps and dates, \
technical properties (codec, resolution, duration), and any fields that could \
help assess authenticity or context.
Flag any anomalies (e.g. creation date after upload date, mismatched codecs, \
missing expected fields).
Output only the summary — no preamble, no markdown.

RAW METADATA:
{metadata}"""

_CHUNK_PROMPT = """\
You are a media forensics analyst working for a fact-checking newsroom.
Below is a PARTIAL excerpt of raw metadata extracted from a media file \
(chunk {chunk_index} of {total_chunks}).
Summarise the key facts present in this excerpt only.
Be concise. Do not invent information not present in the excerpt.
Output only the summary — no preamble, no markdown.

METADATA EXCERPT:
{metadata}"""

_SYNTHESIS_PROMPT = """\
You are a media forensics analyst working for a fact-checking newsroom.
Below are partial summaries produced from sequential excerpts of the raw \
metadata of a single media file.
Merge them into one coherent, non-redundant summary.
Focus on: provenance, timestamps, technical properties, and authenticity signals.
Flag any anomalies or contradictions across the partial summaries.
Output only the final summary — no preamble, no markdown.

PARTIAL SUMMARIES:
{summaries}"""


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class MetadataAnalyzerTool(VisionTool):
    TOOL_NAME = "Metadata Analyzer"
    INPUTS = ["Metadata"]

    def __init__(
        self,
        model: str = OLLAMA_SYNTH_MODEL,
        host: str = OLLAMA_HOST,
        timeout: int = 120,
    ):
        self.model = model
        self.host = host
        self.timeout = timeout

    @staticmethod
    def _split_into_chunks(text: str, max_chars: int) -> list[str]:
        lines = text.splitlines(keepends=True)
        chunks: list[str] = []
        current: list[str] = []
        current_len: int = 0

        for line in lines:
            if len(line) > max_chars:
                if current:
                    chunks.append("".join(current))
                    current, current_len = [], 0
                for i in range(0, len(line), max_chars):
                    chunks.append(line[i: i + max_chars])
                continue
            if current_len + len(line) > max_chars:
                chunks.append("".join(current))
                current, current_len = [], 0
            current.append(line)
            current_len += len(line)

        if current:
            chunks.append("".join(current))
        return chunks

    def _call_ollama(self, prompt: str) -> str:
        result = _ollama_post(
            self.host,
            {"model": self.model, "prompt": prompt, "stream": False},
            timeout=self.timeout,
        )
        return _ollama_response(result).strip()

    def _summarise_chunk(self, chunk: str, index: int, total: int) -> str:
        return self._call_ollama(
            _CHUNK_PROMPT.format(chunk_index=index, total_chunks=total, metadata=chunk)
        )

    def _synthesise(self, partial_summaries: list[str]) -> str:
        joined = "\n\n---\n\n".join(
            f"[Chunk {i + 1}]\n{s}" for i, s in enumerate(partial_summaries)
        )
        return self._call_ollama(_SYNTHESIS_PROMPT.format(summaries=joined))

    def run(self, data: DataManager) -> Optional[dict]:
        raw = data.raw_metadata
        if not raw or not raw.strip():
            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS, None,
                explanation=(
                    "No raw metadata available. "
                    "MetadataTool must run before MetadataAnalyzerTool."
                ),
                has_run=0,
            )

        try:
            if len(raw) <= _CHUNK_MAX_CHARS:
                print(
                    f"MetadataAnalyzerTool: single-call path ({len(raw)} chars).",
                    file=sys.stderr,
                )
                summary = self._call_ollama(_SINGLE_PROMPT.format(metadata=raw))
                partial_summaries = None
                n_chunks = 1
            else:
                chunks = self._split_into_chunks(raw, _CHUNK_MAX_CHARS)
                n_chunks = len(chunks)
                print(
                    f"MetadataAnalyzerTool: multi-chunk path — "
                    f"{len(raw)} chars split into {n_chunks} chunk(s).",
                    file=sys.stderr,
                )
                partial_summaries = []
                for i, chunk in enumerate(chunks, start=1):
                    print(
                        f"  MetadataAnalyzerTool: summarising chunk "
                        f"{i}/{n_chunks} ({len(chunk)} chars) ...",
                        file=sys.stderr,
                    )
                    partial_summaries.append(self._summarise_chunk(chunk, i, n_chunks))
                print("MetadataAnalyzerTool: synthesising partial summaries ...", file=sys.stderr)
                summary = self._synthesise(partial_summaries)

            output = {
                "summary": summary,
                "chunks_used": n_chunks,
                "partial_summaries": partial_summaries,
            }

            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS,
                output=output,
                explanation=(
                    f"Metadata synthesised from {n_chunks} chunk(s). "
                    + (
                        "Single-call path (metadata fits within context window)."
                        if n_chunks == 1
                        else f"Multi-chunk path: {n_chunks} partial summaries merged."
                    )
                ),
                confidence=75,
                confidence_explanation=(
                    "LLM-generated summary of structured metadata. "
                    "Factual accuracy is high for fields present verbatim in the raw data. "
                    "The model may misinterpret ambiguous field names or codec strings. "
                    "Always cross-check against data.raw_metadata for precision."
                ),
                corroborating_tools=["Metadata Gatherer", "Description", "NER"],
            )

        except Exception as e:
            print(f"MetadataAnalyzerTool error: {e}", file=sys.stderr)
            return _make_tool_json(
                self.TOOL_NAME, self.INPUTS, None, explanation=str(e), has_run=0
            )