"""
base.py — Shared helpers, constants, and VisionTool base class.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import urllib.error
import urllib.request

from data_manager import DataManager


# ---------------------------------------------------------------------------
# Environment-driven config
# ---------------------------------------------------------------------------

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "qwen3.5:2b")
OLLAMA_SYNTH_MODEL = os.environ.get("OLLAMA_SYNTH_MODEL", OLLAMA_VISION_MODEL)
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")


# ---------------------------------------------------------------------------
# Tool JSON factory
# ---------------------------------------------------------------------------

def _make_tool_json(
    tool_name: str,
    inputs: list[str],
    output: object,
    explanation: str = "",
    confidence: int = 0,
    confidence_explanation: str = "",
    corroborating_tools: list = None,
    has_run: int = 1,
) -> dict:
    return {
        "ToolName": tool_name,
        "Input": inputs,
        "hasRun": has_run,
        "Output": output,
        "Explanation": explanation,
        "Confidence": confidence,
        "ConfidenceExplanation": confidence_explanation,
        "CorroboratingTools": corroborating_tools or [],
    }


# ---------------------------------------------------------------------------
# Ollama HTTP helpers
# ---------------------------------------------------------------------------

def _ollama_post(host: str, payload: dict, timeout: int = 120, think: bool = False) -> dict:
    if not think:
        payload = {**payload, "think": False}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    result_container = [None]
    error_container = [None]

    def do_request():
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result_container[0] = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            error_container[0] = RuntimeError(f"Ollama HTTP {e.code}: {body}")
        except urllib.error.URLError as e:
            error_container[0] = RuntimeError(f"Ollama connection error: {e.reason}")
        except Exception as e:
            error_container[0] = e

    thread = threading.Thread(target=do_request, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise RuntimeError(
            f"Ollama request timed out after {timeout}s "
            f"(model={payload.get('model')}, prompt length={len(payload.get('prompt', ''))})"
        )
    if error_container[0] is not None:
        raise error_container[0]
    return result_container[0]


def _ollama_response(result: dict) -> str:
    text = result.get("response", "").strip()
    if text:
        return text
    thinking = result.get("thinking", "").strip()
    if thinking:
        print("WARNING: 'response' empty, using 'thinking' field.", file=sys.stderr)
        return thinking
    return ""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class VisionTool:
    TOOL_NAME: str = "BaseTool"
    INPUTS: list[str] = []

    def run(self, data: DataManager) -> dict | None:
        """Execute the tool and return a tool-result dict, or None on skip."""
        return None

    def addData(self, data: DataManager) -> None:
        result = self.run(data)
        if result is not None:
            data.addToolResult(result)


# ---------------------------------------------------------------------------
# Stub base
# ---------------------------------------------------------------------------

class _StubTool(VisionTool):
    """Base for tools that are not yet implemented."""

    def run(self, data: DataManager) -> dict | None:
        return _make_tool_json(
            self.TOOL_NAME,
            self.INPUTS,
            output=None,
            explanation="Not implemented.",
            has_run=0,
        )