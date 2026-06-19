"""
graph_model.py — turns a finished DataManager.toolResult dict into a small
node/edge graph for visualisation: one node per tool output, one root node
for the source media, and edges showing which data fed which tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

ROOT_ID = "root"

# Tools whose only dependency is the raw source media.
_ROOT_ONLY = {
    "Metadata Gatherer", "Keyframes", "Transcript", "Lipsync Detection",
    "AI Detection", "Deepfake Detection",
}

# Tools that consume Keyframes when the source is a video, and the raw
# media directly otherwise.
_KEYFRAME_OR_ROOT = {
    "OCR", "Weather Detection", "Geolocation", "Facial Recognition",
    "Reverse Image Search",
}


@dataclass
class GraphNode:
    node_id: str
    label: str
    kind: str  # "root" or "tool"
    tool_json: Optional[dict]
    depth: int = 0


@dataclass
class GraphEdge:
    source: str
    target: str


def _dependencies_for(tool_name: str, is_video: bool, tool_result: dict) -> list[str]:
    if tool_name in _ROOT_ONLY:
        return [ROOT_ID]

    if tool_name in _KEYFRAME_OR_ROOT:
        return ["Keyframes"] if is_video and "Keyframes" in tool_result else [ROOT_ID]

    if tool_name == "Metadata Analyzer":
        return ["Metadata Gatherer"] if "Metadata Gatherer" in tool_result else [ROOT_ID]

    if tool_name == "Description":
        deps = [ROOT_ID]
        if is_video and "Transcript" in tool_result:
            deps.append("Transcript")
        return deps

    if tool_name == "NER":
        deps = [d for d in ("Transcript", "Description", "OCR", "Metadata Analyzer")
                if d in tool_result]
        return deps or [ROOT_ID]

    if tool_name == "Knowledge Graph":
        return ["NER"] if "NER" in tool_result else [ROOT_ID]

    return [ROOT_ID]


def build_graph(data) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Build the node/edge lists for a finished DataManager."""
    tool_result: dict = data.toolResult
    root_label = "Video" if data.isVideo else "Image"

    nodes: dict[str, GraphNode] = {
        ROOT_ID: GraphNode(ROOT_ID, root_label, "root",
                            {"path": data.originalMedia}, depth=0)
    }
    edges: list[GraphEdge] = []

    # toolResult preserves insertion order, which already respects the
    # pipeline's dependency order, so each tool's dependencies are always
    # already present in `nodes` by the time we reach it.
    for tool_name, tool_json in tool_result.items():
        deps = _dependencies_for(tool_name, data.isVideo, tool_result)
        known_deps = [d for d in deps if d in nodes]
        depth = (max(nodes[d].depth for d in known_deps) + 1) if known_deps else 1
        nodes[tool_name] = GraphNode(tool_name, tool_name, "tool", tool_json, depth=depth)
        for d in known_deps:
            edges.append(GraphEdge(d, tool_name))

    return list(nodes.values()), edges


def node_color(node: GraphNode) -> str:
    if node.kind == "root":
        return "#cfd8dc"

    tj = node.tool_json or {}
    if not tj.get("hasRun", 1):
        return "#9e9e9e"

    conf = tj.get("Confidence", 0)
    if conf is None or conf < 0:
        return "#64b5f6"
    if conf >= 60:
        return "#66bb6a"
    if conf >= 30:
        return "#ffa726"
    return "#ef5350"