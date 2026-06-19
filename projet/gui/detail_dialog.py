"""
detail_dialog.py — popup shown when a graph node is clicked: explanation,
confidence + confidence explanation, corroborating tools, and the full
output (a thumbnail gallery for Keyframes, formatted JSON/text otherwise).
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QSize
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPlainTextEdit, QListWidget, QListWidgetItem,
)

from graph_model import GraphNode


def _format_output(output) -> str:
    if output is None:
        return "(no output)"
    if isinstance(output, (dict, list)):
        try:
            return json.dumps(output, indent=2, ensure_ascii=False)
        except TypeError:
            return str(output)
    return str(output)


def _readonly_box(text: str, max_height: int | None = None) -> QPlainTextEdit:
    box = QPlainTextEdit(text)
    box.setReadOnly(True)
    if max_height:
        box.setMaximumHeight(max_height)
    return box


class DetailDialog(QDialog):
    def __init__(self, node: GraphNode, parent=None):
        super().__init__(parent)
        self.setWindowTitle(node.label)
        self.resize(680, 560)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"<h2>{node.label}</h2>"))

        if node.kind == "root":
            path = (node.tool_json or {}).get("path", "")
            layout.addWidget(QLabel(f"Source file: {path}"))
            return

        tj = node.tool_json or {}

        if not tj.get("hasRun", 1):
            layout.addWidget(QLabel("<i>This tool did not run, or is not implemented.</i>"))

        layout.addWidget(QLabel("<b>Explanation</b>"))
        layout.addWidget(_readonly_box(tj.get("Explanation") or "(none)", 90))

        conf = tj.get("Confidence", 0)
        conf_text = "N/A" if conf is None or conf < 0 else f"{conf}/100"
        layout.addWidget(QLabel(f"<b>Confidence:</b> {conf_text}"))

        layout.addWidget(QLabel("<b>Confidence explanation</b>"))
        layout.addWidget(_readonly_box(tj.get("ConfidenceExplanation") or "(none)", 90))

        corroborating = tj.get("CorroboratingTools") or []
        if corroborating:
            layout.addWidget(QLabel(f"<b>Corroborating tools:</b> {', '.join(corroborating)}"))

        layout.addWidget(QLabel("<b>Output</b>"))
        output = tj.get("Output")

        if node.label == "Keyframes" and isinstance(output, list) and output:
            layout.addWidget(self._keyframe_gallery(output))
        else:
            layout.addWidget(_readonly_box(_format_output(output)))

    @staticmethod
    def _keyframe_gallery(paths: list[str]) -> QListWidget:
        gallery = QListWidget()
        gallery.setViewMode(QListWidget.IconMode)
        gallery.setIconSize(QSize(140, 100))
        gallery.setResizeMode(QListWidget.Adjust)
        gallery.setMinimumHeight(240)
        for p in paths:
            pix = QPixmap(p)
            item = QListWidgetItem(QIcon(pix), Path(p).name)
            gallery.addItem(item)
        return gallery