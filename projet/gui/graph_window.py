"""
graph_window.py — shows the result graph: a root node for the source media,
one node per tool, and arrows showing which data fed which tool. Clicking a
node opens a detail dialog (explanation, confidence, full output).
"""

from __future__ import annotations

import math
import textwrap
from collections import defaultdict

from PySide6.QtCore import Qt, QPointF, QSize
from PySide6.QtGui import QBrush, QPen, QColor, QPixmap, QPolygonF, QPainter
from PySide6.QtWidgets import (
    QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem,
    QGraphicsTextItem, QGraphicsPixmapItem, QGraphicsLineItem,
    QGraphicsPolygonItem, QGraphicsItem, QFileDialog
)

from graph_model import build_graph, node_color, GraphNode
from detail_dialog import DetailDialog


def _wrap_label(label: str) -> str:
    return "\n".join(textwrap.wrap(label, width=12)) or label


class ZoomableGraphicsView(QGraphicsView):
    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)


class NodeItem(QGraphicsEllipseItem):
    RADIUS = 42

    def __init__(self, node: GraphNode, on_click):
        r = self.RADIUS
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.node = node
        self._on_click = on_click

        self.setBrush(QBrush(QColor(node_color(node))))
        self.setPen(QPen(QColor("#37474f"), 2))
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setZValue(1)

        label = QGraphicsTextItem(_wrap_label(node.label), self)
        label.setDefaultTextColor(QColor("#102027"))
        lb_rect = label.boundingRect()
        label.setPos(-lb_rect.width() / 2, -lb_rect.height() / 2)

        if node.label == "Keyframes":
            self._add_thumbnail()

    def _add_thumbnail(self) -> None:
        tj = self.node.tool_json or {}
        output = tj.get("Output")
        if not output:
            return
        pix = QPixmap(output[0])
        if pix.isNull():
            return
        pix = pix.scaled(56, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        thumb = QGraphicsPixmapItem(pix, self)
        thumb.setPos(-pix.width() / 2, self.RADIUS + 6)

    def mousePressEvent(self, event) -> None:
        self._on_click(self.node)
        super().mousePressEvent(event)


def _add_edge(scene: QGraphicsScene, src_item: NodeItem, tgt_item: NodeItem) -> None:
    src_c = src_item.scenePos()
    tgt_c = tgt_item.scenePos()
    dx, dy = tgt_c.x() - src_c.x(), tgt_c.y() - src_c.y()
    dist = math.hypot(dx, dy) or 1.0
    ux, uy = dx / dist, dy / dist

    start = QPointF(src_c.x() + ux * src_item.RADIUS, src_c.y() + uy * src_item.RADIUS)
    end = QPointF(tgt_c.x() - ux * tgt_item.RADIUS, tgt_c.y() - uy * tgt_item.RADIUS)

    pen = QPen(QColor("#90a4ae"), 2)
    line = QGraphicsLineItem(start.x(), start.y(), end.x(), end.y())
    line.setPen(pen)
    line.setZValue(0)
    scene.addItem(line)

    angle = math.atan2(end.y() - start.y(), end.x() - start.x())
    size = 10
    p2 = QPointF(end.x() - size * math.cos(angle - math.pi / 6),
                 end.y() - size * math.sin(angle - math.pi / 6))
    p3 = QPointF(end.x() - size * math.cos(angle + math.pi / 6),
                 end.y() - size * math.sin(angle + math.pi / 6))
    arrow = QGraphicsPolygonItem(QPolygonF([end, p2, p3]))
    arrow.setBrush(QBrush(QColor("#90a4ae")))
    arrow.setPen(QPen(Qt.NoPen))
    arrow.setZValue(0)
    scene.addItem(arrow)


def _layout(nodes: list[GraphNode]) -> dict[str, tuple[float, float]]:
    by_depth: dict[int, list[GraphNode]] = defaultdict(list)
    for n in nodes:
        by_depth[n.depth].append(n)

    ring_gap = 230
    positions: dict[str, tuple[float, float]] = {}
    for depth, group in by_depth.items():
        if depth == 0:
            positions[group[0].node_id] = (0.0, 0.0)
            continue
        radius = depth * ring_gap
        count = len(group)
        for i, n in enumerate(group):
            angle = 2 * math.pi * i / count
            positions[n.node_id] = (radius * math.cos(angle), radius * math.sin(angle))
    return positions


class GraphWindow(QMainWindow):
    def __init__(self, data):
        super().__init__()
        self.setWindowTitle("AFC — Result graph")
        self.resize(1100, 800)
        self.data = data

        self.view = ZoomableGraphicsView()
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        self.setCentralWidget(self.view)

        self._build_scene()
        self.statusBar().showMessage(
            "Scroll to zoom, drag to pan. Click a node for details."
        )

        export_action = self.menuBar().addAction("Export analysis...")
        export_action.triggered.connect(self._export)

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export analysis", filter="Pickle (*.pkl)")
        if path:
            self.data.save(path)

    def _build_scene(self) -> None:
        nodes, edges = build_graph(self.data)
        positions = _layout(nodes)
        items: dict[str, NodeItem] = {}

        for n in nodes:
            item = NodeItem(n, self._open_detail)
            x, y = positions[n.node_id]
            item.setPos(x, y)
            self.scene.addItem(item)
            items[n.node_id] = item

        for e in edges:
            _add_edge(self.scene, items[e.source], items[e.target])

        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-60, -60, 60, 60))

    def _open_detail(self, node: GraphNode) -> None:
        dlg = DetailDialog(node, self)
        dlg.exec()