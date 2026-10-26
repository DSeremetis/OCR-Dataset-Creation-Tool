#!/usr/bin/env python3
"""PyQt5 reviewer for automatic or manual handwritten text-line regions.

The tool pairs files by their shared sequential dataset order, for example:

    images/handwritten/00005_..._handwritten.png
    labels/clean_from_doc/00005_..._ground_truth.txt

Draw a rectangle around each text line, select the corresponding transcript
line, and save. An annotation JSON and an overlay PNG are written for every
page. The JSON is deliberately model-independent so it can feed either
Tesseract training or a later Keras pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from statistics import median
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.line_ocr.kraken_to_proposals import regions_from_kraken
from scripts.line_ocr.project_core import ensure_project, import_page, read_manifest

from PyQt5.QtCore import QProcess, QPoint, QPointF, QRect, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QKeySequence, QPainter, QPen, QPixmap, QTransform
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QShortcut, QDoubleSpinBox,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


ID_SUFFIXES = ("_handwritten", "_ground_truth")


@dataclass
class Region:
    x: float
    y: float
    width: float
    height: float
    text_index: int | None = None

    def rect(self) -> QRectF:
        return QRectF(self.x, self.y, self.width, self.height)

    def to_dict(self, text: str | None = None) -> dict:
        result = {
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "width": round(self.width, 2),
            "height": round(self.height, 2),
            "text_index": self.text_index,
        }
        if text is not None:
            result["text"] = text
        return result

    @classmethod
    def from_dict(cls, value: dict) -> "Region":
        return cls(
            float(value["x"]), float(value["y"]),
            float(value["width"]), float(value["height"]),
            value.get("text_index"),
        )


def find_pairs(images_dir: Path, labels_dir: Path, typed_dir: Path) -> list[tuple[str, Path, Path, Path | None]]:
    """Pair the already-renamed folders by sorted sequence, never old stems."""
    images = sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    labels = sorted(labels_dir.glob("*.txt"))
    typed = sorted(typed_dir.glob("*.png"))
    if len(images) != len(labels):
        raise ValueError(f"Found {len(images)} handwritten images but {len(labels)} transcripts.")
    if typed and len(typed) != len(images):
        raise ValueError(f"Found {len(images)} handwritten images but {len(typed)} typed images.")
    result = []
    for index, (image_path, label_path) in enumerate(zip(images, labels)):
        # Queue files retain their local IDs even if a rejected source page is
        # removed. Pairing remains positional, while proposal/ready records
        # continue to address the same surviving queue item.
        item_id = image_path.stem
        typed_path = typed[index] if typed else None
        result.append((item_id, image_path, label_path, typed_path))
    return result


def priority_order(path: Path | None) -> dict[str, int]:
    """Return active-learning ranks; absent/malformed files leave UI order unchanged."""
    if path is None or not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return {
                str(row["ui_page_id"]): int(row["priority_rank"])
                for row in csv.DictReader(handle)
                if row.get("ui_page_id") and row.get("priority_rank")
            }
    except (OSError, ValueError, KeyError):
        return {}


def transcript_lines(path: Path) -> list[str]:
    # Blank source lines are paragraph spacing rather than handwriting lines.
    content = path.read_text(encoding="utf-8-sig")
    return [line.strip() for line in content.splitlines() if line.strip()]


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path.resolve())


def detect_line_regions(image: QImage) -> list[Region]:
    """Propose horizontal text-line boxes using a lightweight ink projection.

    This is deliberately a proposal, not a final annotation. It ignores the
    outer margin when measuring ink, smooths the horizontal projection, and
    returns full-width regions that can be corrected in the UI.
    """
    if image.isNull():
        return []
    gray = image.convertToFormat(QImage.Format_Grayscale8)
    bits = gray.bits()
    bits.setsize(gray.byteCount())
    pixels = bytes(bits)
    stride = gray.bytesPerLine()
    width, height = gray.width(), gray.height()
    x_start = int(width * 0.07)
    x_end = int(width * 0.98)
    x_step = max(2, width // 900)
    y_step = max(1, height // 1400)

    scores: list[int] = []
    for y in range(0, height, y_step):
        row = y * stride
        dark = 0
        for x in range(x_start, x_end, x_step):
            if pixels[row + x] < 175:
                dark += 1
        scores.append(dark)
    if not scores or max(scores) == 0:
        return []

    smooth: list[float] = []
    window = 2
    for index in range(len(scores)):
        start, end = max(0, index - window), min(len(scores), index + window + 1)
        smooth.append(sum(scores[start:end]) / (end - start))
    baseline = median(smooth)
    sorted_scores = sorted(smooth)
    p90 = sorted_scores[min(len(sorted_scores) - 1, int(len(sorted_scores) * 0.90))]
    threshold = max(3.0, baseline + (p90 - baseline) * 0.20, p90 * 0.12)

    active = [value >= threshold for value in smooth]
    runs: list[tuple[int, int]] = []
    start: int | None = None
    gap_limit = max(3, round(18 / y_step))
    gap = 0
    for index, is_active in enumerate(active + [False]):
        if is_active:
            if start is None:
                start = index
            gap = 0
        elif start is not None:
            gap += 1
            if gap > gap_limit or index == len(active):
                end = index - gap + 1
                if end - start >= max(2, round(8 / y_step)):
                    runs.append((start, end))
                start = None
                gap = 0

    padding = max(8, round(height * 0.004))
    regions = []
    for start, end in runs:
        top = max(0, start * y_step - padding)
        bottom = min(height, end * y_step + padding)
        if bottom - top >= 12:
            regions.append(Region(x_start, top, x_end - x_start, bottom - top))
    return regions


class ImageCanvas(QWidget):
    """Scrollable, zoomable image canvas with draggable line rectangles."""

    region_selected = pyqtSignal(int)
    regions_changed = pyqtSignal()
    image_clicked = pyqtSignal(float, float)

    HANDLE = 12

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.image = QImage()
        self.regions: list[Region] = []
        self.selected = -1
        self.zoom = 0.28
        self.fit_zoom = self.zoom
        self.offset = QPointF(20, 20)
        self.mode = "select"
        self.pending_text_index: int | None = None
        self.drag_start = QPointF()
        self.drag_origin: Region | None = None
        self.drag_kind = ""
        self.setMouseTracking(True)
        self.setMinimumSize(500, 500)
        self.setFocusPolicy(Qt.StrongFocus)

    def set_image(self, image: QImage) -> None:
        self.image = image
        self.regions = []
        self.selected = -1
        self.fit_to_view()
        self.update()

    def set_regions(self, regions: list[Region]) -> None:
        self.regions = regions
        self.selected = -1
        self.update()

    def fit_to_view(self, available_width: int | None = None, available_height: int | None = None) -> None:
        if self.image.isNull() or self.width() < 50 or self.height() < 50:
            return
        padding = 30
        available_width = available_width or self.width()
        available_height = available_height or self.height()
        self.fit_zoom = min(
            (available_width - padding) / self.image.width(),
            (available_height - padding) / self.image.height(),
        )
        self.zoom = max(0.03, min(self.fit_zoom, 3.0))
        self.setMinimumSize(
            max(500, int(self.image.width() * self.zoom + padding)),
            max(500, int(self.image.height() * self.zoom + padding)),
        )
        self._center_image()

    def _center_image(self) -> None:
        drawn_w = self.image.width() * self.zoom
        drawn_h = self.image.height() * self.zoom
        self.offset = QPointF(max(10, (self.width() - drawn_w) / 2), max(10, (self.height() - drawn_h) / 2))

    def resizeEvent(self, event) -> None:
        if self.image.isNull():
            return
        # Preserve the current zoom while keeping a fitted image centered.
        self._center_image()
        super().resizeEvent(event)

    def set_zoom(self, value: int) -> None:
        self.zoom = max(0.03, value / 100.0)
        if not self.image.isNull():
            self.setMinimumSize(
                max(500, int(self.image.width() * self.zoom + 30)),
                max(500, int(self.image.height() * self.zoom + 30)),
            )
        self._center_image()
        self.update()

    def image_point(self, point: QPointF) -> QPointF:
        return QPointF((point.x() - self.offset.x()) / self.zoom, (point.y() - self.offset.y()) / self.zoom)

    def canvas_point(self, point: QPointF) -> QPointF:
        return QPointF(self.offset.x() + point.x() * self.zoom, self.offset.y() + point.y() * self.zoom)

    def _clamp_region(self, region: Region) -> None:
        width, height = self.image.width(), self.image.height()
        region.x = max(0, min(region.x, width - 1))
        region.y = max(0, min(region.y, height - 1))
        region.width = max(2, min(region.width, width - region.x))
        region.height = max(2, min(region.height, height - region.y))

    def _hit_test(self, point: QPointF) -> tuple[int, str] | None:
        for index in range(len(self.regions) - 1, -1, -1):
            rect = self.regions[index].rect()
            handle = self.HANDLE / self.zoom
            corners = {
                "tl": QPointF(rect.left(), rect.top()),
                "tr": QPointF(rect.right(), rect.top()),
                "bl": QPointF(rect.left(), rect.bottom()),
                "br": QPointF(rect.right(), rect.bottom()),
            }
            for name, corner in corners.items():
                if abs(point.x() - corner.x()) <= handle and abs(point.y() - corner.y()) <= handle:
                    return index, name
            if rect.contains(point):
                return index, "move"
        return None

    def mousePressEvent(self, event) -> None:
        if self.image.isNull() or event.button() != Qt.LeftButton:
            return
        point = self.image_point(event.localPos())
        if self.mode == "draw":
            self.drag_start = point
            self.regions.append(Region(point.x(), point.y(), 1, 1))
            self.selected = len(self.regions) - 1
            self.drag_kind = "draw"
            self.region_selected.emit(self.selected)
            self.update()
            return
        hit = self._hit_test(point)
        if hit is None:
            self.selected = -1
            self.update()
            return
        self.selected, self.drag_kind = hit
        self.drag_start = point
        original = self.regions[self.selected]
        self.drag_origin = Region(original.x, original.y, original.width, original.height, original.text_index)
        self.region_selected.emit(self.selected)
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self.selected < 0 or not self.drag_kind or not (event.buttons() & Qt.LeftButton):
            return
        current = self.image_point(event.localPos())
        region = self.regions[self.selected]
        if self.drag_kind == "draw":
            region.x = min(self.drag_start.x(), current.x())
            region.y = min(self.drag_start.y(), current.y())
            region.width = abs(current.x() - self.drag_start.x())
            region.height = abs(current.y() - self.drag_start.y())
        else:
            original = self.drag_origin
            dx, dy = current.x() - self.drag_start.x(), current.y() - self.drag_start.y()
            if self.drag_kind == "move":
                region.x, region.y = original.x + dx, original.y + dy
            else:
                if "l" in self.drag_kind:
                    region.x, region.width = original.x + dx, original.width - dx
                if "r" in self.drag_kind:
                    region.width = original.width + dx
                if "t" in self.drag_kind:
                    region.y, region.height = original.y + dy, original.height - dy
                if "b" in self.drag_kind:
                    region.height = original.height + dy
        self._clamp_region(region)
        self.regions_changed.emit()
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self.drag_kind == "draw" and self.selected >= 0:
            region = self.regions[self.selected]
            if region.width < 10 or region.height < 10:
                self.regions.pop(self.selected)
                self.selected = -1
            elif self.pending_text_index is not None:
                region.text_index = self.pending_text_index
            self.mode = "select"
            self.pending_text_index = None
            self.regions_changed.emit()
            self.update()
        self.drag_kind = ""
        self.drag_origin = None

    def keyPressEvent(self, event) -> None:
        if self.selected < 0 or self.selected >= len(self.regions):
            return
        region = self.regions[self.selected]
        step = 5 if event.modifiers() & Qt.ShiftModifier else 1
        if event.key() == Qt.Key_Delete:
            self.regions.pop(self.selected)
            self.selected = min(self.selected, len(self.regions) - 1)
        elif event.key() == Qt.Key_Up:
            region.y -= step
        elif event.key() == Qt.Key_Down:
            region.y += step
        elif event.key() == Qt.Key_Left:
            region.x -= step
        elif event.key() == Qt.Key_Right:
            region.x += step
        else:
            return
        self._clamp_region(region) if self.selected >= 0 else None
        self.regions_changed.emit()
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#252525"))
        if self.image.isNull():
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "No image loaded")
            return
        target = QRectF(self.offset.x(), self.offset.y(), self.image.width() * self.zoom, self.image.height() * self.zoom)
        painter.drawImage(target, self.image)
        for index, region in enumerate(self.regions):
            rect = region.rect()
            top_left = self.canvas_point(QPointF(rect.left(), rect.top()))
            bottom_right = self.canvas_point(QPointF(rect.right(), rect.bottom()))
            canvas_rect = QRectF(top_left, bottom_right)
            selected = index == self.selected
            pen = QPen(QColor("#00e676") if selected else QColor("#ffcc00"), 3 if selected else 2)
            painter.setPen(pen)
            painter.setBrush(QColor(0, 230, 118, 35) if selected else QColor(255, 204, 0, 25))
            painter.drawRect(canvas_rect)
            painter.setPen(Qt.white)
            painter.setFont(QFont("Sans", max(8, int(12 * self.zoom))))
            label = str(index + 1)
            painter.drawText(canvas_rect.topLeft() + QPointF(4, 16), label)
            if selected:
                painter.setBrush(QColor("#00e676"))
                for point in (canvas_rect.topLeft(), canvas_rect.topRight(), canvas_rect.bottomLeft(), canvas_rect.bottomRight()):
                    painter.drawEllipse(point, 5, 5)


class MainWindow(QMainWindow):
    def __init__(
        self, images_dir: Path, labels_dir: Path, typed_dir: Path, proposals_dir: Path,
        logical_manifests_dir: Path, output_dir: Path, ocr_alignment_dir: Path | None = None,
        ready_dir: Path | None = None, priority_csv: Path | None = None, project_dir: Path | None = None,
    ):
        super().__init__()
        self.images_dir, self.labels_dir, self.typed_dir = images_dir, labels_dir, typed_dir
        self.proposals_dir, self.logical_manifests_dir, self.output_dir = proposals_dir, logical_manifests_dir, output_dir
        self.project_dir = project_dir.resolve() if project_dir else None
        self.project_paths = ensure_project(self.project_dir) if self.project_dir else None
        self.line_pairs_dir = self.project_paths["line_pairs"] if self.project_paths else output_dir / "line_pairs"
        self.drafts_dir = self.project_paths["drafts"] if self.project_paths else output_dir / "transcripts"
        self.ocr_alignment_dir = ocr_alignment_dir or output_dir.parent / "ocr_alignment_proposals"
        # Ready pages live outside the active accepted directory. They remain
        # available for later promotion/export, but are never loaded into this
        # review queue again.
        self.ready_dir = ready_dir or output_dir.parent / "ready"
        ready_ids = set() if self.project_dir else {path.stem for path in self.ready_dir.glob("*.json")}
        self.pairs = self.project_pairs() if self.project_dir else [pair for pair in find_pairs(images_dir, labels_dir, typed_dir) if pair[0] not in ready_ids]
        ranks = priority_order(priority_csv)
        if ranks:
            self.pairs.sort(key=lambda pair: (ranks.get(pair[0], len(ranks) + 1), pair[0]))
        self.index = 0
        self.current_id = ""
        self.current_label_path: Path | None = None
        self.original_image = QImage()
        self.image_rotation_degrees = 0
        self.lines: list[str] = []
        self.span_hints: list[str] = []
        self.kraken_process: QProcess | None = None
        self.kraken_output_path: Path | None = None
        self.ocr_alignment_process: QProcess | None = None
        self.ocr_alignment_page_id: str | None = None
        self.dirty = False

        self.setWindowTitle("OCR Dataset Annotation")
        self.resize(1500, 950)
        self.canvas = ImageCanvas()
        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scroll = scroll
        self.typed_canvas = ImageCanvas()
        typed_scroll = QScrollArea()
        typed_scroll.setWidget(self.typed_canvas)
        typed_scroll.setWidgetResizable(True)
        self.typed_scroll = typed_scroll
        self.image_splitter = QSplitter(Qt.Horizontal)
        self.image_splitter.addWidget(scroll)
        if not self.project_dir:
            self.image_splitter.addWidget(typed_scroll)
        self.image_splitter.setStretchFactor(0, 1)
        self.line_list = QListWidget()
        self.line_list.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.line_list.currentRowChanged.connect(self.select_text_line)
        self.line_list.itemChanged.connect(self.text_line_changed)
        self.line_list.itemDoubleClicked.connect(lambda item: self.canvas.setFocus())

        self.import_button = QPushButton("Import page + transcript")
        self.prev_button = QPushButton("◀ Previous")
        self.next_button = QPushButton("Next ▶")
        self.draw_button = QPushButton("Draw line region")
        self.auto_button = QPushButton("Auto-detect line regions")
        self.rotate_button = QPushButton("Rotate handwritten image ↻")
        self.kraken_button = QPushButton("Re-segment current orientation with Kraken")
        self.delete_button = QPushButton("Delete selected")
        self.assign_button = QPushButton("Assign text to selected region")
        self.assign_sequential_button = QPushButton("Assign all by box number")
        self.box_number_spin = QSpinBox()
        self.box_number_spin.setPrefix("Box # ")
        self.box_number_spin.setRange(1, 1)
        self.set_box_number_button = QPushButton("Set selected box number")
        self.deskew_spin = QDoubleSpinBox()
        self.deskew_spin.setPrefix("Deskew ")
        self.deskew_spin.setSuffix("°")
        self.deskew_spin.setRange(-10.0, 10.0)
        self.deskew_spin.setDecimals(2)
        self.deskew_spin.setSingleStep(0.25)
        self.apply_deskew_button = QPushButton("Apply deskew")
        self.save_text_button = QPushButton("Save transcript edits")
        self.insert_text_button = QPushButton("Insert text line below")
        self.delete_text_button = QPushButton("Delete selected text line")
        self.save_button = QPushButton("Save (Ctrl+S)")
        self.ready_button = QPushButton("Export reviewed page")
        self.save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        self.save_shortcut.activated.connect(self.save)
        self.import_button.clicked.connect(self.import_page_pair)
        self.prev_button.clicked.connect(lambda: self.change_page(-1))
        self.next_button.clicked.connect(lambda: self.change_page(1))
        self.draw_button.clicked.connect(self.start_draw)
        self.auto_button.clicked.connect(self.auto_detect)
        self.rotate_button.clicked.connect(self.rotate_handwritten_image)
        self.kraken_button.clicked.connect(self.resegment_with_kraken)
        self.delete_button.clicked.connect(self.delete_selected)
        self.assign_button.clicked.connect(self.assign_text)
        self.assign_sequential_button.clicked.connect(self.assign_sequential)
        self.set_box_number_button.clicked.connect(self.set_selected_box_number)
        self.apply_deskew_button.clicked.connect(self.apply_deskew)
        self.save_text_button.clicked.connect(self.save_transcript_edits)
        self.insert_text_button.clicked.connect(self.insert_text_line)
        self.delete_text_button.clicked.connect(self.delete_text_line)
        self.save_button.clicked.connect(self.save)
        self.ready_button.clicked.connect(self.export_reviewed_page if self.project_dir else self.mark_complete_and_remove_from_queue)
        self.canvas.region_selected.connect(self.select_region)
        self.canvas.regions_changed.connect(self.mark_dirty)
        self.canvas.regions_changed.connect(self.update_box_number_controls)

        self.page_label = QLabel()
        self.zoom_label = QLabel("Zoom")
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(5, 100)
        self.zoom_slider.valueChanged.connect(self.canvas.set_zoom)
        self.typed_zoom_label = QLabel("Typed zoom")
        self.typed_zoom_slider = QSlider(Qt.Horizontal)
        self.typed_zoom_slider.setRange(5, 100)
        self.typed_zoom_slider.valueChanged.connect(self.typed_canvas.set_zoom)
        self.fit_button = QPushButton("Fit")
        self.fit_button.clicked.connect(self.fit_image)
        self.help_label = QLabel(
            "Transcript text: double-click a row to edit it. Use Insert/Delete to correct line splitting.\n"
            "Draw: select the matching text row, click ‘Draw line region’, then drag a rectangle.\n"
            "Edit: drag inside to move; drag corner handles to resize.\n"
            "Keyboard: arrows move selected region; Shift+arrows move 5 px; Delete removes it."
        )
        self.help_label.setWordWrap(True)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self.import_button)
        right_layout.addWidget(self.page_label)
        right_layout.addWidget(QLabel("Transcript lines — double-click to edit:"))
        right_layout.addWidget(self.line_list, 1)
        right_layout.addWidget(self.auto_button)
        right_layout.addWidget(self.rotate_button)
        right_layout.addWidget(self.kraken_button)
        right_layout.addWidget(self.draw_button)
        right_layout.addWidget(self.delete_button)
        right_layout.addWidget(self.assign_button)
        right_layout.addWidget(self.assign_sequential_button)
        right_layout.addWidget(self.box_number_spin)
        right_layout.addWidget(self.set_box_number_button)
        right_layout.addWidget(self.deskew_spin)
        right_layout.addWidget(self.apply_deskew_button)
        right_layout.addWidget(self.save_text_button)
        right_layout.addWidget(self.insert_text_button)
        right_layout.addWidget(self.delete_text_button)
        right_layout.addWidget(self.save_button)
        right_layout.addWidget(self.ready_button)
        right_layout.addWidget(self.help_label)

        top = QWidget()
        top_layout = QHBoxLayout(top)
        top_layout.addWidget(self.prev_button)
        top_layout.addWidget(self.next_button)
        top_layout.addWidget(self.zoom_label)
        top_layout.addWidget(self.zoom_slider, 1)
        if not self.project_dir:
            top_layout.addWidget(self.typed_zoom_label)
            top_layout.addWidget(self.typed_zoom_slider, 1)
        top_layout.addWidget(self.fit_button)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.image_splitter)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([1100, 380])
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(top)
        layout.addWidget(splitter, 1)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

        self.load_page()

    def mark_dirty(self) -> None:
        self.dirty = True
        self.statusBar().showMessage("Unsaved changes")

    def project_pairs(self) -> list[tuple[str, Path, Path, Path | None]]:
        """Load only manifest-declared pairs; source files are never paired by position."""
        if not self.project_dir:
            return []
        return [(page_id, image, transcript, None) for page_id, image, transcript in read_manifest(self.project_dir)]

    def import_page_pair(self) -> None:
        """Add one immutable source image/transcript pair to the open project."""
        if not self.project_dir:
            self.statusBar().showMessage("Use --project to import into a standalone project.")
            return
        image_name, _ = QFileDialog.getOpenFileName(
            self, "Choose page scan", str(self.project_dir), "Page scans (*.png *.jpg *.jpeg *.tif *.tiff)"
        )
        if not image_name:
            return
        transcript_name, _ = QFileDialog.getOpenFileName(
            self, "Choose its UTF-8 transcript", str(Path(image_name).parent), "Text files (*.txt)"
        )
        if not transcript_name:
            return
        try:
            page_id = import_page(self.project_dir, Path(image_name), Path(transcript_name))
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Import page", str(error))
            return
        self.pairs = self.project_pairs()
        self.index = next((index for index, pair in enumerate(self.pairs) if pair[0] == page_id), 0)
        self.dirty = False
        self.load_page()
        self.statusBar().showMessage(f"Imported {page_id}. Add or review its line boxes.")

    def export_reviewed_page(self) -> None:
        """Export only a fully reviewed one-to-one set of line pairs."""
        complete, message = self.ready_annotation_is_complete()
        if not complete:
            QMessageBox.warning(self, "Page is not ready to export", message)
            return
        self.save()
        QMessageBox.information(
            self,
            "Reviewed line pairs exported",
            f"Saved {len(self.canvas.regions)} reviewed line-image/text pairs to:\n{self.line_pairs_dir / self.current_id}",
        )

    def start_draw(self) -> None:
        selected_row = self.line_list.currentRow()
        if selected_row < 0:
            self.statusBar().showMessage("Select a transcript line first")
            return
        self.canvas.mode = "draw"
        self.canvas.pending_text_index = selected_row
        self.canvas.setFocus()
        self.statusBar().showMessage("Drag across a text line on the image")

    def auto_detect(self) -> None:
        if self.canvas.image.isNull():
            return
        if self.canvas.regions:
            answer = QMessageBox.question(
                self,
                "Replace line regions?",
                "Automatic detection will replace the current regions. Continue?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        regions = detect_line_regions(self.canvas.image)
        if len(regions) == len(self.lines):
            for index, region in enumerate(regions):
                region.text_index = index
            assignment = "text assigned in top-to-bottom order"
        else:
            assignment = "text left unassigned because the counts differ"
        self.canvas.set_regions(regions)
        self.mark_dirty()
        self.statusBar().showMessage(
            f"Proposed {len(regions)} regions for {len(self.lines)} transcript lines; {assignment}. Review before saving."
        )

    def assign_text(self) -> None:
        region_index = self.canvas.selected
        text_index = self.line_list.currentRow()
        if region_index < 0 or text_index < 0:
            self.statusBar().showMessage("Select a region and a transcript line first")
            return
        self.canvas.regions[region_index].text_index = text_index
        self.canvas.regions_changed.emit()
        self.canvas.update()
        self.statusBar().showMessage(f"Assigned transcript line {text_index + 1} to region {region_index + 1}")

    def assign_sequential(self) -> None:
        """Explicitly pair box N with transcript line N in displayed order."""
        if len(self.canvas.regions) != len(self.lines):
            self.statusBar().showMessage(
                f"Cannot assign sequentially: {len(self.canvas.regions)} regions but {len(self.lines)} text lines."
            )
            return
        for text_index, region in enumerate(self.canvas.regions):
            region.text_index = text_index
        self.canvas.regions_changed.emit()
        self.canvas.update()
        self.statusBar().showMessage("Assigned box N to transcript line N using the displayed box numbers.")

    def shift_selected_boundary(self, direction: int) -> None:
        """Move one word across the selected row's right-hand boundary.

        This is safer than editing both neighbouring rows manually: it cannot
        lose, duplicate, or reorder a word in the candidate transcript stream.
        """
        row = self.line_list.currentRow()
        if row < 0 or row >= len(self.lines) - 1:
            self.statusBar().showMessage("Select a row that has a following row to adjust its boundary.")
            return
        left_words = self.lines[row].split()
        right_words = self.lines[row + 1].split()
        if direction > 0:
            if len(right_words) <= 1:
                self.statusBar().showMessage("Cannot take the next word: the following candidate would become empty.")
                return
            left_words.append(right_words.pop(0))
        else:
            if len(left_words) <= 1:
                self.statusBar().showMessage("Cannot give the last word: the selected candidate would become empty.")
                return
            right_words.insert(0, left_words.pop())
        self.lines[row] = " ".join(left_words)
        self.lines[row + 1] = " ".join(right_words)
        self.span_hints = []
        self.populate_line_list()
        self.line_list.setCurrentRow(row)
        self.mark_dirty()
        self.statusBar().showMessage("Moved one word across the boundary; the transcript stream remains contiguous.")

    def propose_transcript_spans(self) -> None:
        """Show editable, unverified text candidates for the displayed boxes."""
        self.statusBar().showMessage("Transcript span proposals are not part of the core reviewer.")

    def run_v7_alignment(self) -> None:
        """Ask v7 to format the known transcript into the verified line boxes.

        The subprocess receives only the current saved annotation and its known
        transcript. Its output is a review proposal: v7 supplies approximate
        visual evidence, while every text row loaded back into this UI is
        copied from the known transcript, never from OCR output.
        """
        if self.ocr_alignment_process is not None:
            self.statusBar().showMessage("v7 transcript-formatting is already running for this page.")
            return
        if not self.canvas.regions:
            QMessageBox.information(
                self,
                "Verify line boxes first",
                "Create and verify the Kraken line boxes before asking v7 to propose transcript line breaks.",
            )
            return
        answer = QMessageBox.question(
            self,
            "Use v7 for transcript formatting?",
            "The current boxes and transcript edits will be saved first. v7 will use the boxes only to propose "
            "where the already-known transcript should break. It will not create ground truth, assign boxes, "
            "or mark this page complete. Continue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        # Persist the exact current geometry and transcript stream so the
        # background process never reads a different page state than the UI.
        self.save()
        model = PROJECT_ROOT / "outputs/unicode_line_ocr_v7/best_recognizer.keras"
        vocabulary = PROJECT_ROOT / "outputs/unicode_line_ocr_v7/vocabulary.json"
        script = PROJECT_ROOT / "scripts/line_ocr/build_ocr_alignment_proposals.py"
        if not model.is_file() or not vocabulary.is_file() or not script.is_file():
            QMessageBox.warning(self, "v7 unavailable", "The v7 model, vocabulary, or alignment script is missing.")
            return

        process = QProcess(self)
        process.setProcessChannelMode(QProcess.SeparateChannels)
        process.finished.connect(self.v7_alignment_finished)
        process.errorOccurred.connect(self.v7_alignment_error)
        self.ocr_alignment_process = process
        self.ocr_alignment_page_id = self.current_id
        self.run_v7_alignment_button.setEnabled(False)
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.statusBar().showMessage(f"v7 is proposing transcript line breaks for {self.current_id}…")
        process.start(
            sys.executable,
            [
                "-u", str(script),
                "--annotations", str(self.output_dir),
                "--images", str(self.images_dir),
                "--transcripts", str(self.output_dir / "transcripts"),
                "--output", str(self.ocr_alignment_dir),
                "--model", str(model),
                "--vocabulary", str(vocabulary),
                "--page-id", self.current_id,
                "--force",
            ],
        )

    def v7_alignment_error(self, error) -> None:
        """Restore controls if the v7 formatter subprocess cannot launch."""
        if error != QProcess.FailedToStart or self.ocr_alignment_process is None:
            return
        process = self.ocr_alignment_process
        self.ocr_alignment_process = None
        self.ocr_alignment_page_id = None
        self.run_v7_alignment_button.setEnabled(True)
        self.prev_button.setEnabled(self.index > 0)
        self.next_button.setEnabled(self.index < len(self.pairs) - 1)
        QMessageBox.warning(self, "v7 could not start", process.errorString())

    def v7_alignment_finished(self, exit_code: int, _exit_status) -> None:
        """Offer to load a successfully generated exact-text proposal."""
        process = self.ocr_alignment_process
        page_id = self.ocr_alignment_page_id
        self.ocr_alignment_process = None
        self.ocr_alignment_page_id = None
        self.run_v7_alignment_button.setEnabled(True)
        self.prev_button.setEnabled(self.index > 0)
        self.next_button.setEnabled(self.index < len(self.pairs) - 1)
        stderr = bytes(process.readAllStandardError()).decode("utf-8", errors="replace") if process else ""
        if exit_code != 0 or page_id != self.current_id:
            detail = stderr.strip()[-3500:] or "v7 did not write a transcript-format proposal."
            QMessageBox.warning(self, "v7 transcript formatting failed", detail)
            return
        path = self.ocr_alignment_dir / f"{self.current_id}.json"
        if not path.is_file():
            QMessageBox.warning(self, "v7 transcript formatting failed", "The v7 process ended without writing a proposal file.")
            return
        self.statusBar().showMessage("v7 created exact transcript line-break candidates; review them before assigning boxes.")
        self.load_ocr_alignment_proposal()

    def load_ocr_alignment_proposal(self) -> None:
        """Load exact transcript candidates inferred from v7 OCR evidence."""
        path = self.ocr_alignment_dir / f"{self.current_id}.json"
        if not path.is_file():
            self.statusBar().showMessage("No v7 alignment proposal for this page. Verify boxes, then use the v7 transcript-format button.")
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            candidates = data.get("candidates", [])
            if data.get("status") != "ocr_guided_ground_truth_candidates_require_review" or len(candidates) != len(self.canvas.regions):
                raise ValueError("proposal does not match the displayed verified regions")
            texts = [str(item["ground_truth_candidate"]).strip() for item in candidates]
            if any(not text for text in texts):
                raise ValueError("proposal contains an empty ground-truth candidate")
        except (OSError, ValueError, TypeError, KeyError) as error:
            QMessageBox.warning(self, "OCR alignment proposal", f"Could not load proposal: {error}")
            return
        answer = QMessageBox.question(
            self,
            "Load v7 transcript-format proposal?",
            "This replaces the displayed transcript rows with exact text from the known page transcript. "
            "It clears all box assignments so you must review the boundaries and explicitly assign the rows before marking the page complete.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self.lines = texts
        self.span_hints = [
            f"v7 OCR: {item.get('ocr_prediction', '')}\n"
            f"confidence: {item.get('mean_confidence', 'n/a')}; OCR/transcript distance: {item.get('ocr_edit_distance', 'n/a')}\n"
            f"{item.get('review_status', 'requires_review')}"
            for item in candidates
        ]
        self.populate_line_list()
        for region in self.canvas.regions:
            region.text_index = None
        self.mark_dirty()
        self.statusBar().showMessage("Loaded exact transcript candidates from v7. Review boundaries, then assign by box number.")

    def set_selected_box_number(self) -> None:
        """Move the selected region to an explicit displayed box position."""
        source = self.canvas.selected
        if source < 0 or source >= len(self.canvas.regions):
            self.statusBar().showMessage("Select a box on the image first.")
            return
        target = min(self.box_number_spin.value() - 1, len(self.canvas.regions) - 1)
        if source == target:
            self.statusBar().showMessage(f"Selected region is already box {target + 1}.")
            return
        region = self.canvas.regions.pop(source)
        self.canvas.regions.insert(target, region)
        self.canvas.selected = target
        self.canvas.regions_changed.emit()
        self.canvas.update()
        self.statusBar().showMessage(f"Moved selected region to box {target + 1}.")

    def update_box_number_controls(self) -> None:
        count = max(1, len(self.canvas.regions))
        self.box_number_spin.setRange(1, count)
        if self.canvas.selected >= 0:
            self.box_number_spin.setValue(self.canvas.selected + 1)

    def transcript_draft_path(self) -> Path:
        """Return the per-page editable transcript, separate from source labels."""
        return self.drafts_dir / f"{self.current_id}.txt"

    def save_transcript_edits(self) -> None:
        """Persist edited text without treating the current geometry as accepted."""
        if not self.current_id:
            return
        path = self.transcript_draft_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        self.statusBar().showMessage(f"Saved editable transcript: {path.name}")

    def load_transcript_draft(self) -> list[str] | None:
        path = self.transcript_draft_path()
        if not path.exists():
            return None
        try:
            return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except OSError as error:
            QMessageBox.warning(self, "Transcript error", f"Could not load {path}: {error}")
            return None

    def delete_selected(self) -> None:
        if self.canvas.selected < 0:
            return
        self.canvas.regions.pop(self.canvas.selected)
        self.canvas.selected = min(self.canvas.selected, len(self.canvas.regions) - 1)
        self.canvas.regions_changed.emit()
        self.canvas.update()

    def text_line_changed(self, item: QListWidgetItem) -> None:
        row = self.line_list.row(item)
        if row < 0:
            return
        text = item.text().strip()
        if text != item.text():
            item.setText(text)
        if row < len(self.lines):
            self.lines[row] = text
        self.mark_dirty()

    def insert_text_line(self) -> None:
        row = self.line_list.currentRow()
        insert_at = row + 1 if row >= 0 else self.line_list.count()
        self.lines.insert(insert_at, "")
        for region in self.canvas.regions:
            if region.text_index is not None and region.text_index >= insert_at:
                region.text_index += 1
        item = QListWidgetItem("")
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        self.line_list.insertItem(insert_at, item)
        self.line_list.setCurrentRow(insert_at)
        self.line_list.editItem(item)
        self.mark_dirty()

    def delete_text_line(self) -> None:
        row = self.line_list.currentRow()
        if row < 0:
            return
        self.lines.pop(row)
        for region in self.canvas.regions:
            if region.text_index == row:
                region.text_index = None
            elif region.text_index is not None and region.text_index > row:
                region.text_index -= 1
        self.line_list.takeItem(row)
        self.mark_dirty()

    def fit_image(self) -> None:
        viewport = self.scroll.viewport().size()
        self.canvas.fit_to_view(viewport.width(), viewport.height())
        typed_viewport = self.typed_scroll.viewport().size()
        self.typed_canvas.fit_to_view(typed_viewport.width(), typed_viewport.height())
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(round(self.canvas.zoom * 100))
        self.zoom_slider.blockSignals(False)
        self.typed_zoom_slider.blockSignals(True)
        self.typed_zoom_slider.setValue(round(self.typed_canvas.zoom * 100))
        self.typed_zoom_slider.blockSignals(False)
        self.canvas.update()
        self.typed_canvas.update()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.fit_image()

    def select_region(self, index: int) -> None:
        if 0 <= index < len(self.canvas.regions):
            self.update_box_number_controls()
            text_index = self.canvas.regions[index].text_index
            self.line_list.blockSignals(True)
            self.line_list.setCurrentRow(text_index if text_index is not None else -1)
            self.line_list.blockSignals(False)
            self.canvas.setFocus()

    def select_text_line(self, row: int) -> None:
        if row < 0:
            return
        for index, region in enumerate(self.canvas.regions):
            if region.text_index == row:
                self.canvas.selected = index
                self.canvas.update()
                return
        # The next drawn rectangle will use this transcript line.
        self.statusBar().showMessage(f"Selected transcript line {row + 1}; draw its image region")

    def load_page(self) -> None:
        if not self.pairs:
            self.page_label.setText("No image/label pairs found")
            self.canvas.set_image(QImage())
            self.typed_canvas.set_image(QImage())
            self.line_list.clear()
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            self.ready_button.setEnabled(False)
            self.statusBar().showMessage("The active review queue is empty. Completed pages are in the Ready list.")
            return
        self.current_id, image_path, self.current_label_path, typed_path = self.pairs[self.index]
        self.image_rotation_degrees = 0
        self.lines = transcript_lines(self.current_label_path)
        self.span_hints = []
        logical_manifest = self.load_logical_manifest()
        typed_regions: list[Region] = []
        if logical_manifest and logical_manifest.get("typed_text_candidates"):
            self.lines = [item["text"] for item in logical_manifest["typed_text_candidates"]]
            typed_regions = [
                Region.from_dict({**item["geometry"], "text_index": line_index})
                for line_index, item in enumerate(logical_manifest.get("typed", []))
            ]
        self.populate_line_list()
        self.original_image = QImage(str(image_path))
        if self.original_image.isNull():
            QMessageBox.warning(self, "Image error", f"Could not load {image_path}")
        self.canvas.set_image(self.original_image)
        if typed_path and typed_path.exists():
            self.typed_canvas.set_image(QImage(str(typed_path)))
            self.typed_canvas.set_regions(typed_regions)
        else:
            self.typed_canvas.set_image(QImage())
        self.load_annotation()
        self.update_box_number_controls()
        # A separately saved text edit is newer than either the immutable
        # source transcript or an older accepted annotation. If its line
        # sequence changed, existing numeric assignments are unsafe and must
        # be made again explicitly by the reviewer.
        draft_lines = self.load_transcript_draft()
        if draft_lines is not None:
            previous_lines = self.lines
            self.lines = draft_lines
            self.populate_line_list()
            if draft_lines != previous_lines:
                for region in self.canvas.regions:
                    region.text_index = None
                self.statusBar().showMessage("Loaded newer transcript edits; cleared stale text assignments.")
        typed_name = typed_path.name if typed_path and typed_path.exists() else "missing"
        self.page_label.setText(f"Page {self.index + 1}/{len(self.pairs)} — handwritten: {image_path.name} — typed: {typed_name}")
        self.prev_button.setEnabled(self.index > 0)
        self.next_button.setEnabled(self.index < len(self.pairs) - 1)
        self.ready_button.setEnabled(True)
        self.dirty = False
        status = "typed line candidates require review" if logical_manifest and logical_manifest.get("typed_text_candidates") else "source transcript lines"
        self.statusBar().showMessage(f"Loaded {len(self.lines)} {status}")
        self.fit_image()

    def annotation_path(self) -> Path:
        return self.output_dir / f"{self.current_id}.json"

    def proposal_path(self) -> Path:
        return self.proposals_dir / f"{self.current_id}.json"

    def logical_manifest_path(self) -> Path:
        return self.logical_manifests_dir / f"{self.current_id}.json"

    def load_logical_manifest(self) -> dict | None:
        path = self.logical_manifest_path()
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as error:
            QMessageBox.warning(self, "Logical manifest error", f"Could not load {path}: {error}")
            return None

    def populate_line_list(self) -> None:
        self.line_list.blockSignals(True)
        self.line_list.clear()
        for number, text in enumerate(self.lines, 1):
            item = QListWidgetItem(text)
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            hint = self.span_hints[number - 1] if number <= len(self.span_hints) else None
            item.setToolTip(f"Line {number}: {text}" + (f"\n{hint}" if hint else ""))
            self.line_list.addItem(item)
        self.line_list.blockSignals(False)

    def preview_path(self) -> Path:
        return self.output_dir / f"{self.current_id}_preview.png"

    def load_annotation(self) -> None:
        path = self.annotation_path()
        is_proposal = False
        if not path.exists():
            path = self.proposal_path()
            is_proposal = True
        if not path.exists():
            self.canvas.set_regions([])
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # Persist corrected transcript splitting, not just region geometry.
            if not is_proposal and isinstance(data.get("transcript_lines"), list):
                self.lines = [str(line) for line in data["transcript_lines"]]
                self.populate_line_list()
            regions = [Region.from_dict(item) for item in data.get("regions", [])]
            if not is_proposal:
                rotation = float(data.get("image_rotation_degrees", 0))
                if not -360 <= rotation <= 360:
                    raise ValueError(f"Unsupported saved image rotation: {rotation}")
                self.image_rotation_degrees = rotation
                if abs(rotation) > 0.001:
                    self.canvas.set_image(self.rotated_original_image())
            self.canvas.set_regions(regions)
            if is_proposal:
                self.statusBar().showMessage("Loaded Kraken proposal; review and save to accept it.")
        except (OSError, ValueError, KeyError, TypeError) as error:
            QMessageBox.warning(self, "Annotation error", f"Could not load {path}: {error}")
            self.canvas.set_regions([])

    def rotated_original_image(self) -> QImage:
        """Return the source page at the persisted display orientation."""
        if abs(self.image_rotation_degrees) <= 0.001:
            return self.original_image.copy()
        return self.original_image.transformed(QTransform().rotate(self.image_rotation_degrees))

    def rotate_display(self, degrees: float) -> None:
        """Rotate the current page and map every box into the new image frame."""
        if self.canvas.image.isNull() or abs(degrees) <= 0.001:
            return
        old_width, old_height = self.canvas.image.width(), self.canvas.image.height()
        transform = QTransform().rotate(degrees)
        old_bounds = transform.mapRect(QRectF(0, 0, old_width, old_height))
        rotated_regions = []
        for region in self.canvas.regions:
            mapped = transform.mapRect(region.rect())
            rotated_regions.append(Region(
                mapped.x() - old_bounds.x(),
                mapped.y() - old_bounds.y(),
                mapped.width(),
                mapped.height(),
                region.text_index,
            ))
        self.image_rotation_degrees += degrees
        # Keep saved rotations readable and avoid unbounded accumulation.
        if self.image_rotation_degrees > 180:
            self.image_rotation_degrees -= 360
        elif self.image_rotation_degrees <= -180:
            self.image_rotation_degrees += 360
        self.canvas.set_image(self.canvas.image.transformed(transform))
        self.canvas.set_regions(rotated_regions)
        self.mark_dirty()
        self.fit_image()
        self.statusBar().showMessage(f"Rotated page by {degrees:+.2f}°; line boxes were rotated with it.")

    def apply_deskew(self) -> None:
        """Apply the requested small rotation before refining line boxes."""
        degrees = float(self.deskew_spin.value())
        if abs(degrees) <= 0.001:
            self.statusBar().showMessage("Choose a non-zero deskew angle first.")
            return
        self.rotate_display(degrees)
        self.deskew_spin.setValue(0.0)

    def rotate_handwritten_image(self) -> None:
        """Rotate the review canvas clockwise and keep all boxes aligned."""
        self.rotate_display(90.0)

    def resegment_with_kraken(self) -> None:
        """Run Kraken on the page exactly as it is currently displayed.

        The source image and its original Kraken output are never changed. A
        rotated working PNG and the resulting native JSON are kept separately
        so the reviewer can rerun segmentation at 0/90/180/270 degrees.
        """
        if self.canvas.image.isNull() or not self.current_id:
            return
        if self.kraken_process is not None:
            self.statusBar().showMessage("Kraken segmentation is already running for this page.")
            return
        answer = QMessageBox.question(
            self,
            "Re-segment with Kraken?",
            "Kraken will process the page at its current displayed rotation "
            f"({self.image_rotation_degrees}°). Its new boxes will replace the displayed boxes and be unassigned. "
            "The original page and original Kraken proposal will be preserved. Continue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        kraken_bin = Path(sys.executable).with_name("kraken")
        executable = str(kraken_bin) if kraken_bin.is_file() else shutil.which("kraken")
        if not executable:
            QMessageBox.warning(
                self,
                "Kraken unavailable",
                "Could not find Kraken in this Python environment or on PATH. Install it only if you want automatic line proposals.",
            )
            return

        run_root = (self.project_paths["annotations"] if self.project_paths else self.output_dir.parent) / "kraken_runs"
        rotation_tag = f"{self.image_rotation_degrees:+07.2f}".replace("+", "p").replace("-", "m").replace(".", "_")
        image_path = run_root / "images" / f"{self.current_id}_rotation_{rotation_tag}.png"
        output_path = run_root / "segments" / f"{self.current_id}_rotation_{rotation_tag}.json"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.canvas.image.save(str(image_path), "PNG"):
            QMessageBox.warning(self, "Working-image error", f"Could not write rotated working image: {image_path}")
            return
        if output_path.exists():
            output_path.unlink()

        process = QProcess(self)
        process.setProcessChannelMode(QProcess.SeparateChannels)
        process.finished.connect(self.kraken_segmentation_finished)
        process.errorOccurred.connect(self.kraken_process_error)
        self.kraken_process = process
        self.kraken_output_path = output_path
        self.kraken_button.setEnabled(False)
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.rotate_button.setEnabled(False)
        self.statusBar().showMessage(f"Running Kraken on {self.current_id} at {self.image_rotation_degrees}°…")
        process.start(
            executable,
            [
                "--raise-on-error", "--device", "cpu", "--threads", "8",
                "-i", str(image_path), str(output_path), "segment", "-bl",
            ],
        )

    def kraken_process_error(self, error) -> None:
        """Restore the UI if the Kraken executable cannot be launched."""
        if error != QProcess.FailedToStart or self.kraken_process is None:
            return
        process = self.kraken_process
        self.kraken_process = None
        self.kraken_output_path = None
        self.kraken_button.setEnabled(True)
        self.prev_button.setEnabled(self.index > 0)
        self.next_button.setEnabled(self.index < len(self.pairs) - 1)
        self.rotate_button.setEnabled(True)
        QMessageBox.warning(self, "Kraken could not start", process.errorString())

    def kraken_segmentation_finished(self, exit_code: int, _exit_status) -> None:
        """Load a completed asynchronous Kraken run into the current canvas."""
        process, output_path = self.kraken_process, self.kraken_output_path
        self.kraken_process = None
        self.kraken_output_path = None
        self.kraken_button.setEnabled(True)
        self.prev_button.setEnabled(self.index > 0)
        self.next_button.setEnabled(self.index < len(self.pairs) - 1)
        self.rotate_button.setEnabled(True)
        stderr = bytes(process.readAllStandardError()).decode("utf-8", errors="replace") if process else ""
        if exit_code != 0 or output_path is None or not output_path.is_file():
            detail = stderr.strip()[-3500:] or "Kraken did not create a segmentation JSON file."
            QMessageBox.warning(self, "Kraken segmentation failed", detail)
            return
        try:
            regions = [Region.from_dict(region) for region in regions_from_kraken(
                json.loads(output_path.read_text(encoding="utf-8"))
            )]
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            QMessageBox.warning(self, "Kraken result error", f"Could not load {output_path.name}: {error}")
            return
        self.canvas.set_regions(regions)
        self.span_hints = []
        self.mark_dirty()
        self.update_box_number_controls()
        self.statusBar().showMessage(
            f"Loaded {len(regions)} Kraken boxes at {self.image_rotation_degrees}°. Review, assign text, then Save."
        )

    def change_page(self, delta: int) -> None:
        if self.dirty:
            answer = QMessageBox.question(self, "Unsaved changes", "Save changes before changing page?", QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            if answer == QMessageBox.Save:
                self.save()
            elif answer == QMessageBox.Cancel:
                return
        self.index = max(0, min(len(self.pairs) - 1, self.index + delta))
        self.load_page()

    def save(self) -> None:
        if not self.current_id or self.canvas.image.isNull():
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_transcript_edits()
        # Preserve the displayed box numbers. In particular, box 27 must stay
        # paired with transcript line 27 after reopening or exporting.
        regions = list(self.canvas.regions)
        payload = {
            "version": 2,
            "image": relative_or_absolute(self.pairs[self.index][1], self.images_dir.parent.parent),
            "ground_truth": relative_or_absolute(self.current_label_path, self.labels_dir.parent.parent),
            "typed_reference": relative_or_absolute(self.pairs[self.index][3], self.typed_dir.parent.parent) if self.pairs[self.index][3] else None,
            "image_width": self.canvas.image.width(),
            "image_height": self.canvas.image.height(),
            "image_rotation_degrees": self.image_rotation_degrees,
            "transcript_lines": self.lines,
            "regions": [item.to_dict(self.lines[item.text_index] if item.text_index is not None and item.text_index < len(self.lines) else None) for item in regions],
        }
        pairs_manifest = self.save_line_pairs(regions)
        payload["line_pairs_manifest"] = relative_or_absolute(pairs_manifest, self.output_dir.parent)
        self.annotation_path().write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.save_preview(regions)
        self.dirty = False
        self.statusBar().showMessage(f"Saved annotation, preview, and line pairs for {len(regions)} regions")

    def ready_annotation_is_complete(self) -> tuple[bool, str]:
        """Require an exact, one-to-one assignment before hiding a page."""
        regions = self.canvas.regions
        if not regions:
            return False, "Add at least one line box before marking this page complete."
        assignments = [region.text_index for region in regions]
        if any(index is None or index < 0 or index >= len(self.lines) for index in assignments):
            return False, "Every box must have an assigned transcript row."
        if any(not self.lines[index].strip() for index in assignments if index is not None):
            return False, "Every assigned transcript row must contain text."
        if len(regions) != len(self.lines) or sorted(assignments) != list(range(len(self.lines))):
            return False, "Completion requires exactly one assigned box for every transcript row."
        return True, ""

    def mark_complete_and_remove_from_queue(self) -> None:
        """Move a validated page to Ready and immediately hide it from review."""
        complete, message = self.ready_annotation_is_complete()
        if not complete:
            QMessageBox.warning(self, "Page is not complete", message)
            return
        answer = QMessageBox.question(
            self,
            "Mark page complete?",
            "This saves the annotation, moves its exported files to the Ready list, "
            "and removes the page from this review queue. Continue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self.save()
        source_annotation = self.annotation_path()
        source_pairs = self.line_pairs_dir / self.current_id
        if not source_annotation.is_file() or not source_pairs.is_dir():
            QMessageBox.warning(self, "Ready-list error", "Could not find the saved annotation or its line-pair export.")
            return
        destination_annotation = self.ready_dir / source_annotation.name
        destination_pairs = self.ready_dir / "line_pairs" / self.current_id
        destination_preview = self.ready_dir / self.preview_path().name
        destination_draft = self.ready_dir / "transcripts" / f"{self.current_id}.txt"
        if destination_annotation.exists() or destination_pairs.exists():
            QMessageBox.warning(self, "Ready-list error", f"{self.current_id} is already in the Ready list; it was not moved.")
            return
        self.ready_dir.mkdir(parents=True, exist_ok=True)
        destination_pairs.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_annotation), str(destination_annotation))
        shutil.move(str(source_pairs), str(destination_pairs))
        preview = self.preview_path()
        if preview.is_file():
            shutil.move(str(preview), str(destination_preview))
        draft = self.transcript_draft_path()
        if draft.is_file():
            destination_draft.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(draft), str(destination_draft))

        completed_id = self.current_id
        self.pairs.pop(self.index)
        if self.index >= len(self.pairs):
            self.index = max(0, len(self.pairs) - 1)
        self.dirty = False
        self.load_page()
        self.statusBar().showMessage(f"Moved {completed_id} to Ready; it will no longer appear in this queue.")

    def save_line_pairs(self, regions: list[Region]) -> Path:
        page_dir = self.line_pairs_dir / self.current_id
        page_dir.mkdir(parents=True, exist_ok=True)
        image_bounds = QRect(0, 0, self.canvas.image.width(), self.canvas.image.height())
        manifest = []
        for number, region in enumerate(regions, 1):
            rect = QRect(
                round(region.x), round(region.y),
                max(1, round(region.width)), max(1, round(region.height)),
            ).intersected(image_bounds)
            image_path = page_dir / f"{self.current_id}_line_{number:03d}.png"
            text_path = page_dir / f"{self.current_id}_line_{number:03d}.txt"
            self.canvas.image.copy(rect).save(str(image_path), "PNG")
            text = ""
            if region.text_index is not None and 0 <= region.text_index < len(self.lines):
                text = self.lines[region.text_index]
            text_path.write_text(text + "\n", encoding="utf-8")
            manifest.append({
                "line_number": number,
                "image": image_path.name,
                "text": text_path.name,
                "text_index": region.text_index,
                "paired": bool(text),
                "x": rect.x(), "y": rect.y(), "width": rect.width(), "height": rect.height(),
            })
        manifest_path = page_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return manifest_path

    def save_preview(self, regions: list[Region]) -> None:
        preview = self.canvas.image.convertToFormat(QImage.Format_ARGB32)
        painter = QPainter(preview)
        for index, region in enumerate(regions, 1):
            painter.setPen(QPen(QColor("#00aa55"), max(3, preview.width() // 800)))
            painter.setBrush(QColor(0, 200, 100, 30))
            painter.drawRect(region.rect())
            painter.setPen(Qt.red)
            painter.setFont(QFont("Sans", max(18, preview.width() // 100)))
            painter.drawText(QPointF(region.x + 5, region.y + 25), str(index))
        painter.end()
        preview.save(str(self.preview_path()), "PNG")

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_S and event.modifiers() & Qt.ControlModifier:
            self.save()
        elif event.key() == Qt.Key_PageDown:
            self.change_page(1)
        elif event.key() == Qt.Key_PageUp:
            self.change_page(-1)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        if not self.dirty:
            event.accept()
            return
        answer = QMessageBox.question(self, "Unsaved changes", "Save changes before exiting?", QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if answer == QMessageBox.Save:
            self.save()
            event.accept()
        elif answer == QMessageBox.Discard:
            event.accept()
        else:
            event.ignore()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project", type=Path, help="Standalone annotation project. Created if it does not exist.")
    parser.add_argument("--images", type=Path, default=project_root / "dataset/images/handwritten", help="Legacy mode only.")
    parser.add_argument("--labels", type=Path, default=project_root / "dataset/labels/clean_from_doc", help="Legacy mode only.")
    parser.add_argument("--typed", type=Path, default=project_root / "dataset/images/typed", help="Legacy mode only.")
    parser.add_argument("--proposals", type=Path, default=project_root / "outputs/line_annotations/proposals", help="Legacy mode only.")
    parser.add_argument("--logical-manifests", type=Path, default=project_root / "outputs/logical_line_manifests", help="Legacy mode only.")
    parser.add_argument("--output", type=Path, default=project_root / "outputs/line_annotations", help="Legacy mode only.")
    parser.add_argument("--ocr-alignment-proposals", type=Path, help="Directory of offline v6 transcript-format proposals.")
    parser.add_argument("--ready", type=Path, help="Directory for validated pages removed from the active review queue.")
    parser.add_argument("--priority", type=Path, help="Optional active-learning page_priority.csv; lowest rank opens first.")
    args = parser.parse_args()
    if args.project:
        paths = ensure_project(args.project)
        args.images = paths["images"]
        args.labels = paths["transcripts"]
        args.typed = paths["root"] / "source" / "typed"
        args.proposals = paths["proposals"]
        args.logical_manifests = paths["root"] / "annotations" / "logical_manifests"
        args.output = paths["pages"]
        args.ready = None
    app = QApplication(sys.argv)
    window = MainWindow(args.images, args.labels, args.typed, args.proposals, args.logical_manifests, args.output, args.ocr_alignment_proposals, args.ready, args.priority, args.project)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
