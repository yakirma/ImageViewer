"""Regression tests for I09 (toggling the crosshair turned off mouse tracking,
so the hover pixel readout stopped) and I43 (hover readout used
QRect.center() = (w-1)/2 while paint uses width()/2, so it could report the
neighbouring pixel).

QOpenGLWidget cannot paint on the offscreen platform, so the label's paint
routine (`_paint_contents`) is rendered into a QImage of the widget's size.
"""
from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QImage, QMouseEvent, QPainter

from widgets import ZoomableDraggableLabel

H, W = 401, 503
VW, VH = 500, 400


def _encoded_rgb(h=H, w=W):
    """R+256*G encodes x, B encodes y % 256."""
    yy, xx = np.mgrid[0:h, 0:w]
    return np.stack([xx % 256, xx // 256, yy % 256], -1).astype(np.uint8)


def _render_rgb(label):
    img = QImage(label.size(), QImage.Format.Format_RGB32)
    img.fill(0)
    p = QPainter(img)
    label._paint_contents(p)
    p.end()
    buf = np.frombuffer(img.constBits().asstring(img.sizeInBytes()), np.uint8)
    arr = buf.reshape(img.height(), img.bytesPerLine() // 4, 4)[:, : img.width()]
    return arr[..., [2, 1, 0]].astype(int)


@pytest.fixture
def label(qtbot):
    lab = ZoomableDraggableLabel()
    qtbot.addWidget(lab)
    lab.resize(VW, VH)
    lab.set_data(_encoded_rgb())
    return lab


def _hover(label, pos):
    got = []
    label.hover_moved.connect(lambda x, y: got.append((x, y)))
    try:
        label._update_crosshair_from_pos(QPoint(*pos))
    finally:
        label.hover_moved.disconnect()
    return got[-1] if got else None


def test_crosshair_toggle_keeps_mouse_tracking(label):
    assert label.hasMouseTracking()
    label.set_crosshair_enabled(True)
    label.set_crosshair_enabled(False)
    assert label.hasMouseTracking()


def test_hover_reports_after_crosshair_toggle(label):
    label.set_crosshair_enabled(True)
    label.set_crosshair_enabled(False)
    got = []
    label.hover_moved.connect(lambda x, y: got.append((x, y)))
    ev = QMouseEvent(QEvent.Type.MouseMove, QPointF(250, 200), QPointF(250, 200),
                     Qt.MouseButton.NoButton, Qt.MouseButton.NoButton,
                     Qt.KeyboardModifier.NoModifier)
    label.mouseMoveEvent(ev)
    assert got, "hover readout should update with no button held"


def _make_label(qtbot, img_size, view_size):
    (iw, ih), (vw, vh) = img_size, view_size
    lab = ZoomableDraggableLabel()
    qtbot.addWidget(lab)
    lab.resize(vw, vh)
    lab.set_data(_encoded_rgb(ih, iw))
    return lab


def _mismatches(label, scale, offset, skip_ties=False):
    label._scale_factor = scale
    label._pixmap_offset = QPointF(*offset)
    img = _render_rgb(label)
    vw, vh = label.width(), label.height()
    ih, iw = label.original_data.shape[:2]
    tl = label._image_top_left()
    bad, checked = [], 0
    for px in range(0, vw, 7):
        for py in range(0, vh, 11):
            if skip_ties:
                # Non-integer scales: skip samples within 0.1 image px of a
                # pixel boundary, where the rasteriser's rounding decides.
                fx = (px + 0.5 - tl.x()) / scale
                fy = (py + 0.5 - tl.y()) / scale
                if min(fx % 1, 1 - fx % 1, fy % 1, 1 - fy % 1) < 0.1:
                    continue
            got = _hover(label, (px, py))
            if got is None:
                continue
            checked += 1
            r, g, b = img[py, px]
            painted = (r + 256 * g, b)
            if (got[0], got[1] % 256) != painted:
                bad.append(((px, py), got, painted))
    assert checked > 0
    return bad


# Integer scales, zero/integer/fractional pans, mixed odd/even image and view
# sizes and a small image in a large view: every sample must agree exactly
# (paint and hover share the device-pixel-snapped origin, I43).
@pytest.mark.parametrize("img_size", [(503, 401), (500, 400), (501, 400), (255, 255)])
@pytest.mark.parametrize("view_size", [(500, 400), (501, 401), (800, 600)])
@pytest.mark.parametrize("scale", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("offset", [(0, 0), (-19, 14), (0.5, -0.5), (100.3, -70.7)])
def test_hover_matches_painted_pixel(qtbot, img_size, view_size, scale, offset):
    label = _make_label(qtbot, img_size, view_size)
    bad = _mismatches(label, scale, offset)
    assert not bad, (len(bad), bad[:5])


@pytest.mark.parametrize("scale, offset", [(1.6, (5.05, -3.1)), (2.5, (0, 0))])
def test_hover_matches_painted_pixel_fractional_scale(qtbot, scale, offset):
    label = _make_label(qtbot, (503, 401), (500, 400))
    bad = _mismatches(label, scale, offset, skip_ties=True)
    assert not bad, (len(bad), bad[:5])


def test_hover_uses_float_event_position(label):
    """mouseMoveEvent maps event.position() (float), not the rounded pos()."""
    label._scale_factor = 2.0
    label._pixmap_offset = QPointF(0, 0)
    seen = []
    orig = label._update_crosshair_from_pos
    label._update_crosshair_from_pos = lambda p: (seen.append(p), orig(p))
    ev = QMouseEvent(QEvent.Type.MouseMove, QPointF(120.75, 80.25),
                     QPointF(120.75, 80.25), Qt.MouseButton.NoButton,
                     Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    label.mouseMoveEvent(ev)
    assert seen and isinstance(seen[0], QPointF)
    assert (seen[0].x(), seen[0].y()) == (120.75, 80.25)


def test_stored_offset_stays_unsnapped(label):
    """Snapping is for drawing only; the pan state keeps sub-pixel values."""
    label._scale_factor = 1.0
    label._pixmap_offset = QPointF(0.3, -0.2)
    _render_rgb(label)
    _hover(label, (100, 100))
    assert (label._pixmap_offset.x(), label._pixmap_offset.y()) == (0.3, -0.2)


def test_hover_edges_use_paint_centre(label):
    """Paint centres the image at (width/2, height/2). With offset (0.5, 0.5)
    at 100% the 503x401 image's left edge is at x = 250.5 - 251.5 = -1 and
    its top edge at y = 200.5 - 200.5 = 0, so widget pixel (0, 0) shows image
    pixel (1, 0). The old (w-1)/2 centring reported (2, 0)/(1, 1)-style
    neighbours at such positions."""
    label._scale_factor = 1.0
    label._pixmap_offset = QPointF(0.5, 0.5)
    assert _hover(label, (0, 0)) == (1, 0)
    assert _hover(label, (VW - 1, VH - 1)) == (VW, VH - 1)
