"""Regression tests for I06 (large images shown from a strided proxy even when
zoomed in), I20 (export/copy saved the proxy) and I31 (zoomed-out thumbnail
chosen from the proxy width instead of the on-screen width).

QOpenGLWidget cannot paint on the offscreen platform, so these render the
label's paint routine (`_paint_contents`) into a QImage of the widget's size.
"""
from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QImage, QKeyEvent, QPainter
from PyQt6.QtWidgets import QApplication, QFileDialog

from widgets import ZoomableDraggableLabel

# 2101 px wide -> proxy step 2, proxy 1051 px (covers 2102 px: not divisible).
H, W = 999, 2101
VW, VH = 400, 300  # even widget size -> integer translation at 100%


def _checker(h=H, w=W):
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx + yy) % 2).astype(np.float32)


def _render(label, painter_cls=QPainter):
    img = QImage(label.size(), QImage.Format.Format_RGB32)
    img.fill(0xFF123456)
    p = painter_cls(img)
    label._paint_contents(p)
    p.end()
    return img


def _gray(img):
    """Red channel of the rendered QImage as an (h, w) uint8 array."""
    img = img.convertToFormat(QImage.Format.Format_RGB32)
    buf = np.frombuffer(img.constBits().asstring(img.sizeInBytes()), np.uint8)
    arr = buf.reshape(img.height(), img.bytesPerLine() // 4, 4)[:, : img.width()]
    return arr[..., 2].copy()  # BGRA in memory -> red


def _label(qtbot, data):
    label = ZoomableDraggableLabel()
    qtbot.addWidget(label)
    label.resize(VW, VH)
    label.set_data(data)
    return label


def _set_view(label, scale, offset=(0.0, 0.0)):
    label._scale_factor = scale
    label._pixmap_offset = QPointF(*offset)


def _expected(scale, offset, data):
    """Nearest-neighbour expected gray values for each widget pixel."""
    h, w = data.shape
    # The image top-left is drawn snapped to a whole pixel (I43).
    left = np.floor(VW / 2 + offset[0] - w * scale / 2 + 0.5)
    top = np.floor(VH / 2 + offset[1] - h * scale / 2 + 0.5)
    xs = (np.arange(VW) + 0.5 - left) / scale
    ys = (np.arange(VH) + 0.5 - top) / scale
    xi, yi = np.floor(xs).astype(int), np.floor(ys).astype(int)
    assert xi.min() >= 0 and xi.max() < w and yi.min() >= 0 and yi.max() < h
    return (data[np.ix_(yi, xi)] * 255).astype(np.uint8)


def test_large_image_uses_proxy_but_keeps_full_res(qtbot):
    label = _label(qtbot, _checker())
    assert label._proxy_scale == 0.5
    assert label.current_pixmap.width() == 1051
    assert label._has_full_res()
    assert label._full_rgb.shape[:2] == (H, W)


@pytest.mark.parametrize("scale,offset", [
    (1.0, (0.0, 0.0)),        # 100% at the centre
    (1.0, (840.0, -340.0)),   # 100% panned far (forces a new tile)
    (1.0, (-837.0, 341.0)),   # other side, odd offset
    (4.0, (0.0, 0.0)),        # 400%
    (3.0, (1234.25, 77.25)),  # 300% panned (offset avoids exact texel ties)
])
def test_zoomed_in_shows_every_source_pixel(qtbot, scale, offset):
    data = _checker()
    label = _label(qtbot, data)
    _set_view(label, scale, offset)
    got = _gray(_render(label))
    np.testing.assert_array_equal(got, _expected(scale, offset, data))


def test_single_pixel_feature_visible_at_100(qtbot):
    data = np.zeros((H, W), np.float32)
    data[H // 2 + 1, W // 2 + 1] = 1.0  # odd coords: dropped by the ::2 proxy
    label = _label(qtbot, data)
    _set_view(label, 1.0)
    got = _gray(_render(label))
    assert got.sum() == 255
    np.testing.assert_array_equal(got, _expected(1.0, (0, 0), data))


def test_tile_follows_colormap_and_contrast_changes(qtbot):
    data = _checker()
    label = _label(qtbot, data)
    _set_view(label, 2.0, (31.0, -17.0))
    _render(label)
    label.set_colormap("gray_r")
    got = _gray(_render(label))
    np.testing.assert_array_equal(got, 255 - _expected(2.0, (31.0, -17.0), data))


def test_interaction_does_not_rebuild_tile_then_final_paint_is_exact(qtbot):
    data = _checker()
    label = _label(qtbot, data)
    _set_view(label, 1.0)
    _render(label)
    tile = label._tile
    assert tile is not None

    label._begin_single_interaction()
    _set_view(label, 1.0, (845.0, 0.0))  # outside the cached tile
    _render(label)  # must not raise; draws proxy + stale tile
    assert label._tile is tile

    label._end_single_interaction()
    got = _gray(_render(label))
    assert label._tile is not tile
    np.testing.assert_array_equal(got, _expected(1.0, (845.0, 0.0), data))


def test_proxy_drawn_over_full_image_extent_when_zoomed_out(qtbot):
    """At fit the proxy (still used) covers the image rect exactly."""
    data = np.ones((H, W), np.float32)
    data[0, 0] = 0.0
    label = _label(qtbot, data)
    label.fit_to_view()
    scale = label._get_effective_scale_factor()
    assert scale < label._proxy_scale
    got = _gray(_render(label))
    # image spans widget width; rows outside the image keep the fill colour
    draw_h = H * scale
    top = int(np.ceil((VH - draw_h) / 2)) + 1
    bottom = int((VH + draw_h) / 2) - 1
    assert (got[top:bottom, 1:-1] == 255).all()


def test_small_image_has_no_full_res_copy(qtbot):
    data = _checker(40, 60)
    label = _label(qtbot, data)
    assert label._proxy_scale == 1.0
    assert not label._has_full_res()
    img = label.full_resolution_image()
    assert (img.width(), img.height()) == (60, 40)


def test_clearing_pixmap_drops_full_res(qtbot):
    label = _label(qtbot, _checker())
    assert label._full_rgb is not None
    label.current_pixmap = None
    assert label._full_rgb is None and label._tile is None


# ---- I20: export / copy use full resolution --------------------------------

def test_full_resolution_image_matches_data(qtbot):
    data = _checker()
    label = _label(qtbot, data)
    img = label.full_resolution_image()
    assert (img.width(), img.height()) == (W, H)
    got = _gray(img)
    np.testing.assert_array_equal(got, (data * 255).astype(np.uint8))


def _inject(viewer, data):
    label = viewer.image_label
    viewer.active_label = label
    label.set_data(data, reset_view=True, is_pristine=True)
    return label


def test_export_current_view_saves_full_resolution(viewer, tmp_path, monkeypatch):
    data = _checker()
    _inject(viewer, data)
    out = tmp_path / "export.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(out), "PNG (*.png)"))
    viewer.export_current_view()
    img = QImage(str(out))
    assert (img.width(), img.height()) == (W, H)
    np.testing.assert_array_equal(_gray(img), (data * 255).astype(np.uint8))


def test_copy_image_copies_full_resolution(viewer):
    _inject(viewer, _checker())
    QApplication.clipboard().clear()
    ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_C,
                   Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
    viewer.keyPressEvent(ev)
    img = QApplication.clipboard().image()
    assert (img.width(), img.height()) == (W, H)


# ---- I31: thumbnail chosen from the true on-screen width ---------------------

class _RecordingPainter(QPainter):
    def __init__(self, dev):
        super().__init__(dev)
        self.drawn_widths = []

    def drawPixmap(self, *args):
        for a in args:
            if hasattr(a, "width") and hasattr(a, "cacheKey"):
                self.drawn_widths.append(a.width())
        return super().drawPixmap(*args)


def _drawn_widths(label):
    img = QImage(label.size(), QImage.Format.Format_RGB32)
    p = _RecordingPainter(img)
    label._paint_contents(p)
    p.end()
    return p.drawn_widths


def test_thumbnail_not_used_when_on_screen_wider_than_it(qtbot):
    label = _label(qtbot, _checker())
    label.resize(1200, 600)
    assert label.thumbnail_pixmap.width() == 800
    _set_view(label, 0.45)  # on screen: 2101 * 0.45 = 945 px > 800
    assert 800 not in _drawn_widths(label)


def test_thumbnail_used_when_small_on_screen(qtbot):
    label = _label(qtbot, _checker())
    _set_view(label, 0.3)  # on screen: ~630 px < 800
    assert _drawn_widths(label) == [800]


# ---- tile size cap (follow-up to I06) ----------------------------------------

def _rgb_checker(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    c = (((xx + yy) % 2) * 255).astype(np.uint8)
    return np.dstack([c, (xx % 256).astype(np.uint8), (yy % 256).astype(np.uint8)])


@pytest.fixture
def wide_label(qtbot, monkeypatch):
    """6001x3000 RGB (proxy step 3) in a wide label just above proxy scale,
    with the tile cap scaled down so the visible region (~3800 px) exceeds it."""
    monkeypatch.setattr(ZoomableDraggableLabel, "_TILE_MAX_DIM", 2000)
    monkeypatch.setattr(ZoomableDraggableLabel, "_TILE_MIN_MARGIN", 128)
    data = _rgb_checker(3000, 6001)
    label = ZoomableDraggableLabel()
    qtbot.addWidget(label)
    label.resize(1280, 720)
    label.set_data(data)
    assert label._proxy_scale == pytest.approx(1 / 3)
    _set_view(label, 1.01 * label._proxy_scale)
    return label, data


def test_tile_capped_and_strided_when_visible_region_is_huge(wide_label):
    label, data = wide_label
    vis = label._visible_image_rect()
    assert vis[2] - vis[0] > 2000  # the uncapped tile would exceed the cap
    _render(label)
    _, (x0, y0, x1, y1), pix, stride = label._tile
    assert stride == 2
    cap = 2000 + 2 * 128
    assert pix.width() <= cap and pix.height() <= cap
    # tile still covers the visible region
    assert x0 <= vis[0] and y0 <= vis[1] and x1 >= vis[2] and y1 >= vis[3]
    # nearest-neighbour subsample of the real pixels
    expected = data[y0:y1:2, x0:x1:2]
    img = pix.toImage().convertToFormat(QImage.Format.Format_RGB888)
    got = np.frombuffer(img.constBits().asstring(img.sizeInBytes()), np.uint8)
    got = got.reshape(img.height(), img.bytesPerLine())[:, : img.width() * 3]
    np.testing.assert_array_equal(got.reshape(img.height(), img.width(), 3), expected)


def test_small_pan_reuses_capped_tile(wide_label):
    label, _ = wide_label
    _render(label)
    tile = label._tile
    _set_view(label, label._scale_factor, (20.0, 10.0))  # < margin on screen
    _render(label)
    assert label._tile is tile


def test_stays_on_proxy_when_stride_would_match_proxy_step(qtbot, wide_label, monkeypatch):
    label, _ = wide_label
    monkeypatch.setattr(ZoomableDraggableLabel, "_TILE_MAX_DIM", 1500)  # stride 3 == step
    label._tile = None
    _render(label)
    assert label._tile is None


def test_real_cap_bounds_tile_for_8000_wide_image_geometry(qtbot):
    """The repro geometry (8000x6000 in 2560x1440 at 1.01x proxy scale)
    with the default cap: stride 2, so the tile is at most ~4608 px."""
    label = ZoomableDraggableLabel()
    qtbot.addWidget(label)
    label.resize(2560, 1440)
    label._proxy_scale = 0.25
    vis = (0, 0, 8000, 5704)
    assert label._tile_stride(vis) == 2
    assert label._tile_stride((0, 0, 4096, 2000)) == 1
    assert label._tile_stride((0, 0, 16000, 9000)) is None  # stride 4 == step
