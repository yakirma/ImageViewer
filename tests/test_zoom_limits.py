"""Regression tests for I23: zoom was clamped to an effective scale >= 0.1, so a
large image whose fit-to-view is below 10% could not be zoomed out past fit,
and the first wheel step (in *or* out) snapped the view up to 10%.
"""
from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent

from widgets import SharedViewState, ZoomableDraggableLabel

VW, VH = 600, 400


def _label(qtbot, shape, shared_state=None, size=(VW, VH)):
    label = ZoomableDraggableLabel(shared_state=shared_state)
    qtbot.addWidget(label)
    label.resize(*size)
    label.set_data(np.zeros(shape, dtype=np.uint8))
    label.fit_to_view()
    return label


def _wheel(label, up, pos=None):
    pos = QPointF(label.width() / 2, label.height() / 2) if pos is None else pos
    ev = QWheelEvent(pos, QPointF(label.mapToGlobal(pos.toPoint())), QPoint(0, 0),
                     QPoint(0, 120 if up else -120), Qt.MouseButton.NoButton,
                     Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    label.wheelEvent(ev)


@pytest.mark.parametrize("shape", [(6000, 8000), (200, 20000)])
def test_large_image_can_zoom_out_below_fit(qtbot, shape):
    label = _label(qtbot, shape)
    fit = label._fit_scale
    assert fit < 0.1
    assert label.zoom_scale == pytest.approx(fit)

    _wheel(label, up=False)
    assert label.zoom_scale == pytest.approx(fit / label.zoom_speed)

    for _ in range(50):
        _wheel(label, up=False)
    # Bottoms out at half of fit, not at 10% (which is above fit).
    assert label.zoom_scale == pytest.approx(fit * 0.5)


@pytest.mark.parametrize("shape", [(6000, 8000), (200, 20000)])
def test_first_zoom_in_from_fit_is_one_step(qtbot, shape):
    label = _label(qtbot, shape)
    fit = label._fit_scale
    _wheel(label, up=True)
    assert label.zoom_scale == pytest.approx(fit * label.zoom_speed)
    assert label.zoom_scale < 0.1


def test_small_image_limits_unchanged(qtbot):
    label = _label(qtbot, (8, 8))
    fit = label._fit_scale
    assert fit > 1
    for _ in range(200):
        _wheel(label, up=False)
    assert label.zoom_scale == pytest.approx(0.1)
    for _ in range(400):
        _wheel(label, up=True)
    # Max zoom: one image pixel fills the view.
    assert label.zoom_scale == pytest.approx(max(VW, VH))


def test_huge_image_max_zoom_still_reachable(qtbot):
    label = _label(qtbot, (200, 20000))
    for _ in range(400):
        _wheel(label, up=True)
    assert label.zoom_scale == pytest.approx(max(VW, VH))


def test_clamp_never_moves_view_the_wrong_way(qtbot):
    """If the view is already below the limit (e.g. after a resize), a zoom-in
    step must not be blocked and a zoom-out must not jump up."""
    label = _label(qtbot, (8, 8))
    label._scale_factor = 0.05  # below the 10% floor for this image
    _wheel(label, up=False)
    assert label.zoom_scale == pytest.approx(0.05)
    _wheel(label, up=True)
    assert label.zoom_scale == pytest.approx(0.05 * label.zoom_speed)


def test_montage_shared_zoom_out_below_fit(qtbot):
    shared = SharedViewState()
    big = _label(qtbot, (6000, 8000), shared_state=shared)
    small = _label(qtbot, (60, 80), shared_state=shared)
    assert shared.zoom_multiplier == pytest.approx(1.0)

    _wheel(big, up=False)
    m = 1.0 / big.zoom_speed
    assert shared.zoom_multiplier == pytest.approx(m)
    # Both tiles share the same relative zoom.
    assert big.zoom_scale == pytest.approx(big._fit_scale * m)
    assert small.zoom_scale == pytest.approx(small._fit_scale * m)

    for _ in range(50):
        _wheel(big, up=False)
    assert shared.zoom_multiplier == pytest.approx(0.5)

    # First zoom-in from fit is a single multiplicative step.
    big.fit_to_view()
    _wheel(big, up=True)
    assert shared.zoom_multiplier == pytest.approx(big.zoom_speed)


def test_zoom_percent_display(viewer, monkeypatch):
    label = viewer.image_label
    viewer.active_label = label
    # The slot ignores signals from non-active labels via sender().
    monkeypatch.setattr(viewer, "sender", lambda: label, raising=False)

    for scale, text in [(0.0375, "Zoom: 3.8%"), (0.075, "Zoom: 7.5%"),
                        (1.0, "Zoom: 100%"), (2.5, "Zoom: 250%")]:
        viewer._on_image_label_zoom_changed(scale)
        assert viewer.zoom_status_label.text() == text
