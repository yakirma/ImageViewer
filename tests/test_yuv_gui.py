"""GUI-level regression tests for I03: YUV raw files must be usable through
open_file / the InfoPane, not only through ImageHandler."""
from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtWidgets import QMessageBox

from image_handler import ImageHandler

W, H = 64, 48


def _nv12_frame():
    """NV12 bytes for a frame with distinct colours per 2x2 block."""
    Y = (16 + np.arange(W * H) % 200).astype(np.uint8)
    cw, ch = W // 2, H // 2
    U = (40 + np.arange(cw * ch) % 180).astype(np.uint8)
    V = (220 - np.arange(cw * ch) % 170).astype(np.uint8)
    return np.concatenate([Y, np.stack([U, V], -1).ravel()])


def _yuyv_frame():
    return (np.arange(W * H * 2) * 37 % 256).astype(np.uint8)


def _expected(payload, fmt):
    return ImageHandler()._process_yuv(payload, W, H, fmt)


@pytest.fixture
def errors(monkeypatch):
    rec = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: rec.append(a[2] if len(a) > 2 else a))
    return rec


def _shown(viewer):
    label = viewer.active_label
    assert label is not None
    return label.original_data


def test_resolution_named_yuv_defaults_to_i420(viewer, tmp_path, errors):
    payload = _nv12_frame()  # 1.5*W*H bytes; decoded as planar yuv420p
    p = tmp_path / f"clip_{W}x{H}.yuv"
    payload.tofile(p)

    viewer.open_file(str(p))

    assert not errors
    assert viewer.image_handler.color_format == "YUV I420"
    np.testing.assert_array_equal(_shown(viewer), _expected(payload, "YUV I420"))
    assert viewer.info_pane.color_format_combo.currentText() == "YUV I420"


def test_failed_resolution_named_raw_keeps_infopane_editable(viewer, tmp_path, errors):
    payload = _nv12_frame()  # 1.5*W*H bytes: not decodable as Grayscale
    p = tmp_path / f"clip_{W}x{H}.raw"
    payload.tofile(p)

    viewer.open_file(str(p))

    assert len(errors) == 1  # load failure is still reported
    ip = viewer.info_pane
    assert ip.color_format_combo.isEnabled()
    assert ip.width_spinbox.value() == W
    assert ip.height_spinbox.value() == H
    assert viewer.info_action.isEnabled()

    ip.color_format_combo.setCurrentText("YUV NV12")

    assert len(errors) == 1
    assert ip.height_spinbox.value() == H
    assert viewer.active_label is viewer.image_label
    np.testing.assert_array_equal(_shown(viewer), _expected(payload, "YUV NV12"))
    assert viewer.image_label.current_pixmap is not None


def test_inherited_yuv_format_with_explicit_resolution(viewer, tmp_path, errors):
    payload = _nv12_frame()
    first = tmp_path / f"a_{W}x{H}.raw"
    second = tmp_path / f"b_{W}x{H}.raw"
    gray = tmp_path / f"g_{W}x{H}.raw"
    payload.tofile(first)
    payload.tofile(second)
    np.arange(W * H, dtype=np.uint8).tofile(gray)

    viewer.open_file(str(first))
    viewer.info_pane.color_format_combo.setCurrentText("YUV NV12")
    errors.clear()

    # Same-size file selected next inherits the YUV layout.
    viewer._on_explorer_files_selected([str(second)])
    assert not errors
    assert viewer.image_handler.color_format == "YUV NV12"
    np.testing.assert_array_equal(_shown(viewer), _expected(payload, "YUV NV12"))

    # A file whose size doesn't match the layout is not forced into it.
    viewer._on_explorer_files_selected([str(gray)])
    assert not errors
    assert viewer.image_handler.color_format == "Grayscale"
    assert _shown(viewer).shape == (H, W)


@pytest.mark.parametrize("ext,fmt,make", [
    (".nv12", "YUV NV12", _nv12_frame),
    (".yuyv", "YUV YUYV", _yuyv_frame),
])
def test_extension_format_used_when_resolution_missing(viewer, tmp_path, errors, ext, fmt, make):
    payload = make()
    p = tmp_path / f"frame{ext}"
    payload.tofile(p)

    viewer.open_file(str(p))

    assert not errors
    assert viewer.image_handler.color_format == fmt
    assert (viewer.image_handler.width, viewer.image_handler.height) == (W, H)
    np.testing.assert_array_equal(_shown(viewer), _expected(payload, fmt))
    assert viewer.info_pane.color_format_combo.currentText() == fmt


def test_failed_raw_recovery_does_not_hijack_montage(viewer, tmp_path, errors):
    bad = tmp_path / f"b_{W}x{H}.raw"
    _nv12_frame().tofile(bad)  # fails as Grayscale -> recovery armed
    montage = []
    for k in range(2):
        p = tmp_path / f"m{k}_{W}x{H}.raw"
        np.full(W * H, 40 * k, np.uint8).tofile(p)
        montage.append(str(p))

    viewer.open_file(str(bad))
    assert len(errors) == 1
    viewer._on_explorer_files_selected(montage)
    assert viewer.stacked_widget.currentWidget() is viewer.montage_widget

    viewer.info_pane.color_format_combo.setCurrentText("Bayer RGGB")

    assert viewer.stacked_widget.currentWidget() is viewer.montage_widget
    assert viewer.active_label is not viewer.image_label
