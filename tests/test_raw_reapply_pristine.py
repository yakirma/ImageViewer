"""Regression tests for I12: applying new raw parameters from the InfoPane.

reapply_raw_parameters used to update only the label's display data, leaving
pristine_data with the previous width/height/format. Math transforms
(x = pristine_data), "restore original" and montage channel selection then
brought back the stale interpretation. A failed apply must also leave the
previous image (handler + label) consistent.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def shown(viewer, qtbot):
    viewer.show()
    qtbot.waitExposed(viewer)
    return viewer


def _raw(tmp_path, name, n):
    p = tmp_path / name
    np.arange(n, dtype=np.uint8).tofile(p)
    return str(p)


GRAY_8x16 = {'width': 8, 'height': 16, 'dtype': 'uint8', 'color_format': 'Grayscale'}


def test_reapply_updates_pristine_math_and_restore(shown, tmp_path):
    v = shown
    v.open_file(_raw(tmp_path, "re_16x8.raw", 128))
    assert v.image_label.pristine_data.shape == (8, 16)

    v.math_transform_pane.show()
    v.apply_math_transform("x*2")
    assert v.image_label.original_data.shape == (8, 16)

    v.reapply_raw_parameters(GRAY_8x16)
    expected = np.arange(128, dtype=np.uint8).reshape(16, 8)

    lbl = v.image_label
    assert lbl.pristine_data.shape == (16, 8)
    np.testing.assert_array_equal(lbl.pristine_data, expected)
    # The active math expression is re-run against the new data.
    np.testing.assert_allclose(lbl.original_data, expected.astype(np.float64) * 2)
    assert lbl.metadata['width'] == 8 and lbl.metadata['height'] == 16

    v.apply_math_transform("x+1")
    np.testing.assert_allclose(lbl.original_data, expected.astype(np.float64) + 1)

    v.restore_original_image()
    np.testing.assert_array_equal(lbl.original_data, expected)
    assert v.histogram_window.data.size == expected.size


def test_reapply_to_rgb_updates_channels_and_histogram(shown, tmp_path):
    v = shown
    # 24 bytes: 24x1 gray by name, 4x2 RGB after reapply.
    v.open_file(_raw(tmp_path, "rgb_24x1.raw", 24))
    assert v.image_label.pristine_data.shape == (1, 24)

    v.reapply_raw_parameters({'width': 4, 'height': 2, 'dtype': 'uint8', 'color_format': 'RGB'})
    rgb = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    assert v.image_label.pristine_data.shape == (2, 4, 3)
    items = [v.channel_combo.itemText(i) for i in range(v.channel_combo.count())]
    assert "R" in items and "G" in items

    v.channel_combo.setCurrentText("G")
    v.update_image_display()
    np.testing.assert_array_equal(v.image_label.original_data, rgb[:, :, 1])
    assert v.histogram_window.data.size == rgb[:, :, 1].size

    v.restore_original_image()
    np.testing.assert_array_equal(v.image_label.original_data, rgb[:, :, 1])


def test_failed_reapply_keeps_previous_state(shown, tmp_path):
    v = shown
    v.open_file(_raw(tmp_path, "re_16x8.raw", 128))
    v.reapply_raw_parameters(GRAY_8x16)
    h = v.image_handler
    before = (h.width, h.height, h.dtype, h.color_format)
    pristine = v.image_label.pristine_data
    meta = dict(v.image_label.metadata)

    # 100x100 doesn't fit a 128-byte file.
    v.reapply_raw_parameters({'width': 100, 'height': 100, 'dtype': 'uint8', 'color_format': 'Grayscale'})

    assert v.info_pane.error_label.text()
    assert (h.width, h.height, h.dtype, h.color_format) == before
    assert h.original_image_data.shape == (16, 8)
    assert v.image_label.pristine_data is pristine
    assert v.image_label.original_data.shape == (16, 8)
    assert v.image_label.metadata == meta

    v.apply_math_transform("x+0")
    assert v.image_label.original_data.shape == (16, 8)


def test_reapply_in_montage_updates_label_pristine(shown, tmp_path):
    v = shown
    a = _raw(tmp_path, "a_16x8.raw", 128)
    b = _raw(tmp_path, "b_16x8.raw", 128)
    v.open_files([a, b])
    assert v.stacked_widget.currentWidget() is v.montage_widget
    lbl = next(l for l in v.montage_labels if l.file_path == b)
    v._set_active_montage_label(lbl)
    assert v.current_file_path == b

    v.reapply_raw_parameters(GRAY_8x16)
    expected = np.arange(128, dtype=np.uint8).reshape(16, 8)
    np.testing.assert_array_equal(lbl.pristine_data, expected)
    np.testing.assert_array_equal(lbl.original_data, expected)

    idx = v.montage_labels.index(lbl) + 1
    v.apply_math_transform(f"x{idx}*3")
    np.testing.assert_allclose(lbl.original_data, expected.astype(np.float64) * 3)
    v.restore_original_image()
    np.testing.assert_array_equal(lbl.original_data, expected)
