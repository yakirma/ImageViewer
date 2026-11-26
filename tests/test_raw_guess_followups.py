"""Regression tests for I10: opening a raw file without a resolution in its name.

- The Info pane must end up visible (it used to be shown and then toggled
  hidden again when the main window was visible).
- A successful load with estimated parameters must update last_raw_settings
  (the early return skipped it), so the next raw file inherits them.
- A bare .yuv is guessed as planar 4:2:0 (YUV I420), not Grayscale 1 bpp.
"""
from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtWidgets import QMessageBox

from image_handler import ImageHandler


@pytest.fixture
def dialogs(monkeypatch):
    rec = []
    rec_fn = lambda *a, **k: rec.append(a[2] if len(a) > 2 else a)
    monkeypatch.setattr(QMessageBox, "warning", rec_fn)
    monkeypatch.setattr(QMessageBox, "critical", rec_fn)
    return rec


@pytest.fixture
def shown_viewer(viewer, qtbot):
    viewer.show()
    qtbot.waitExposed(viewer)
    viewer._set_dock_visibility_preserving_window(viewer.info_pane, False)
    assert not viewer.info_pane.isVisible()
    return viewer


def test_info_pane_visible_after_estimated_load(shown_viewer, tmp_path, dialogs):
    p = tmp_path / "nores.bin"
    np.zeros(1000, np.uint8).tofile(p)

    shown_viewer.open_file(str(p))

    assert not dialogs
    assert shown_viewer.info_pane.isVisible()


def test_info_pane_visible_after_failed_estimate(shown_viewer, tmp_path, dialogs):
    p = tmp_path / "prime.bin"
    np.zeros(997, np.uint8).tofile(p)

    shown_viewer.open_file(str(p))

    assert len(dialogs) == 1
    assert shown_viewer.info_pane.isVisible()


def test_estimated_load_sets_last_raw_settings(viewer, tmp_path, dialogs):
    viewer.last_raw_settings = {"width": 8, "height": 8, "dtype": np.uint8,
                                "color_format": "Grayscale"}
    p = tmp_path / "nores.bin"
    np.zeros(1000, np.uint8).tofile(p)

    viewer.open_file(str(p))

    assert not dialogs
    s = viewer.last_raw_settings
    assert (s["width"], s["height"]) == (40, 25)
    assert s["color_format"] == "Grayscale"


def test_guess_layout_bare_yuv_is_i420(tmp_path):
    p = tmp_path / "clip.yuv"
    np.zeros(640 * 480 * 3 // 2, np.uint8).tofile(p)
    w, h, dtype, fmt, bpp = ImageHandler().guess_raw_layout(str(p))
    assert (w, h, fmt, bpp) == (640, 480, "YUV I420", 1.5)


def test_guess_layout_yuv_falls_back_to_grayscale(tmp_path):
    # 999 bytes = 37*27 pixels at 1 bpp; not a whole 4:2:0 frame (666 px, odd).
    p = tmp_path / "odd.yuv"
    np.zeros(999, np.uint8).tofile(p)
    w, h, _, fmt, bpp = ImageHandler().guess_raw_layout(str(p))
    assert fmt == "Grayscale" and bpp == 1.0 and w * h == 999


def test_open_bare_yuv_decodes_as_i420(viewer, tmp_path, dialogs):
    w, h = 64, 48
    y = np.full((h, w), 128, np.uint8)
    u = np.full((h // 2, w // 2), 128, np.uint8)
    v = np.full((h // 2, w // 2), 128, np.uint8)
    p = tmp_path / "clip.yuv"
    np.concatenate([y.ravel(), u.ravel(), v.ravel()]).tofile(p)

    viewer.open_file(str(p))

    assert not dialogs
    assert viewer.image_handler.color_format == "YUV I420"
    assert viewer.image_handler.original_image_data.shape[:2] == (h, w)
    assert viewer.info_pane.width_spinbox.value() == w
    assert viewer.info_pane.height_spinbox.value() == h
    assert viewer.last_raw_settings["color_format"] == "YUV I420"
