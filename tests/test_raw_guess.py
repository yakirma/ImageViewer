"""Regression tests for I05: raw files without a resolution in the name.

The estimated size must decode when the file allows it, and a failed estimate
must be reported (not swallowed), must not leave the previous image/video UI
on screen, and must leave the Info pane open to recover.
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


def _open_previous(viewer, tmp_path):
    prev = tmp_path / "prev_8x8.raw"
    np.full(64, 7, np.uint8).tofile(prev)
    viewer.open_file(str(prev))
    assert viewer.image_label.original_data.shape == (8, 8)


@pytest.mark.parametrize("size,bpp,fmt,expected", [
    (1000, 1, "Grayscale", (40, 25)),
    (1920 * 1080, 1, "Grayscale", (1920, 1080)),
    (1280 * 720, 1, "Grayscale", (1280, 720)),
    (3840 * 2160, 1, "Grayscale", (3840, 2160)),
    (4096 * 2160 * 2, 2, "Grayscale", (4096, 2160)),
    (2560 * 1440, 1, "Grayscale", (2560, 1440)),
    (4000 * 3000, 1, "Grayscale", (4000, 3000)),
    (1920 * 1080 * 3 // 2, 1.5, "YUV NV12", (1920, 1080)),
    (1920 * 1080 * 2, 2, "YUV YUYV", (1920, 1080)),
    (640 * 480 * 3 // 2, 1.5, "YUV NV12", (640, 480)),
    (640 * 480 * 3, 3, "RGB", (640, 480)),
    (512 * 512 * 4, 4, "Grayscale", (512, 512)),   # square before 4:3
    (300 * 300, 1, "Grayscale", (300, 300)),
    (997, 1, "Grayscale", None),   # prime: only 997x1, too elongated
    (1001, 2, "Grayscale", None),  # not a whole number of uint16 pixels
])
def test_guess_exact_dims(size, bpp, fmt, expected):
    got = ImageHandler.guess_exact_dims(size, bpp, fmt)
    assert got == expected
    if got:
        assert got[0] * got[1] * bpp == size


def test_parse_resolution_does_not_guess(tmp_path):
    p = tmp_path / "square.raw"
    np.zeros(1920 * 1080, np.uint8).tofile(p)  # 1440**2 bytes
    assert ImageHandler().parse_resolution(str(p))[:2] == (0, 0)
    named = tmp_path / "img_1920x1080.raw"
    np.zeros(1920 * 1080, np.uint8).tofile(named)
    assert ImageHandler().parse_resolution(str(named))[:2] == (1920, 1080)


@pytest.mark.parametrize("name,data,shape,fmt", [
    ("frame.raw", np.zeros((1080, 1920), np.uint8), (1080, 1920), "Grayscale"),
    ("depth.f32", np.zeros((512, 512), np.float32), (512, 512), "Grayscale"),
    ("photo.rgb", np.zeros((480, 640, 3), np.uint8), (480, 640, 3), "RGB"),
])
def test_open_file_guesses_from_bytes_per_pixel(viewer, tmp_path, dialogs, name, data, shape, fmt):
    p = tmp_path / name
    data.tofile(p)

    viewer.open_file(str(p))

    assert not dialogs
    assert viewer.image_handler.original_image_data.shape == shape
    assert viewer.image_label.original_data.shape == shape
    assert viewer.image_handler.color_format == fmt
    assert viewer.info_pane.width_spinbox.value() == shape[1]
    assert viewer.info_pane.height_spinbox.value() == shape[0]


def test_handler_load_without_name_uses_guess(tmp_path):
    p = tmp_path / "photo.rgb"
    np.zeros((480, 640, 3), np.uint8).tofile(p)
    h = ImageHandler()
    h.load_image(str(p))  # e.g. the thumbnail pane's path (no override)
    assert h.original_image_data.shape == (480, 640, 3)


def test_guessed_size_decodes_whole_file(viewer, tmp_path, dialogs):
    _open_previous(viewer, tmp_path)
    p = tmp_path / "nores.bin"
    np.arange(1000, dtype=np.uint16).astype(np.uint8).tofile(p)

    viewer.open_file(str(p))

    assert not dialogs
    assert viewer.image_handler.original_image_data.shape == (25, 40)
    assert viewer.image_label.original_data.shape == (25, 40)


def test_failed_guess_is_reported_and_clears_previous_image(viewer, tmp_path, dialogs):
    _open_previous(viewer, tmp_path)
    viewer.video_toolbar.show()  # as if the previous file were a video
    viewer.active_label = None  # as if a montage label had been active
    p = tmp_path / "prime.bin"
    np.arange(997, dtype=np.uint16).astype(np.uint8).tofile(p)

    viewer.open_file(str(p))

    assert len(dialogs) == 1 and "prime.bin" in dialogs[0]
    assert viewer.image_handler.original_image_data is None
    assert viewer.image_label.original_data is None
    assert viewer.image_label.current_pixmap is None
    assert viewer.active_label is viewer.image_label
    assert viewer.video_toolbar.isHidden()
    assert not viewer.histogram_action.isEnabled()
    assert not viewer.info_pane.isHidden()
    assert viewer._raw_recovery_path == str(p)

    # Entering valid parameters in the Info pane then shows the file.
    viewer.reapply_raw_parameters({"width": 997, "height": 1, "dtype": np.uint8,
                                   "color_format": "Grayscale"})
    assert len(dialogs) == 1
    assert viewer.active_label is viewer.image_label
    assert viewer.image_label.original_data.shape == (1, 997)
    assert viewer._raw_recovery_path is None
