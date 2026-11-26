"""Regression tests (I02): opening a non-NPZ file after a multi-key NPZ must
not keep the NPZ keys/arrays around (they hijacked the Channel combo and
replaced the newly opened image)."""
from __future__ import annotations

import numpy as np
from PIL import Image

from image_handler import ImageHandler


def _files(tmp_path):
    npz = str(tmp_path / "multi.npz")
    np.savez(npz, a=np.zeros((5, 6)), b=np.ones((5, 6, 3)))
    png = str(tmp_path / "rgb.png")
    rgb = np.zeros((10, 12, 3), np.uint8)
    rgb[..., 0] = 255
    Image.fromarray(rgb).save(png)
    return npz, png


def test_handler_clears_npz_state_on_non_npz_load(tmp_path):
    npz, png = _files(tmp_path)
    h = ImageHandler()
    h.load_image(npz)
    old = h.npz_data
    assert set(h.npz_keys) == {"a", "b"}

    h.load_image(png)
    assert h.npz_keys == {}
    assert h.npz_data is None
    assert h.current_npz_key is None
    assert h.npz_file_path is None
    assert old.fid is None  # NpzFile handle was closed
    assert h.original_image_data.shape == (10, 12, 3)


def test_handler_reload_npz_still_works(tmp_path):
    npz, _ = _files(tmp_path)
    h = ImageHandler()
    h.load_image(npz)
    h.load_image(npz)
    assert set(h.npz_keys) == {"a", "b"}
    assert h.npz_data["b"].shape == (5, 6, 3)


def test_viewer_png_after_npz_shows_png(viewer, qtbot, tmp_path):
    npz, png = _files(tmp_path)
    viewer.open_file(npz)
    viewer.open_file(png)
    combo = [viewer.channel_combo.itemText(i) for i in range(viewer.channel_combo.count())]
    assert "a" not in combo and "b" not in combo
    assert viewer.image_handler.original_image_data.shape == (10, 12, 3)
    assert viewer.image_label.original_data.shape[:2] == (10, 12)
