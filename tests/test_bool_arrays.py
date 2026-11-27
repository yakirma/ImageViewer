"""Regression tests (I47): boolean arrays (e.g. NPZ mask keys) must be
presented as uint8 0/1 so contrast presets (np.percentile) and thumbnail
generation don't fail with numpy-bool subtraction TypeErrors."""
from __future__ import annotations

import numpy as np

from image_handler import ImageHandler


def _bool_npz(tmp_path, bool_first=False):
    path = str(tmp_path / "masks.npz")
    mask = np.zeros((8, 10), dtype=bool)
    mask[2:5, 3:7] = True
    img = np.linspace(0, 1, 80, dtype=np.float32).reshape(8, 10)
    if bool_first:
        np.savez(path, mask=mask, img=img)
    else:
        np.savez(path, img=img, mask=mask)
    return path, mask


def test_bool_npz_key_loads_as_uint8(tmp_path):
    path, mask = _bool_npz(tmp_path, bool_first=True)
    h = ImageHandler()
    h.load_image(path)
    assert h.current_npz_key == "mask"
    assert h.npz_keys == {"mask": True, "img": True}
    data = h.original_image_data
    assert data.dtype == np.uint8
    assert h.dtype == np.uint8
    np.testing.assert_array_equal(data, mask.astype(np.uint8))
    assert set(np.unique(data).tolist()) == {0, 1}
    # Key switching goes through npz_data[key]: must also be uint8.
    assert h.npz_data["mask"].dtype == np.uint8
    assert h.npz_data["img"].dtype == np.float32
    # Proxy still exposes the NpzFile API.
    assert list(h.npz_data.files) == ["mask", "img"]


def test_bool_math_transform_result_is_uint8():
    h = ImageHandler()
    h.original_image_data = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = h.apply_math_transform("x > 5")
    assert out.dtype == np.uint8
    assert int(out.sum()) == 6


def test_viewer_switch_to_bool_key_and_presets(viewer, qtbot, tmp_path):
    path, _mask = _bool_npz(tmp_path)
    viewer.open_file(path)
    combo = viewer.channel_combo
    items = [combo.itemText(i) for i in range(combo.count())]
    assert "mask" in items
    # Switching to the bool key used to raise in _apply_histogram_preset.
    combo.setCurrentIndex(items.index("mask"))
    qtbot.wait(10)
    assert viewer.image_handler.current_npz_key == "mask"
    assert viewer.image_handler.original_image_data.dtype == np.uint8
    viewer._apply_histogram_preset(5, 95)
    viewer._apply_histogram_preset(0, 100)
    # Thumbnail pane builds pixmaps for non-current keys from npz_data[key].
    combo.setCurrentIndex(items.index("img"))
    qtbot.wait(10)
    pixmap = viewer.thumbnail_pane._generate_pixmap_from_array(
        viewer.image_handler.npz_data["mask"])
    assert pixmap is not None and not pixmap.isNull()
