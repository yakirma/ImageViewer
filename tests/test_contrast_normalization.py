"""Regression tests for I13 (contrast limits wider than the data were ignored:
the image was always renormalized to its own min/max) and I14 (non-uint8 RGB
wrapped modulo 256 when no contrast limits were set, e.g. in montage).

Rendered values are read from the label's current_pixmap (the colormapped
display image), so they reflect exactly what is painted.
"""
from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtGui import QImage

from widgets import ZoomableDraggableLabel
from tests.conftest import inject_data


def _rgb(label):
    img = label.current_pixmap.toImage().convertToFormat(QImage.Format.Format_RGB32)
    buf = np.frombuffer(img.constBits().asstring(img.sizeInBytes()), np.uint8)
    arr = buf.reshape(img.height(), img.bytesPerLine() // 4, 4)[:, : img.width()]
    return arr[..., [2, 1, 0]].astype(int)


def _gray(label):
    return _rgb(label)[..., 0]


@pytest.fixture
def make_label(qtbot):
    def _make(data, limits=None, colormap=None):
        lab = ZoomableDraggableLabel()
        qtbot.addWidget(lab)
        lab.resize(200, 200)
        lab.set_data(data)
        if colormap:
            lab.set_colormap(colormap)
        if limits is not None:
            lab.set_contrast_limits(*limits)
        return lab
    return _make


# --------------------------------------------------------------------- I13

def test_wider_limits_keep_their_meaning(make_label):
    data = np.array([[100.0, 200.0]], dtype=np.float32)
    g = _gray(make_label(data, limits=(0.0, 255.0)))
    # 100/255 and 200/255 of full range -> mid greys, not 0 and 255.
    assert abs(g[0, 0] - 100) <= 1
    assert abs(g[0, 1] - 200) <= 1


def test_limits_inside_data_range_still_stretch(make_label):
    data = np.array([[0.0, 50.0, 100.0, 150.0, 200.0]], dtype=np.float32)
    g = _gray(make_label(data, limits=(50.0, 150.0)))
    assert g[0].tolist()[0] == 0 and g[0].tolist()[1] == 0
    assert abs(g[0, 2] - 127) <= 1
    assert g[0, 3] == 255 and g[0, 4] == 255


def test_no_limits_stretches_to_data_range(make_label):
    data = np.array([[10.0, 15.0, 20.0]], dtype=np.float32)
    g = _gray(make_label(data))
    assert g[0, 0] == 0 and g[0, 2] == 255
    assert abs(g[0, 1] - 127) <= 1


def test_shared_limits_give_common_scale(make_label):
    """Two montage images with the same limits render the same value alike."""
    a = make_label(np.array([[10.0, 20.0]], dtype=np.float32), limits=(0.0, 40.0))
    b = make_label(np.array([[20.0, 40.0]], dtype=np.float32), limits=(0.0, 40.0))
    assert _gray(a)[0, 1] == _gray(b)[0, 0]
    assert abs(_gray(a)[0, 1] - 127) <= 1


def test_non_finite_values(make_label):
    data = np.array([[0.0, 10.0, np.inf, -np.inf, np.nan]], dtype=np.float32)
    for limits in (None, (0.0, 10.0)):
        g = _gray(make_label(data, limits=limits))
        assert g[0, 0] == 0 and g[0, 1] == 255
        assert g[0, 2] == 255, "+inf should map to the top of the range"
        assert g[0, 3] == 0 and g[0, 4] == 0


def test_wider_limits_with_colormap(make_label):
    import matplotlib
    data = np.array([[100.0, 200.0]], dtype=np.float32)
    rgb = _rgb(make_label(data, limits=(0.0, 400.0), colormap="viridis"))
    exp = (np.array(matplotlib.colormaps["viridis"]([0.25, 0.5]))[:, :3] * 255).astype(int)
    assert np.abs(rgb[0] - exp).max() <= 1


def test_degenerate_limits_do_not_crash(make_label):
    data = np.array([[1.0, 2.0]], dtype=np.float32)
    g = _gray(make_label(data, limits=(5.0, 5.0)))
    assert g.shape == (1, 2)


# --------------------------------------------------------------------- I14

def test_uint16_rgb_no_limits_does_not_wrap(make_label):
    data = np.zeros((1, 3, 3), dtype=np.uint16)
    data[0, 0] = 769          # wrapped to 1 before the fix
    data[0, 1] = 65535
    data[0, 2] = 32768
    rgb = _rgb(make_label(data))
    # Stretched by the global finite min..max (769..65535), no wrap-around.
    assert rgb[0, 0].tolist() == [0, 0, 0]
    assert rgb[0, 1].tolist() == [255, 255, 255]
    assert abs(rgb[0, 2, 0] - 126) <= 1  # (32768-769)/(65535-769)*255


def test_12bit_in_uint16_rgb_montage_uses_full_range(make_label):
    """12-bit data stored in uint16 must not render nearly black."""
    data = np.zeros((1, 3, 3), dtype=np.uint16)
    data[0, 1] = 2048
    data[0, 2] = 4095
    rgb = _rgb(make_label(data))
    assert rgb[0, 0, 0] == 0
    assert abs(rgb[0, 1, 0] - 127) <= 1
    assert rgb[0, 2].tolist() == [255, 255, 255]


def test_float_rgb_0_to_3_montage_is_stretched(make_label):
    data = np.zeros((1, 3, 3), dtype=np.float32)
    data[0, 1] = 1.5
    data[0, 2] = 3.0
    rgb = _rgb(make_label(data))
    assert rgb[0, 0, 0] == 0
    assert abs(rgb[0, 1, 0] - 127) <= 1
    assert rgb[0, 2].tolist() == [255, 255, 255]


def test_rgb_stretch_is_global_across_channels(make_label):
    data = np.zeros((1, 1, 3), dtype=np.float32)
    data[0, 0] = [0.0, 2.0, 4.0]
    rgb = _rgb(make_label(data))
    assert rgb[0, 0, 0] == 0
    assert abs(rgb[0, 0, 1] - 127) <= 1
    assert rgb[0, 0, 2] == 255


def test_float_rgb_above_one_no_limits_stretches(make_label):
    data = np.zeros((1, 3, 3), dtype=np.float32)
    data[0, 0] = 0.0
    data[0, 1] = 500.0
    data[0, 2] = 1000.0
    rgb = _rgb(make_label(data))
    assert rgb[0, 0, 0] == 0
    assert abs(rgb[0, 1, 0] - 127) <= 1
    assert rgb[0, 2, 0] == 255


def test_float_rgb_unit_range_and_uint8_unchanged(make_label):
    f = np.full((1, 1, 3), 0.5, dtype=np.float32)
    assert abs(_rgb(make_label(f))[0, 0, 0] - 127) <= 1
    u = np.array([[[10, 20, 30]]], dtype=np.uint8)
    assert _rgb(make_label(u))[0, 0].tolist() == [10, 20, 30]
    # Integer data already inside [0, 255] passes through as-is.
    i = np.array([[[10, 20, 30]]], dtype=np.uint16)
    assert _rgb(make_label(i))[0, 0].tolist() == [10, 20, 30]


def test_uint16_rgba_no_limits(make_label):
    data = np.zeros((1, 2, 4), dtype=np.uint16)
    data[0, :, 3] = 65535
    data[0, 1, :3] = 65535
    lab = make_label(data)
    img = lab.current_pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    px = img.pixelColor(1, 0)
    assert (px.red(), px.green(), px.blue(), px.alpha()) == (255, 255, 255, 255)
    assert img.pixelColor(0, 0).alpha() == 255


# ------------------------------------------------------ histogram/contrast UI

def test_histogram_region_reset_and_auto_contrast(viewer, qtbot):
    data = np.tile(np.linspace(100.0, 200.0, 64, dtype=np.float32), (8, 1))
    inject_data(viewer, data)
    viewer._update_active_view()  # wires histogram region -> contrast limits
    label = viewer.image_label

    # Dragging the region to [0, 255] -> data renders mid-grey.
    viewer.histogram_window.region.setRegion([0.0, 255.0])
    qtbot.waitUntil(lambda: label.contrast_limits is not None
                    and abs(label.contrast_limits[1] - 255.0) < 1e-3, timeout=2000)
    g = _gray(label)
    assert abs(g[0, 0] - 100) <= 1 and abs(g[0, -1] - 200) <= 1

    # Auto contrast (0-100% preset) -> limits are the data range -> full stretch.
    viewer._apply_histogram_preset(0, 100)
    g = _gray(label)
    assert g[0, 0] == 0 and g[0, -1] == 255

    # Narrow the region, then full reset: limits go back to none or to the
    # data range (the histogram re-emits its reset region) -> full stretch.
    label.set_contrast_limits(140.0, 160.0)
    viewer.reset_image_full()
    lim = label.contrast_limits
    assert lim is None or (abs(lim[0] - 100.0) < 1e-3 and abs(lim[1] - 200.0) < 1e-3)
    g = _gray(label)
    assert g[0, 0] == 0 and g[0, -1] == 255


# ----------------------------------------------- channel / NPZ-key switching

def test_npz_key_switch_refits_contrast(viewer, qtbot, tmp_path):
    """Switching NPZ keys must not reuse the previous key's absolute limits
    (big 0..4095 -> small 0..0.5 used to render all black)."""
    npz = str(tmp_path / "keys.npz")
    ramp = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    np.savez(npz, big=np.tile(ramp * 4095.0, (8, 1)), small=np.tile(ramp * 0.5, (8, 1)))
    viewer.open_file(npz)
    viewer.channel_combo.setCurrentText("big")
    g = _gray(viewer.image_label)
    assert g[0, 0] == 0 and g[0, -1] == 255

    viewer.channel_combo.setCurrentText("small")
    assert viewer.image_handler.current_npz_key == "small"
    lim = viewer.image_label.contrast_limits
    assert lim is None or (abs(lim[0]) < 1e-6 and abs(lim[1] - 0.5) < 1e-6)
    g = _gray(viewer.image_label)
    assert g[0, 0] == 0 and g[0, -1] == 255
    assert abs(g[0, 32] - 129) <= 2

    viewer.channel_combo.setCurrentText("big")
    g = _gray(viewer.image_label)
    assert g[0, 0] == 0 and g[0, -1] == 255


def test_rgb_to_single_channel_switch_refits_contrast(viewer, qtbot, tmp_path):
    """RGB -> B must not reuse the luminance limits (B rendered flat)."""
    from PIL import Image
    rgb = np.zeros((8, 64, 3), np.uint8)
    rgb[..., 0] = 200
    rgb[..., 2] = np.linspace(10, 50, 64).astype(np.uint8)
    png = str(tmp_path / "rgb.png")
    Image.fromarray(rgb).save(png)
    viewer.open_file(png)

    viewer.channel_combo.setCurrentText("B")
    g = _gray(viewer.image_label)
    assert g[0, 0] == 0 and g[0, -1] == 255
    assert len(np.unique(g[0])) > 10

    viewer.channel_combo.setCurrentText("R")  # constant channel: no crash
    viewer.channel_combo.setCurrentText("RGB")
    out = _rgb(viewer.image_label)
    assert out[0, 0, 0] > out[0, 0, 1]


def _scaled_npz(tmp_path):
    npz = str(tmp_path / "multi.npz")
    ramp = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    # 'big' is the first key, so it is loaded (and contrast-fitted) first.
    np.savez(npz, big=np.tile(ramp * 4095.0, (8, 1)), small=np.tile(ramp * 0.5, (8, 1)))
    return npz


def _assert_small_fitted(viewer):
    assert viewer.image_handler.current_npz_key == "small"
    lim = viewer.image_label.contrast_limits
    assert lim is None or (abs(lim[0]) < 1e-6 and abs(lim[1] - 0.5) < 1e-6)
    g = _gray(viewer.image_label)
    assert g[0, 0] == 0 and g[0, -1] == 255
    assert abs(g[0, 32] - 129) <= 2


def test_open_npz_hash_non_first_key_fresh(viewer, tmp_path):
    npz = _scaled_npz(tmp_path)
    viewer.open_file(npz + "#small")
    _assert_small_fitted(viewer)
    # Histogram region follows the re-fitted limits.
    rmin, rmax = viewer.histogram_window.region.getRegion()
    assert abs(rmax - 0.5) < 1e-3


def test_open_npz_hash_key_while_other_key_displayed(viewer, tmp_path):
    npz = _scaled_npz(tmp_path)
    viewer.open_file(npz + "#big")
    g = _gray(viewer.image_label)
    assert g[0, 0] == 0 and g[0, -1] == 255
    viewer.open_file(npz + "#small")
    _assert_small_fitted(viewer)


def test_open_npz_hash_key_maintain_view_state_percentiles(viewer, tmp_path):
    npz = _scaled_npz(tmp_path)
    state = {"zoom_scale": 1.0, "colormap": "gray", "contrast_percentiles": (0.0, 50.0)}
    viewer.open_file(npz + "#small", maintain_view_state=state)
    lo, hi = viewer.image_label.contrast_limits
    assert abs(lo) < 1e-6 and abs(hi - 0.25) < 0.01  # percentiles of 'small'
    g = _gray(viewer.image_label)
    assert g[0, 0] == 0 and g[0, -1] == 255 and abs(g[0, 16] - 129) <= 6
