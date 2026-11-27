"""Regression tests for I45 (NPZ key switch updates color_format / Info pane)
and I46 (flow max_flow ignores unknown 1e9 / non-finite vectors)."""
from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtGui import QImage

from utils import flow_magnitude, flow_to_color


def _rendered_rgb(label):
    img = label.current_pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    buf = np.frombuffer(img.constBits().asstring(bpl * h), np.uint8).reshape(h, bpl)
    return buf[:, : w * 3].reshape(h, w, 3).copy()


def _select_key(viewer, key):
    idx = viewer.channel_combo.findText(key)
    assert idx != -1, key
    viewer.channel_combo.setCurrentIndex(idx)


# ---- I45 -----------------------------------------------------------------

def test_npz_key_switch_updates_color_format_and_info_pane(viewer, tmp_path):
    p = str(tmp_path / "multi.npz")
    rng = np.random.default_rng(0)
    np.savez(p,
             gray=rng.random((6, 8)).astype(np.float32),
             rgb=(rng.random((6, 8, 3)) * 255).astype(np.uint8),
             flow=rng.standard_normal((6, 8, 2)).astype(np.float32))
    viewer.open_file(p)
    h = viewer.image_handler
    combo = viewer.info_pane.color_format_combo
    assert h.color_format == "Grayscale"
    assert combo.currentText() == "Grayscale"

    for key, fmt, dtype in (("rgb", "RGB", "uint8"),
                            ("flow", "RGB (Flow)", "float32"),
                            ("gray", "Grayscale", "float32")):
        _select_key(viewer, key)
        assert h.current_npz_key == key
        assert h.color_format == fmt
        assert np.dtype(h.dtype).name == dtype
        assert viewer.image_label.metadata["color_format"] == fmt
        assert combo.currentText() == fmt
        assert viewer.info_pane.dtype_combo.currentText() == dtype


# ---- I46 -----------------------------------------------------------------

def _flow_with_unknowns(h=12, w=16):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    flow = np.stack([xx - w / 2, yy - h / 2], axis=-1).astype(np.float32)
    flow[0, 0] = (1e10, 0.0)
    flow[1, 3] = (0.0, -2e9)
    flow[5, 5] = (np.nan, 1.0)
    flow[7, 2] = (np.inf, np.inf)
    return flow


def test_flow_magnitude_masks_unknowns():
    flow = _flow_with_unknowns()
    mag = flow_magnitude(flow)
    assert np.isnan(mag[0, 0]) and np.isnan(mag[1, 3])
    assert np.isnan(mag[5, 5]) and np.isnan(mag[7, 2])
    assert np.isclose(mag[2, 2], np.hypot(flow[2, 2, 0], flow[2, 2, 1]))


def test_npz_flow_with_unknowns_not_washed_out(viewer, tmp_path):
    flow = _flow_with_unknowns()
    p = str(tmp_path / "flow.npz")
    np.savez(p, flow=flow)
    viewer.open_file(p)

    label = viewer.image_label
    assert label.colormap == "flow"
    valid = np.isfinite(flow).all(axis=2) & (np.abs(flow) <= 1e9).all(axis=2)
    true_max = float(np.hypot(flow[valid][:, 0], flow[valid][:, 1]).max())
    assert label.contrast_limits is not None
    assert label.contrast_limits[1] == pytest.approx(true_max, rel=1e-5)

    expected = flow_to_color(flow, max_flow=true_max)
    rendered = _rendered_rgb(label)
    assert rendered.shape == expected.shape
    diff = np.abs(rendered.astype(int) - expected.astype(int))
    assert diff[valid].max() <= 1
    assert (rendered[~valid] == 0).all()
    # Not washed out: valid pixels are not all near-white.
    assert rendered[valid].min() < 200

    # Histogram presets (min-max) stay in valid-magnitude space too.
    viewer._apply_histogram_preset(0, 100)
    assert label.contrast_limits[1] == pytest.approx(true_max, rel=1e-5)


def test_histogram_limits_for_flow_ignore_unknowns(viewer, qtbot, tmp_path):
    flow = _flow_with_unknowns()
    p = str(tmp_path / "flow.npz")
    np.savez(p, flow=flow)
    viewer.open_file(p)
    viewer.histogram_window.show()
    viewer.image_label.contrast_limits = None
    viewer.update_histogram_data(new_image=True)
    hist = viewer.histogram_window
    assert hist.data is not None
    assert np.isfinite(hist.data).all()
    assert hist.data.max() < 1e3  # magnitudes, unknowns dropped
    hist.set_region_to_percentile(0, 100)
    lo, hi = hist.region.getRegion()
    assert hi < 1e3


def test_flo_file_with_unknowns_renders_like_flow_to_color(viewer, tmp_path):
    flow = _flow_with_unknowns()
    h, w = flow.shape[:2]
    p = str(tmp_path / "f.flo")
    with open(p, "wb") as f:
        np.array([202021.25], np.float32).tofile(f)
        np.array([w, h], np.int32).tofile(f)
        flow.astype(np.float32).tofile(f)
    viewer.open_file(p)
    valid = np.isfinite(flow).all(axis=2) & (np.abs(flow) <= 1e9).all(axis=2)
    true_max = float(np.hypot(flow[valid][:, 0], flow[valid][:, 1]).max())
    expected = flow_to_color(flow, max_flow=true_max)
    rendered = _rendered_rgb(viewer.image_label)
    diff = np.abs(rendered.astype(int) - expected.astype(int))
    assert diff[valid].max() <= 1
