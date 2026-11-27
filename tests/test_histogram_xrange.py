"""I44 (user request): the histogram x-axis follows the contrast limits the
image uses, with ~5% padding, and the histogram is re-binned over that range.
Manual region drags don't rescale mid-drag; the axis refits on release.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from tests.conftest import inject_data, make_random

PAD = 0.05


def _xrange(v):
    return tuple(v.histogram_window.plot_widget.getViewBox().viewRange()[0])


def _expected(lo, hi):
    pad = (hi - lo) * PAD
    return lo - pad, hi + pad


def _assert_fits(v, lo, hi):
    got = _xrange(v)
    lo, hi = float(lo), float(hi)
    exp = _expected(lo, hi)
    tol = (hi - lo) * 1e-3 + 1e-9
    assert got[0] == pytest.approx(exp[0], abs=tol)
    assert got[1] == pytest.approx(exp[1], abs=tol)


@pytest.fixture
def rand_viewer(viewer):
    inject_data(viewer, make_random(64, 64, lo=0.0, hi=1000.0))
    viewer._update_active_view(reset_histogram=True)
    return viewer


def test_percentile_preset_fits_axis(rand_viewer):
    v = rand_viewer
    data = v.active_label.original_data
    v._apply_histogram_preset(9, 95)
    lo, hi = np.percentile(data, 9), np.percentile(data, 95)
    assert v.active_label.contrast_limits == pytest.approx((lo, hi))
    _assert_fits(v, lo, hi)


def test_full_range_preset_shows_full_data(rand_viewer):
    v = rand_viewer
    v._apply_histogram_preset(9, 95)
    v._apply_histogram_preset(0, 100)
    data = v.active_label.original_data
    _assert_fits(v, float(data.min()), float(data.max()))


def test_reset_shows_full_data(rand_viewer):
    v = rand_viewer
    v._apply_histogram_preset(20, 30)
    v.reset_image_full()
    data = v.active_label.original_data
    _assert_fits(v, float(data.min()), float(data.max()))


def test_histogram_rebinned_over_visible_range(rand_viewer):
    v = rand_viewer
    v._apply_histogram_preset(40, 60)
    lo, hi = _xrange(v)
    hw = v.histogram_window
    assert len(hw.histograms) == 1
    x, _y = hw.histograms[0].getOriginalDataset()
    # stepMode="center": x holds the bin edges.
    assert x[0] == pytest.approx(lo) and x[-1] == pytest.approx(hi)
    assert len(x) == hw.N_BINS + 1


def test_preset_does_not_feed_back(rand_viewer, qtbot):
    v = rand_viewer
    hw = v.histogram_window
    qtbot.wait(300)  # let any reset-emitted region update settle
    emitted = []
    hw.region_changed.connect(lambda a, b: emitted.append((a, b)))
    v._apply_histogram_preset(9, 95)
    limits = v.active_label.contrast_limits
    qtbot.wait(300)  # longer than the 200 ms region debounce
    assert emitted == []
    assert v.active_label.contrast_limits == limits


def test_drag_does_not_rescale_until_finished(rand_viewer):
    v = rand_viewer
    hw = v.histogram_window
    v._apply_histogram_preset(40, 60)
    before = _xrange(v)
    lo, _hi = hw.region.getRegion()
    # User drags the upper edge far right: sigRegionChanged fires repeatedly.
    for new_hi in (700.0, 800.0, 900.0):
        hw.region.lines[1].setValue(new_hi)
    assert _xrange(v) == pytest.approx(before)
    # Release: sigRegionChangeFinished -> refit.
    hw.region.lines[1].sigPositionChangeFinished.emit(hw.region.lines[1])
    _assert_fits(v, lo, 900.0)


def test_typed_values_fit_axis(rand_viewer):
    v = rand_viewer
    hw = v.histogram_window
    hw.min_val_input.setText("100")
    hw.max_val_input.setText("300")
    hw._apply_custom_values()
    _assert_fits(v, 100.0, 300.0)


def test_widget_buttons_fit_axis(rand_viewer):
    v = rand_viewer
    hw = v.histogram_window
    hw.p5_95_btn.click()
    lo, hi = np.percentile(hw.data, 5), np.percentile(hw.data, 95)
    _assert_fits(v, lo, hi)


def test_pasted_limits_followed(rand_viewer):
    """Limits set outside the histogram (paste / keep-view-state) are
    followed even in full-image mode, where update_histogram_data skips."""
    v = rand_viewer
    v.active_label.set_contrast_limits(200.0, 250.0)
    v._update_active_view(reset_histogram=False)
    assert v.histogram_window.region.getRegion() == pytest.approx((200.0, 250.0))
    _assert_fits(v, 200.0, 250.0)


def test_rgb(viewer):
    rng = np.random.default_rng(0)
    data = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
    inject_data(viewer, data)
    viewer._update_active_view(reset_histogram=True)
    viewer._apply_histogram_preset(10, 90)
    lum = np.dot(data.astype(np.float64), [0.2989, 0.5870, 0.1140])
    lo, hi = np.percentile(lum, 10), np.percentile(lum, 90)
    assert len(viewer.histogram_window.histograms) == 3
    _assert_fits(viewer, lo, hi)


def test_hidden_histogram_then_shown(rand_viewer):
    v = rand_viewer
    hw = v.histogram_window
    hw.hide()
    v._apply_histogram_preset(9, 95)  # must not crash while hidden
    lo, hi = v.active_label.contrast_limits
    v.toggle_histogram_window()  # show
    assert hw.isVisible()
    _assert_fits(v, lo, hi)


def test_constant_image_no_crash(viewer):
    inject_data(viewer, np.full((8, 8), 5.0, dtype=np.float32))
    viewer._update_active_view(reset_histogram=True)
    viewer._apply_histogram_preset(0, 100)
    lo, hi = _xrange(viewer)
    assert np.isfinite(lo) and np.isfinite(hi) and lo < 5.0 < hi


def _png(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path)
    return path


def test_montage_follows_active_label(viewer, tmp_path):
    a = np.tile(np.arange(0, 200, 2, dtype=np.uint8), (10, 1))
    b = np.tile(np.arange(50, 250, 2, dtype=np.uint8), (10, 1))
    paths = [_png(str(tmp_path / "a.png"), a), _png(str(tmp_path / "b.png"), b)]
    viewer.display_montage(paths)
    la, lb = viewer.montage_labels
    la.set_contrast_limits(20.0, 60.0)
    lb.set_contrast_limits(100.0, 200.0)
    viewer._set_active_montage_label(la)
    _assert_fits(viewer, 20.0, 60.0)
    viewer._set_active_montage_label(lb)
    _assert_fits(viewer, 100.0, 200.0)


# ------------------------------------------------- review follow-ups (I44)

def _bar_edges(v):
    x, _y = v.histogram_window.histograms[0].getOriginalDataset()
    return float(x[0]), float(x[-1])


def test_zoom_out_rebins_full_distribution(rand_viewer, qtbot):
    """After a narrow preset, zooming the plot out must show bars over the
    whole data range (re-binned on the debounced x-range change)."""
    v = rand_viewer
    data = v.active_label.original_data
    v._apply_histogram_preset(40, 60)
    assert _bar_edges(v)[1] - _bar_edges(v)[0] < 400
    v.histogram_window.plot_widget.getViewBox().setXRange(-100, 1100, padding=0)
    qtbot.waitUntil(lambda: _bar_edges(v)[0] <= float(data.min()) + 1e-3, timeout=1000)
    lo, hi = _bar_edges(v)
    # Clamped to the data range, not the (wider) view.
    assert lo == pytest.approx(float(data.min()), abs=1e-3)
    assert hi == pytest.approx(float(data.max()), abs=1e-3)


def test_view_all_shows_full_data_range(rand_viewer, qtbot):
    v = rand_viewer
    data = v.active_label.original_data
    v._apply_histogram_preset(40, 60)
    vb = v.histogram_window.plot_widget.getViewBox()
    vb.autoRange(padding=0)
    x0, x1 = vb.viewRange()[0]
    assert x0 <= float(data.min()) + 1e-3 and x1 >= float(data.max()) - 1e-3
    qtbot.waitUntil(lambda: _bar_edges(v)[0] <= float(data.min()) + 1e-3, timeout=1000)


def test_fitter_does_not_trigger_extra_rebin(rand_viewer):
    v = rand_viewer
    v._apply_histogram_preset(40, 60)
    assert not v.histogram_window._rebin_timer.isActive()


def test_float_inf_values_give_finite_region(viewer):
    data = np.linspace(-50.0, 50.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    data[0, 0] = np.inf
    data[0, 1] = -np.inf
    data[0, 2] = np.nan
    inject_data(viewer, data)
    viewer.active_label.contrast_limits = None
    viewer.update_histogram_data(new_image=True)
    finite = data[np.isfinite(data)]
    region = viewer.histogram_window.region.getRegion()
    assert region == pytest.approx((float(finite.min()), float(finite.max())), abs=1e-3)
    lo, hi = _xrange(viewer)
    assert np.isfinite(lo) and np.isfinite(hi)


def test_npz_key_switch_rebins_bars(viewer, tmp_path, qtbot):
    """Full-image mode: switching NPZ key / channel must re-bin the bars from
    the new array, not just move the axis over the old bars."""
    path = str(tmp_path / "multi.npz")
    rng = np.random.default_rng(1)
    np.savez(path, a=(rng.random((20, 20)) * 10).astype(np.float32),
             b=(500 + rng.random((20, 20)) * 500).astype(np.float32))
    viewer.open_file(path)
    combo = viewer.channel_combo
    assert combo.findText("b") >= 0
    combo.setCurrentIndex(combo.findText("b"))
    b = np.load(path)["b"]
    lo, hi = _bar_edges(viewer)
    assert lo >= float(b.min()) - 1e-3 and hi <= float(b.max()) + 1e-3
    hw = viewer.histogram_window
    assert hw.data.min() >= float(b.min()) - 1e-3  # histogram data is key b
    _x, y = hw.histograms[0].getOriginalDataset()
    assert y.sum() == b.size  # all of b's pixels land in the bars


def test_data_changed_rebins_in_full_image_mode(rand_viewer):
    """update_image_display (channel switch, video frame) re-bins even when
    'Use Visible Area' is off."""
    v = rand_viewer
    new = make_random(64, 64, lo=5000.0, hi=6000.0, seed=7)
    v.active_label.set_data(new, reset_view=False, is_pristine=True)
    v.update_histogram_data(data_changed=True)
    assert v.histogram_window.data.min() >= 5000.0


def test_large_rgb_sampled_compactly(viewer):
    hw = viewer.histogram_window
    data = np.random.default_rng(0).integers(0, 256, (3000, 1000, 3), dtype=np.uint8)
    data[0, 0] = (0, 0, 0)
    data[-1, -1] = (255, 255, 255)
    hw.update_histogram(data, limits=(10.0, 200.0))
    assert len(hw._channels) == 3
    for values, _brush in hw._channels:
        assert values.dtype != np.float64
        assert values.size <= hw.MAX_SAMPLES
    assert hw.data.size <= hw.MAX_SAMPLES
    # Exact data range is kept for View All even when the sample misses it.
    assert hw._data_bounds() == (0.0, 255.0)
