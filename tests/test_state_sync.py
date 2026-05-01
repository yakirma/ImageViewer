"""UI/logic state-sync regression tests.

Catches a class of bugs where derived UI state (histogram, contrast limits,
inspection data, 3D viewer, info pane) drifts out of sync with the underlying
image data after operations like math transforms, channel changes, or restores.

Motivating bug (v1.1.1): applying ``np.log(x)`` updated the displayed image
and recomputed the histogram bars, but left ``contrast_limits``, the histogram
region overlay, and the min/max input boxes anchored to the pre-transform
data range. From the user's perspective, "the histogram still shows values
for the original image."

The fix forces ``contrast_limits = None`` whenever the data scale changes
(math transform, restore_original). These tests pin the invariants down so
the same regression cannot reappear silently.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import make_gradient, make_random, inject_data


# ----------------------------------------------------------------------
# Math transform → contrast / histogram sync
# ----------------------------------------------------------------------

class TestMathTransformContrastSync:
    """The original bug class: applying a math transform must invalidate
    any prior contrast window because the data scale changed."""

    def test_math_transform_clears_contrast_limits(self, loaded_viewer):
        v = loaded_viewer
        # User had previously applied a contrast window on [0, 100] data.
        v.active_label.contrast_limits = (10.0, 80.0)

        v.apply_math_transform("np.log(x + 1)")

        assert v.active_label.contrast_limits is None, (
            "contrast_limits must be reset after a math transform — values from "
            "the previous data scale don't map to the new one."
        )

    def test_math_transform_updates_original_data(self, loaded_viewer):
        v = loaded_viewer
        before = v.active_label.original_data.copy()

        v.apply_math_transform("x + 5")

        after = v.active_label.original_data
        np.testing.assert_allclose(after, before + 5, rtol=1e-5)

    def test_math_transform_preserves_pristine_data(self, loaded_viewer):
        v = loaded_viewer
        pristine_before = v.active_label.pristine_data.copy()

        v.apply_math_transform("x * 2 + 1")

        np.testing.assert_array_equal(
            v.active_label.pristine_data, pristine_before,
            err_msg="pristine_data must remain the original — it's the source of "
                    "truth for restore_original and re-applying transforms.",
        )

    def test_histogram_bars_reflect_post_transform_data(self, loaded_viewer):
        v = loaded_viewer
        v.apply_math_transform("x + 1000")

        hist_data = v.histogram_window.data
        assert hist_data is not None
        # Underlying input was [0, 100]; after +1000 the histogram's stored
        # 1D distribution must lie in the shifted range.
        assert hist_data.min() >= 1000 - 1e-3
        assert hist_data.max() <= 1100 + 1e-3

    def test_histogram_region_resets_to_new_data_range(self, loaded_viewer):
        v = loaded_viewer
        # Stale contrast from a previous interaction, in the OLD data scale.
        v.active_label.contrast_limits = (10.0, 80.0)
        v.histogram_window.region.setRegion([10.0, 80.0])

        v.apply_math_transform("x + 1000")

        rmin, rmax = v.histogram_window.region.getRegion()
        # Region must have moved out of the old [10, 80] window.
        assert rmin > 80.0, f"histogram region min {rmin} is still in the pre-transform range"
        assert rmax > 80.0, f"histogram region max {rmax} is still in the pre-transform range"

    def test_min_max_inputs_reflect_post_transform_range(self, loaded_viewer):
        v = loaded_viewer
        v.active_label.contrast_limits = (10.0, 80.0)

        v.apply_math_transform("x + 1000")

        min_v = float(v.histogram_window.min_val_input.text())
        max_v = float(v.histogram_window.max_val_input.text())
        assert min_v > 80.0, f"min input shows pre-transform value: {min_v}"
        assert max_v > 80.0, f"max input shows pre-transform value: {max_v}"

    def test_histogram_xrange_adapts_to_post_transform_data(self, loaded_viewer):
        v = loaded_viewer
        v.apply_math_transform("x + 1000")

        x_min, x_max = v.histogram_window.plot_widget.viewRange()[0]
        # The plot's X axis must include the new data range, not be stuck near 0–100.
        assert x_max > 500, f"plot X axis didn't adapt to new data: max={x_max}"


# ----------------------------------------------------------------------
# Restore original
# ----------------------------------------------------------------------

class TestRestoreOriginalSync:
    """Restoring pristine data after a math transform must also reset
    state that was meaningful only on the transformed scale."""

    def test_restore_original_clears_contrast_limits(self, loaded_viewer):
        v = loaded_viewer
        v.apply_math_transform("np.log(x + 1)")
        # User dragged contrast on the log-scale data.
        v.active_label.contrast_limits = (1.0, 4.0)

        v.restore_original_image()

        assert v.active_label.contrast_limits is None

    def test_restore_original_returns_to_pristine_data(self, loaded_viewer):
        v = loaded_viewer
        pristine = v.active_label.pristine_data.copy()
        v.apply_math_transform("np.log(x + 1)")

        v.restore_original_image()

        np.testing.assert_array_equal(v.active_label.original_data, pristine)

    def test_restore_clears_current_math_expression(self, loaded_viewer):
        v = loaded_viewer
        v.apply_math_transform("x + 5")
        assert v.current_math_expression is not None

        v.restore_original_image()

        assert v.current_math_expression is None


# ----------------------------------------------------------------------
# Error paths must not corrupt state
# ----------------------------------------------------------------------

class TestMathTransformErrorPaths:
    def test_invalid_expression_keeps_data(self, loaded_viewer):
        v = loaded_viewer
        before = v.active_label.original_data.copy()

        v.apply_math_transform("this_is_not_valid_python(((")

        np.testing.assert_array_equal(v.active_label.original_data, before)

    def test_invalid_expression_surfaces_error_message(self, loaded_viewer):
        v = loaded_viewer
        v.apply_math_transform("this_is_not_valid_python(((")

        # MathTransformPane records the error so the user sees it.
        # Existence of the error_label and that it isn't empty is enough —
        # we don't pin the message text.
        err_widget = getattr(v.math_transform_pane, "error_label", None)
        if err_widget is not None:
            assert err_widget.text() != "", "expected non-empty error feedback"


# ----------------------------------------------------------------------
# Histogram presets
# ----------------------------------------------------------------------

class TestHistogramPresets:
    """Pressing M / Shift+M must compute fresh contrast limits from the
    current data, not use cached values from a previous image."""

    def test_min_max_preset_sets_contrast_to_full_range(self, loaded_viewer):
        v = loaded_viewer
        # Pretend stale contrast from before.
        v.active_label.contrast_limits = (10.0, 80.0)

        v._apply_histogram_preset(0, 100, use_visible_only=False)

        lo, hi = v.active_label.contrast_limits
        # Data is gradient 0..100; full-range preset should hit those.
        assert lo == pytest.approx(0.0, abs=0.5)
        assert hi == pytest.approx(100.0, abs=0.5)

    def test_min_max_preset_after_transform_uses_post_transform_range(self, loaded_viewer):
        v = loaded_viewer
        v.apply_math_transform("x + 1000")

        v._apply_histogram_preset(0, 100, use_visible_only=False)

        lo, hi = v.active_label.contrast_limits
        assert lo == pytest.approx(1000.0, abs=0.5)
        assert hi == pytest.approx(1100.0, abs=0.5)


# ----------------------------------------------------------------------
# set_data invariants — ZoomableDraggableLabel
# ----------------------------------------------------------------------

class TestSetDataInvariants:
    def test_set_data_with_reset_view_clears_contrast_limits(self, loaded_viewer):
        label = loaded_viewer.active_label
        label.contrast_limits = (10.0, 80.0)

        label.set_data(make_gradient(32, 32), reset_view=True)

        assert label.contrast_limits is None

    def test_set_data_without_reset_view_preserves_contrast_limits(self, loaded_viewer):
        """This is the inverse — set_data with reset_view=False is the
        primitive that the math-transform fix layered ``contrast_limits = None``
        on top of. Pinning this behavior guards against an over-broad fix
        in the lower layer that would break video frame stepping."""
        label = loaded_viewer.active_label
        label.contrast_limits = (10.0, 80.0)

        label.set_data(make_gradient(32, 32), reset_view=False)

        assert label.contrast_limits == (10.0, 80.0)

    def test_set_data_is_pristine_updates_both_buffers(self, loaded_viewer):
        label = loaded_viewer.active_label
        new_data = make_random(16, 16)

        label.set_data(new_data, is_pristine=True, reset_view=True)

        np.testing.assert_array_equal(label.original_data, new_data)
        np.testing.assert_array_equal(label.pristine_data, new_data)

    def test_set_data_non_pristine_leaves_pristine_intact(self, loaded_viewer):
        label = loaded_viewer.active_label
        pristine = label.pristine_data.copy()
        new_data = make_random(16, 16)

        label.set_data(new_data, is_pristine=False, reset_view=False)

        np.testing.assert_array_equal(label.original_data, new_data)
        np.testing.assert_array_equal(label.pristine_data, pristine)


# ----------------------------------------------------------------------
# Histogram debounce / visibility gating
# ----------------------------------------------------------------------

class TestHistogramVisibilityGating:
    def test_hidden_histogram_skips_recompute(self, loaded_viewer):
        v = loaded_viewer
        v.histogram_window.hide()
        # Set a sentinel so we can detect whether update_histogram ran.
        v.histogram_window.data = "SENTINEL"

        v.update_histogram_data(new_image=True)

        assert v.histogram_window.data == "SENTINEL", (
            "update_histogram_data must early-return when the histogram window "
            "is hidden — it's a documented performance path."
        )

    def test_visible_histogram_recomputes(self, loaded_viewer):
        v = loaded_viewer
        # Histogram is already shown by the fixture.
        v.histogram_window.data = "SENTINEL"

        v.update_histogram_data(new_image=True)

        # The recompute path replaces the sentinel string with a numpy array
        # of finite samples drawn from the current image.
        assert isinstance(v.histogram_window.data, np.ndarray)
        assert v.histogram_window.data.size > 0


# ----------------------------------------------------------------------
# Round-trip: transform then restore
# ----------------------------------------------------------------------

class TestRoundTrip:
    def test_transform_then_restore_reaches_clean_state(self, loaded_viewer):
        v = loaded_viewer
        original = v.active_label.pristine_data.copy()

        v.apply_math_transform("np.log(x + 1)")
        v.active_label.contrast_limits = (1.0, 4.0)
        v.restore_original_image()

        np.testing.assert_array_equal(v.active_label.original_data, original)
        assert v.active_label.contrast_limits is None
        assert v.current_math_expression is None
