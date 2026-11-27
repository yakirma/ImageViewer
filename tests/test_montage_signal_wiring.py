"""Regression tests for I16: switching the active montage label must not
stack slot connections on region_changed / hover_moved / view_changed.

Before the fix all disconnects sat in one try block; in montage mode the
first one raised TypeError and nothing after it ran, so every tile click
added another histogram region_changed -> set_contrast_limits connection
(and more hover/view connections on the labels).
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image


def _png(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.full((6, 8), value, dtype=np.uint8)).save(path)
    return path


def _counts(v):
    hw = v.histogram_window
    region = hw.receivers(hw.region_changed)
    per_label = [
        (lbl.receivers(lbl.hover_moved),
         lbl.receivers(lbl.view_changed),
         lbl.receivers(lbl.zoom_factor_changed))
        for lbl in v.montage_labels
    ]
    return region, per_label


def _montage(viewer, tmp_path, n=3):
    paths = [_png(str(tmp_path / f"img{i}.png"), 40 * (i + 1)) for i in range(n)]
    viewer.display_montage(paths)
    assert len(viewer.montage_labels) == n
    return paths


def test_receiver_counts_stable_across_active_switches(viewer, tmp_path):
    _montage(viewer, tmp_path)
    baseline = _counts(viewer)

    for _ in range(5):
        for lbl in viewer.montage_labels:
            viewer._set_active_montage_label(lbl)

    assert _counts(viewer)[0] == baseline[0] == 1
    # Each label: hover (creation-time only) = 1; view/zoom wired only when
    # active. Compare per label ignoring which one is currently active.
    region, per_label = _counts(viewer)
    active = viewer.active_label
    for lbl, (hover, view, zoom) in zip(viewer.montage_labels, per_label):
        assert hover == 1
        expected = 1 if lbl is active else 0
        assert view == expected
        assert zoom == expected


def test_single_region_change_calls_slot_once(viewer, tmp_path, monkeypatch):
    _montage(viewer, tmp_path)
    for _ in range(4):
        for lbl in viewer.montage_labels:
            viewer._set_active_montage_label(lbl)

    active = viewer.active_label
    calls = []
    monkeypatch.setattr(active, "set_contrast_limits",
                        lambda lo, hi: calls.append((lo, hi)))

    viewer.histogram_window.region_changed.emit(10.0, 20.0)
    assert calls == [(10.0, 20.0)]


def test_image_label_unwired_in_montage_and_rewired_on_return(viewer, tmp_path):
    from tests.conftest import inject_data, make_gradient

    il = viewer.image_label
    inject_data(viewer, make_gradient(16, 16))
    viewer._update_active_view()
    assert il.receivers(il.hover_moved) == 1
    assert il.receivers(il.view_changed) == 1

    _montage(viewer, tmp_path)
    viewer._set_active_montage_label(viewer.montage_labels[1])
    # Montage mode: the single-image label must not keep active-view slots.
    assert il.receivers(il.hover_moved) == 0
    assert il.receivers(il.view_changed) == 0
    assert il.receivers(il.zoom_factor_changed) == 0

    # Rebuilding the montage deletes the old labels; switching must not
    # raise and must not stack connections.
    _montage(viewer, tmp_path, n=2)
    for _ in range(3):
        for lbl in viewer.montage_labels:
            viewer._set_active_montage_label(lbl)
    hw = viewer.histogram_window
    assert hw.receivers(hw.region_changed) == 1
