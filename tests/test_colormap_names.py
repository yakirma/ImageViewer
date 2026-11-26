"""Regression tests for I07: selecting 'spectral' raised ValueError/KeyError
(matplotlib's name is 'Spectral') and left the label stuck on the bad name,
so every later redraw raised again."""
from __future__ import annotations

import matplotlib

from widgets import resolve_colormap_name


def _combo_items(viewer):
    c = viewer.colormap_combo
    return [c.itemText(i) for i in range(c.count())]


def test_every_offered_colormap_exists_in_matplotlib(viewer):
    missing = [n for n in _combo_items(viewer)
               if n != "flow" and n not in matplotlib.colormaps]
    assert missing == []


def test_every_offered_colormap_applies(loaded_viewer):
    v = loaded_viewer
    for name in _combo_items(v):
        if name == "flow":
            continue
        v.colormap_combo.setCurrentText(name)
        assert v.active_label.colormap == name
        assert v.colormap_combo.currentText() == name


def test_spectral_selection_renders(loaded_viewer):
    v = loaded_viewer
    v.colormap_combo.setCurrentText("Spectral")
    assert v.active_label.colormap == "Spectral"
    # A later redraw (e.g. contrast drag) must not raise.
    v.active_label.set_contrast_limits(10.0, 90.0)


def test_lowercase_name_is_normalized():
    assert resolve_colormap_name("spectral") == "Spectral"
    assert resolve_colormap_name("gray") == "gray"
    assert resolve_colormap_name("flow") == "flow"


def test_unknown_colormap_falls_back_to_gray(loaded_viewer):
    v = loaded_viewer
    v.colormap_combo.setCurrentText("viridis")
    # e.g. a stale name from a saved view state
    v.set_colormap("no_such_cmap")
    assert v.active_label.colormap == "gray"
    assert v.colormap_combo.currentText() == "gray"
    # The label is not stuck: further redraws work.
    v.active_label.set_contrast_limits(0.0, 50.0)
    v.active_label.set_view_state({"colormap": "also_bogus"})
    assert v.active_label.colormap == "gray"
