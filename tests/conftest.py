"""Shared pytest fixtures for ImageViewer UI tests.

Tests run headless (`QT_QPA_PLATFORM=offscreen`). The `viewer` fixture
constructs a real `ImageViewer` window with side-effects neutralized
(no network update check, no recent-files restore, no single-instance
listener) so tests can focus on UI/logic state-sync invariants without
depending on the host environment.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Force headless Qt before any PyQt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Ensure the project root is importable as a flat module layout.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _isolate_side_effects(monkeypatch):
    """Neutralize ImageViewer side-effects that would touch the network,
    the user's settings file, or the file system."""
    import image_viewer as iv_mod

    # Block the update checker from making a real GitHub API call.
    monkeypatch.setattr(iv_mod.CheckForUpdates, "run", lambda self: None, raising=True)
    monkeypatch.setattr(iv_mod.CheckForUpdates, "start", lambda self, *a, **kw: None, raising=True)

    # Avoid mutating the real QSettings during tests.
    from PyQt6.QtCore import QSettings
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(ROOT / ".pytest_qsettings"))


@pytest.fixture
def viewer(qtbot, monkeypatch):
    """A fresh ImageViewer with synthetic data already loaded.

    The histogram window is shown (so update paths actually run) and
    `Use Visible Area` is disabled so tests don't depend on widget geometry.
    """
    from image_viewer import ImageViewer

    v = ImageViewer(window_list=[])
    qtbot.addWidget(v)

    # Show the histogram so update_histogram_data() doesn't early-return.
    v.histogram_window.show()
    v.histogram_window.use_visible_checkbox.setChecked(False)

    return v


@pytest.fixture
def loaded_viewer(viewer):
    """Viewer with deterministic synthetic data already injected as the active label."""
    data = make_gradient(64, 64, lo=0.0, hi=100.0)
    inject_data(viewer, data)
    return viewer


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def make_gradient(h: int, w: int, lo: float = 0.0, hi: float = 255.0) -> np.ndarray:
    """A 2D float32 gradient — predictable values for assertions."""
    row = np.linspace(lo, hi, w, dtype=np.float32)
    return np.tile(row, (h, 1))


def make_random(h: int, w: int, lo: float = 0.0, hi: float = 100.0, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((h, w)) * (hi - lo) + lo).astype(np.float32)


def inject_data(viewer, data: np.ndarray) -> None:
    """Set up the viewer's active_label as if a single-image file had been loaded.

    Bypasses the file pipeline so tests don't need fixtures on disk.
    """
    label = viewer.image_label
    label.set_data(data, reset_view=True, is_pristine=True)
    viewer.active_label = label
    # Mirror what the load pipeline does for histogram dependencies.
    viewer.image_handler.original_image_data = data
    viewer.histogram_action.setEnabled(True)
    viewer.math_transform_action.setEnabled(True)
