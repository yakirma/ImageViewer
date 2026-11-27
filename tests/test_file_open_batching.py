"""Regression tests (I15): macOS Finder "Open With" of several files must
open them together (one open_files() call -> montage) in one existing window
instead of one window per file. The per-window app-wide event filter used to
swallow every FileOpen event, so ImageViewerApp's batching never ran.

QFileOpenEvent cannot be constructed from PyQt6, so the batching logic is
exercised through FileOpenBatcher directly, and the filter with a stand-in
event object."""
from __future__ import annotations

import numpy as np
from PIL import Image
from PyQt6.QtCore import QEvent
from PyQt6.QtWidgets import QApplication

import main as main_mod
from image_viewer import ImageViewer


def _pngs(tmp_path, n):
    paths = []
    for i in range(n):
        p = tmp_path / f"img{i}.png"
        Image.fromarray(np.full((16, 16), i * 40, np.uint8)).save(p)
        paths.append(str(p))
    return paths


def _viewers():
    return {w for w in QApplication.topLevelWidgets() if isinstance(w, ImageViewer)}


class _FakeFileOpen:
    def __init__(self, path):
        self._path = path

    def type(self):
        return QEvent.Type.FileOpen

    def file(self):
        return self._path


def test_window_event_filter_does_not_consume_file_open(viewer, tmp_path):
    window_list = [viewer]
    viewer.window_list = window_list
    before = _viewers()
    path = _pngs(tmp_path, 1)[0]

    handled = viewer.eventFilter(QApplication.instance(), _FakeFileOpen(path))

    # Must propagate to ImageViewerApp.event, which does the batching.
    assert handled is False
    assert window_list == [viewer]
    assert _viewers() <= before  # no new window was created
    assert not viewer.current_file_path


def _only_visible(viewer):
    # Leftover windows from earlier tests must not be picked as the target.
    for w in _viewers():
        if w is not viewer:
            w.hide()
    viewer.show()


def test_three_files_batched_into_one_open_files_call(viewer, qtbot, tmp_path, monkeypatch):
    _only_visible(viewer)
    calls = []
    monkeypatch.setattr(viewer, "open_files", lambda files: calls.append(list(files)))
    before = _viewers()

    batcher = main_mod.FileOpenBatcher(delay_ms=20)
    paths = _pngs(tmp_path, 3)
    for p in paths:
        batcher.add(p)

    qtbot.waitUntil(lambda: bool(calls), timeout=2000)
    qtbot.wait(100)  # no further (split) deliveries
    assert calls == [paths]
    assert batcher.file_buffer == []
    assert _viewers() <= before  # no new window was created


def test_three_files_build_a_montage(viewer, qtbot, tmp_path):
    _only_visible(viewer)
    batcher = main_mod.FileOpenBatcher(delay_ms=20)
    paths = _pngs(tmp_path, 3)
    for p in paths:
        batcher.add(p)

    qtbot.waitUntil(
        lambda: viewer.stacked_widget.currentWidget() is viewer.montage_widget,
        timeout=3000)
    assert [label.file_path for label in viewer.montage_labels] == paths


def test_cold_start_files_delivered_once_window_exists(qtbot, tmp_path, monkeypatch):
    target = {"w": None}
    monkeypatch.setattr(main_mod.FileOpenBatcher, "_target_window",
                        staticmethod(lambda: target["w"]))
    batcher = main_mod.FileOpenBatcher(delay_ms=10)
    batcher.RETRY_MS = 20
    paths = _pngs(tmp_path, 2)
    for p in paths:
        batcher.add(p)

    qtbot.wait(120)  # several flush attempts while no window exists
    assert batcher.file_buffer == paths

    calls = []
    v = ImageViewer(window_list=[])
    qtbot.addWidget(v)
    monkeypatch.setattr(v, "open_files", lambda files: calls.append(list(files)))
    target["w"] = v

    qtbot.waitUntil(lambda: bool(calls), timeout=2000)
    assert calls == [paths]
    assert batcher.file_buffer == []
