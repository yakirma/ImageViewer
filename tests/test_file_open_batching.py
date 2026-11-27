"""Regression tests (I15): macOS Finder "Open With" of several files must
open them together (one open_files() call -> montage) in ONE window, and
must never replace the image the user is viewing: an empty window is
reused, otherwise a new window is created (intentional since 6504ea5,
"Fix Open With to spawn new windows").

The per-window app-wide event filter used to swallow every FileOpen event
and open one window per file, so ImageViewerApp's batching never ran.

QFileOpenEvent cannot be constructed from PyQt6, so the batching logic is
exercised through FileOpenBatcher directly, and the filter with a stand-in
event object."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from PyQt6.QtCore import QEvent
from PyQt6.QtWidgets import QApplication

import main as main_mod
from image_viewer import ImageViewer


def _pngs(tmp_path, n, prefix="img"):
    paths = []
    for i in range(n):
        p = tmp_path / f"{prefix}{i}.png"
        Image.fromarray(np.full((16, 16), i * 40, np.uint8)).save(p)
        paths.append(str(p))
    return paths


class _FakeFileOpen:
    def __init__(self, path):
        self._path = path

    def type(self):
        return QEvent.Type.FileOpen

    def file(self):
        return self._path


@pytest.fixture
def windows(qtbot, monkeypatch):
    """A window list plus a factory; records open_files() calls per window
    and registers every ImageViewer created during the test for cleanup."""
    window_list = []
    calls = []
    created = []
    real_init = ImageViewer.__init__

    def tracking_init(self, *a, **k):
        real_init(self, *a, **k)
        created.append(self)
        qtbot.addWidget(self)

    monkeypatch.setattr(ImageViewer, "__init__", tracking_init)

    real_open_files = ImageViewer.open_files

    def recording_open_files(self, files):
        calls.append((self, list(files)))
        return real_open_files(self, files)

    monkeypatch.setattr(ImageViewer, "open_files", recording_open_files)

    def make(show=True):
        w = ImageViewer(window_list=window_list)
        if show:
            w.show()
        return w

    yield window_list, make, calls, created
    for w in created:
        try:
            w.close()
        except RuntimeError:
            pass


def _occupy(win, tmp_path, name):
    path = _pngs(tmp_path, 1, prefix=name)[0]
    win.open_file(path)
    assert win.current_file_path == path
    return path


def _batcher(window_list, **kw):
    return main_mod.FileOpenBatcher(window_list=window_list, **kw)


def test_window_event_filter_does_not_consume_file_open(windows, tmp_path):
    window_list, make, calls, created = windows
    win = make()
    path = _pngs(tmp_path, 1)[0]

    handled = win.eventFilter(QApplication.instance(), _FakeFileOpen(path))

    # Must propagate to ImageViewerApp.event, which does the batching.
    assert handled is False
    assert window_list == [win]
    assert calls == []
    assert not win.current_file_path


def test_batch_reuses_empty_window_with_one_open_files_call(windows, qtbot, tmp_path):
    window_list, make, calls, created = windows
    win = make()
    batcher = _batcher(window_list, delay_ms=20)
    batcher.last_active = win
    paths = _pngs(tmp_path, 3)
    for p in paths:
        batcher.add(p)

    qtbot.waitUntil(lambda: bool(calls), timeout=2000)
    qtbot.wait(100)  # no further (split) deliveries
    assert calls == [(win, paths)]
    assert batcher.file_buffer == []
    assert window_list == [win]
    assert win.stacked_widget.currentWidget() is win.montage_widget
    assert [label.file_path for label in win.montage_labels] == paths


def test_occupied_active_window_gets_new_window_for_batch(windows, qtbot, tmp_path):
    window_list, make, calls, created = windows
    win = make()
    shown = _occupy(win, tmp_path, "shown")
    calls.clear()
    batcher = _batcher(window_list, delay_ms=20)
    batcher.last_active = win
    paths = _pngs(tmp_path, 3)
    for p in paths:
        batcher.add(p)

    qtbot.waitUntil(lambda: bool(calls), timeout=2000)
    qtbot.wait(100)
    assert len(calls) == 1
    target, files = calls[0]
    assert target is not win and files == paths
    assert target in window_list and target.isVisible()
    assert win.current_file_path == shown  # user's image untouched


def test_single_file_into_occupied_window_opens_new_window(windows, tmp_path):
    window_list, make, calls, created = windows
    win = make()
    shown = _occupy(win, tmp_path, "shown")
    calls.clear()
    batcher = _batcher(window_list)
    batcher.last_active = win
    path = _pngs(tmp_path, 1, prefix="new")[0]

    target = batcher.open_now([path])

    assert target is not win and target.isVisible()
    assert calls == [(target, [path])]
    assert target.current_file_path == path
    assert win.current_file_path == shown


def test_empty_window_reused_when_active_one_is_occupied(windows, tmp_path):
    window_list, make, calls, created = windows
    busy = make()
    _occupy(busy, tmp_path, "shown")
    empty = make()
    calls.clear()
    batcher = _batcher(window_list)
    batcher.last_active = busy
    paths = _pngs(tmp_path, 2)

    target = batcher.open_now(paths)

    assert target is empty
    assert calls == [(empty, paths)]
    assert window_list == [busy, empty]


def test_hidden_or_closed_windows_are_never_targeted(windows, qtbot, tmp_path):
    window_list, make, calls, created = windows
    hidden = make(show=False)
    closed = make()
    closed.close()
    batcher = _batcher(window_list)
    batcher.last_active = closed  # stale "most recently active" window
    path = _pngs(tmp_path, 1)[0]

    target = batcher.open_now([path])

    assert target is not hidden and target is not closed
    assert target.isVisible()
    assert calls == [(target, [path])]
    assert not hidden.current_file_path and not closed.current_file_path


def test_window_activation_is_tracked(windows):
    window_list, make, calls, created = windows
    a = make()
    b = make()
    batcher = _batcher(window_list)
    QApplication.sendEvent(b, QEvent(QEvent.Type.WindowActivate))
    assert batcher.last_active is b
    QApplication.sendEvent(a, QEvent(QEvent.Type.WindowActivate))
    assert batcher.last_active is a


def test_cold_start_files_delivered_once_attached(windows, qtbot, tmp_path):
    window_list, make, calls, created = windows
    batcher = main_mod.FileOpenBatcher(delay_ms=10)  # not attached yet
    batcher.RETRY_MS = 20
    paths = _pngs(tmp_path, 2)
    for p in paths:
        batcher.add(p)

    qtbot.wait(120)  # several flush attempts before the first window exists
    assert batcher.file_buffer == paths
    assert calls == [] and created == []

    first = make()
    batcher.attach(window_list)

    qtbot.waitUntil(lambda: bool(calls), timeout=2000)
    assert calls == [(first, paths)]
    assert batcher.file_buffer == []
    assert window_list == [first]
