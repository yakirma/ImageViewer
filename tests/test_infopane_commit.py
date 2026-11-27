"""Regression tests for I11: InfoPane raw-parameter edits.

Typing into the Width/Data Type fields used to reload the raw file on every
keystroke, and each intermediate size that didn't match the file popped a
modal "Error applying parameters" dialog. Parameters must now apply only on
commit (Enter / focus out / arrow step / list pick), and a failed combination
is reported inline and in the status bar rather than modally.
"""
from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox


@pytest.fixture
def dialogs(monkeypatch):
    rec = []
    rec_fn = lambda *a, **k: rec.append(a[2] if len(a) > 2 else a)
    monkeypatch.setattr(QMessageBox, "warning", rec_fn)
    monkeypatch.setattr(QMessageBox, "critical", rec_fn)
    return rec


@pytest.fixture
def raw_viewer(viewer, qtbot, tmp_path):
    viewer.show()
    qtbot.waitExposed(viewer)
    p = tmp_path / "frame_1920x1080.raw"
    np.zeros(1920 * 1080, np.uint8).tofile(p)
    viewer.open_file(str(p))
    assert viewer.image_handler.original_image_data.shape == (1080, 1920)
    calls = []
    orig = viewer.reapply_raw_parameters
    viewer.info_pane.settings_changed.disconnect(viewer.reapply_raw_parameters)

    def spy(s):
        calls.append(dict(s))
        orig(s)
    viewer.info_pane.settings_changed.connect(spy)
    viewer._reapply_calls = calls
    return viewer


def _type_width(qtbot, pane, text):
    sb = pane.width_spinbox
    sb.setFocus()
    sb.lineEdit().selectAll()
    qtbot.keyClicks(sb.lineEdit(), text)


def test_typing_width_does_not_reload_per_keystroke(raw_viewer, qtbot, dialogs):
    pane = raw_viewer.info_pane
    _type_width(qtbot, pane, "960")

    assert raw_viewer._reapply_calls == []
    assert not dialogs
    assert raw_viewer.image_handler.original_image_data.shape == (1080, 1920)

    qtbot.keyClick(pane.width_spinbox.lineEdit(), Qt.Key.Key_Return)

    assert len(raw_viewer._reapply_calls) == 1
    # Height auto-recalculated from the committed width.
    assert pane.height_spinbox.value() == 2160
    assert raw_viewer._reapply_calls[0]["width"] == 960
    assert raw_viewer._reapply_calls[0]["height"] == 2160
    assert raw_viewer.image_handler.original_image_data.shape == (2160, 960)
    assert not dialogs


def test_invalid_committed_width_reports_inline_not_modal(raw_viewer, qtbot, dialogs):
    pane = raw_viewer.info_pane
    _type_width(qtbot, pane, "1921")
    assert raw_viewer._reapply_calls == []
    qtbot.keyClick(pane.width_spinbox.lineEdit(), Qt.Key.Key_Return)

    assert len(raw_viewer._reapply_calls) == 1
    assert not dialogs
    assert pane.error_label.isVisibleTo(pane)
    assert "Error applying parameters" in pane.error_label.text()
    assert "Error applying parameters" in raw_viewer.status_bar.currentMessage()

    # A valid commit afterwards clears the inline error.
    _type_width(qtbot, pane, "1920")
    qtbot.keyClick(pane.width_spinbox.lineEdit(), Qt.Key.Key_Return)
    assert not pane.error_label.isVisibleTo(pane)
    assert raw_viewer.image_handler.original_image_data.shape == (1080, 1920)
    assert not dialogs


def test_arrow_step_applies_immediately(raw_viewer, qtbot, dialogs):
    pane = raw_viewer.info_pane
    pane.width_spinbox.setFocus()
    qtbot.keyClick(pane.width_spinbox, Qt.Key.Key_Down)
    assert len(raw_viewer._reapply_calls) == 1
    assert raw_viewer._reapply_calls[0]["width"] == 1919
    assert not dialogs


def test_focus_out_without_change_does_not_reload(raw_viewer, qtbot):
    pane = raw_viewer.info_pane
    pane.dtype_combo.lineEdit().editingFinished.emit()
    pane.width_spinbox.editingFinished.emit()
    assert raw_viewer._reapply_calls == []


def test_typing_dtype_applies_once_on_enter(raw_viewer, qtbot, dialogs):
    pane = raw_viewer.info_pane
    le = pane.dtype_combo.lineEdit()
    le.setFocus()
    le.selectAll()
    qtbot.keyClicks(le, "uint16")
    assert raw_viewer._reapply_calls == []
    assert not dialogs

    qtbot.keyClick(le, Qt.Key.Key_Return)

    assert len(raw_viewer._reapply_calls) == 1
    assert raw_viewer._reapply_calls[0]["dtype"] == "uint16"
    # Height recalculated for the 2-byte dtype at width 1920.
    assert pane.height_spinbox.value() == 540
    assert raw_viewer.image_handler.original_image_data.shape == (540, 1920)
    assert not dialogs


def test_picking_dtype_from_list_applies(raw_viewer, qtbot, dialogs):
    pane = raw_viewer.info_pane
    idx = pane.dtype_combo.findText("uint16")
    pane.dtype_combo.setCurrentIndex(idx)
    pane.dtype_combo.activated.emit(idx)  # what a user pick from the popup emits
    assert len(raw_viewer._reapply_calls) == 1
    assert raw_viewer.image_handler.original_image_data.shape == (540, 1920)
    assert not dialogs
