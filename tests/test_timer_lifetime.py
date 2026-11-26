"""Regression tests (I39): delayed QTimer callbacks (single-shot lambdas,
debounce timers) must not fire after an ImageViewer window -- or the widgets
it owns -- has been closed and deleted. Previously such callbacks captured the
deleted objects and intermittently segfaulted later tests."""
from __future__ import annotations

import numpy as np
from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent, QTimer
from PyQt6.QtWidgets import QApplication

from image_viewer import ImageViewer
from widgets import HistogramWidget, SharedViewState, ZoomableDraggableLabel


def _flush_deletes():
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    QApplication.processEvents()


def _owned_timers(obj):
    return [t for t in obj.findChildren(QTimer)]


def test_debounce_timers_are_owned_by_their_widgets(qtbot):
    """Every debounce timer must be a Qt child of its owner so it is destroyed
    (and can never fire) together with that owner."""
    state = SharedViewState()
    assert state._interaction_timer.parent() is state

    hist = HistogramWidget()
    qtbot.addWidget(hist)
    assert hist._region_update_timer.parent() is hist

    label = ZoomableDraggableLabel()
    qtbot.addWidget(label)
    assert label._view_update_timer.parent() is label
    assert label._single_interaction_timer.parent() is label


def test_pending_timers_do_not_fire_after_window_deleted(qtbot, tmp_path):
    npz = str(tmp_path / "multi.npz")
    np.savez(npz, a=np.random.rand(32, 40).astype(np.float32),
             b=np.random.rand(32, 40, 3).astype(np.float32))

    window_list = []
    v = ImageViewer(window_list=window_list)
    v.show()
    v.histogram_window.show()
    v.open_file(npz)

    # Schedule every kind of deferred callback the window uses.
    v._set_dock_visibility_preserving_window(v.info_pane, not v.info_pane.isVisible())
    v._histogram_debounce_timer.start()
    v.histogram_window._region_update_timer.start()
    v.image_label._view_update_timer.start()
    v.image_label._single_interaction_timer.start()
    v.shared_state._interaction_timer.start()
    v.resize(v.width() + 50, v.height())

    label = v.image_label
    hist = v.histogram_window
    v.close()
    v.deleteLater()
    hist.deleteLater()
    _flush_deletes()
    assert sip.isdeleted(v)
    assert sip.isdeleted(label)

    # Run the event loop well past every timer delay (max is 1500 ms for the
    # update check). Any callback touching the deleted window would crash or
    # raise here.
    qtbot.wait(1700)

    # The deferred window-list cleanup still runs even though the window was
    # deleted right after closing.
    assert v not in window_list
