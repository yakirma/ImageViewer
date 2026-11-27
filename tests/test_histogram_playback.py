"""I44 follow-up: during video playback the histogram re-bin is coalesced
(<= ~4 Hz, latest frame) instead of running on every frame; pause, single
step and slider release refresh it immediately."""
from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import inject_data, make_random


@pytest.fixture
def video_viewer(viewer, monkeypatch):
    v = viewer
    inject_data(v, make_random(48, 64, lo=0.0, hi=100.0))
    v._update_active_view(reset_histogram=True)
    h = v.image_handler
    h.is_video = True
    h.video_frame_count = 1000
    h.current_frame_index = 0

    def _frame(index):
        h.current_frame_index = index
        # Each frame has its own value range so stale bars are detectable.
        h.original_image_data = make_random(48, 64, lo=index * 10.0,
                                            hi=index * 10.0 + 5.0, seed=index)
        return True

    monkeypatch.setattr(h, "get_next_frame", lambda: _frame(h.current_frame_index + 1))
    monkeypatch.setattr(h, "seek_frame", lambda i: _frame(max(0, i)))

    calls = []
    orig = v.histogram_window.update_histogram

    def counting(*a, **k):
        calls.append(h.current_frame_index)
        return orig(*a, **k)

    monkeypatch.setattr(v.histogram_window, "update_histogram", counting)
    v._rebin_calls = calls
    yield v
    v.playback_timer.stop()
    v._playback_histogram_timer.stop()


def _start_playback(v):
    # Active but never fires on its own: the test drives the frames.
    v.playback_timer.start(60_000)


def _hist_min(v):
    return float(v.histogram_window.data.min())


def test_playback_coalesces_rebins_and_pause_refreshes(video_viewer, qtbot):
    v = video_viewer
    calls = v._rebin_calls
    _start_playback(v)

    for _ in range(30):  # ~30 frames over ~0.6 s
        v._video_timer_timeout()
        qtbot.wait(20)
    during = len(calls)
    assert 1 <= during <= 4, calls  # throttled (~250 ms), not 30

    v._on_play_pause(False)  # pause
    assert len(calls) == during + 1  # exactly one immediate refresh
    assert calls[-1] == v.image_handler.current_frame_index
    assert _hist_min(v) >= v.image_handler.current_frame_index * 10.0 - 1e-3
    assert not v._playback_histogram_timer.isActive()
    qtbot.wait(300)
    assert len(calls) == during + 1  # nothing pending fires afterwards


def test_burst_without_waiting_defers_to_latest_frame(video_viewer, qtbot):
    v = video_viewer
    calls = v._rebin_calls
    _start_playback(v)
    for _ in range(10):
        v._video_timer_timeout()
    assert calls == []
    qtbot.waitUntil(lambda: len(calls) == 1, timeout=1000)
    assert calls == [10]  # latest frame only


def test_single_step_refreshes_immediately(video_viewer):
    v = video_viewer
    calls = v._rebin_calls
    v._next_frame()
    assert calls == [1]
    v._prev_frame()
    assert calls == [1, 0]


def test_slider_release_refreshes_immediately_even_when_resuming(video_viewer):
    v = video_viewer
    calls = v._rebin_calls
    _start_playback(v)
    v._on_slider_pressed()
    v.video_slider.blockSignals(True)
    v.video_slider.setRange(0, 999)
    v.video_slider.setValue(42)
    v.video_slider.blockSignals(False)
    v._on_slider_released()
    assert calls == [42]
    assert v.playback_timer.isActive()  # playback resumed


def test_hidden_histogram_playback_does_no_work(video_viewer, qtbot):
    v = video_viewer
    v.histogram_window.hide()
    _start_playback(v)
    for _ in range(5):
        v._video_timer_timeout()
    assert not v._playback_histogram_timer.isActive()
    qtbot.wait(300)
    assert v._rebin_calls == []
