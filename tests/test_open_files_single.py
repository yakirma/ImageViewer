"""Regression tests (I04): open_files() -- used by the CLI, Finder/FileOpen
events and the single-instance hand-off -- must route a single path through
the full open_file() pipeline instead of building a 1-image montage. The
montage path never set up video playback and skipped the raw-resolution
guessing, so videos showed a still frame and raw files failed silently."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


def _write_png(path, value):
    Image.fromarray(np.full((16, 16), value, np.uint8)).save(path)
    return str(path)


def test_single_video_plays_in_main_view(viewer, tmp_path):
    cv2 = pytest.importorskip("cv2")
    vp = str(tmp_path / "clip.avi")
    w = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    if not w.isOpened():
        pytest.skip("MJPG writer unavailable")
    for i in range(10):
        w.write(np.full((48, 64, 3), i * 20, np.uint8))
    w.release()

    viewer.show()
    viewer.open_files([vp])

    assert viewer.stacked_widget.currentWidget() is not viewer.montage_widget
    assert viewer.image_handler.is_video
    assert not viewer.video_toolbar.isHidden()


def test_single_raw_without_resolution_loads_with_guess(viewer, tmp_path):
    # 3072 bytes -> the 4:3 guess in open_file() is exactly 64x48 uint8.
    p = tmp_path / "nores.bin"
    np.arange(64 * 48, dtype=np.uint32).astype(np.uint8).tofile(p)

    viewer.open_files([str(p)])

    assert viewer.stacked_widget.currentWidget() is viewer.image_display_container
    assert viewer.active_label is viewer.image_label
    assert viewer.image_label.original_data is not None
    assert viewer.image_label.original_data.shape[:2] == (48, 64)


def test_single_image_opens_in_main_view(viewer, tmp_path):
    p = _write_png(tmp_path / "a.png", 10)

    viewer.open_files([p])

    assert viewer.stacked_widget.currentWidget() is viewer.image_display_container
    assert viewer.current_file_path == p


def test_two_files_still_open_as_montage(viewer, tmp_path):
    a = _write_png(tmp_path / "a.png", 10)
    b = _write_png(tmp_path / "b.png", 200)

    viewer.open_files([a, b])

    assert viewer.stacked_widget.currentWidget() is viewer.montage_widget
    assert len(viewer.montage_labels) == 2
