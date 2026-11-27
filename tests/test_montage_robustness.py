"""Regression tests for montage / NPZ-path robustness.

I24: a montage where every file fails must not leave an empty montage page
     (pressing 'C' there raised IndexError), and load failures must be
     reported instead of silently dropped.
I40: open_files() must accept NPZ key paths ("file.npz#key").
I38: the gallery key for an NPZ opened via a keyed path must not be
     double-keyed ("x.npz#a#b").
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt


def _write_png(path, value):
    Image.fromarray(np.full((16, 16), value, np.uint8)).save(path)
    return str(path)


def _write_bad(path):
    # Not a decodable image of any kind.
    path.write_bytes(b"\x00\x01garbage")
    return str(path)


def _write_npz(path):
    np.savez(path, a=np.zeros((8, 9)), b=np.ones((8, 9)))
    return str(path)


def test_all_failed_montage_does_not_show_empty_page(viewer, qtbot, tmp_path):
    bad1 = _write_bad(tmp_path / "bad1.png")
    bad2 = _write_bad(tmp_path / "bad2.png")
    before = viewer.stacked_widget.currentWidget()

    viewer.display_montage([bad1, bad2])

    assert viewer.montage_labels == []
    assert viewer.stacked_widget.currentWidget() is before
    assert viewer.stacked_widget.currentWidget() is not viewer.montage_widget
    msg = viewer.status_bar.currentMessage()
    assert "bad1.png" in msg and "bad2.png" in msg


def test_previous_montage_preserved_when_all_new_files_fail(viewer, qtbot, tmp_path):
    npz = _write_npz(tmp_path / "m.npz")
    png = _write_png(tmp_path / "p.png", 10)
    viewer.show()
    viewer.open_file(npz)
    viewer.open_files([f"{npz}#a", png])
    labels_before = list(viewer.montage_labels)
    active_before = viewer.active_label
    path_before = viewer.current_file_path
    assert len(labels_before) == 2

    viewer.open_files([_write_bad(tmp_path / "bad1.png"), _write_bad(tmp_path / "bad2.png")])

    assert viewer.stacked_widget.currentWidget() is viewer.montage_widget
    assert viewer.montage_labels == labels_before
    assert all(l.original_data is not None for l in viewer.montage_labels)  # not deleted
    assert viewer.active_label is active_before
    assert viewer.current_file_path == path_before
    assert "bad1.png" in viewer.status_bar.currentMessage()


def test_key_c_on_empty_montage_page_does_not_raise(viewer, qtbot):
    viewer.show()
    viewer.montage_labels.clear()
    viewer.stacked_widget.setCurrentWidget(viewer.montage_widget)
    qtbot.keyClick(viewer, Qt.Key.Key_C)  # previously IndexError


def test_partial_montage_failure_is_reported(viewer, qtbot, tmp_path):
    a = _write_png(tmp_path / "a.png", 10)
    b = _write_png(tmp_path / "b.png", 200)
    bad = _write_bad(tmp_path / "broken.png")

    viewer.display_montage([a, bad, b])

    assert viewer.stacked_widget.currentWidget() is viewer.montage_widget
    assert [l.file_path for l in viewer.montage_labels] == [a, b]
    assert "broken.png" in viewer.status_bar.currentMessage()


def test_failed_montage_keeps_single_image_view(viewer, qtbot, tmp_path):
    good = _write_png(tmp_path / "good.png", 50)
    viewer.open_file(good)
    assert viewer.stacked_widget.currentWidget() is viewer.image_display_container
    data_before = viewer.image_label.original_data

    viewer.display_montage([_write_bad(tmp_path / "x.png"), _write_bad(tmp_path / "y.png")])

    assert viewer.stacked_widget.currentWidget() is viewer.image_display_container
    assert viewer.active_label is viewer.image_label
    assert viewer.current_file_path == good
    assert viewer.image_label.original_data is data_before
    assert viewer.histogram_action.isEnabled()


def test_missing_npz_key_is_reported_not_shown(viewer, qtbot, tmp_path):
    npz = _write_npz(tmp_path / "m.npz")
    png = _write_png(tmp_path / "p.png", 10)

    viewer.open_files([f"{npz}#zzz", png])

    assert [l.file_path for l in viewer.montage_labels] == [png]
    assert "m.npz#zzz" in viewer.status_bar.currentMessage()


def test_existing_npz_key_in_montage_shows_that_key(viewer, qtbot, tmp_path):
    npz = _write_npz(tmp_path / "m.npz")
    png = _write_png(tmp_path / "p.png", 10)

    viewer.open_files([f"{npz}#b", png])

    assert np.all(viewer.montage_labels[0].original_data == 1)  # key b is ones


def test_open_files_accepts_npz_key_paths(viewer, qtbot, tmp_path):
    npz = _write_npz(tmp_path / "m.npz")
    png = _write_png(tmp_path / "p.png", 10)

    viewer.open_files([f"{npz}#b", png])

    assert viewer.stacked_widget.currentWidget() is viewer.montage_widget
    assert [l.file_path for l in viewer.montage_labels] == [f"{npz}#b", png]


def test_open_files_single_npz_key_path(viewer, qtbot, tmp_path):
    npz = _write_npz(tmp_path / "m.npz")

    viewer.open_files([f"{npz}#b"])

    assert viewer.image_handler.current_npz_key == "b"


def test_gallery_keys_from_real_npz_flow(viewer, qtbot, tmp_path):
    # Real UI flow: open a keyed path in single mode, then a montage with a
    # keyed member, then back to single; no gallery key may be double-keyed.
    npz = _write_npz(tmp_path / "m.npz")
    png = _write_png(tmp_path / "p.png", 10)
    viewer.show()
    viewer.open_file(f"{npz}#b")
    viewer.thumbnail_pane.populate()
    viewer.open_files([f"{npz}#a", png])
    viewer.thumbnail_pane.populate()
    viewer.open_file(f"{npz}#a")
    viewer.thumbnail_pane.populate()

    keys = set(viewer.thumbnail_pane.gallery_images)
    assert {f"{npz}#a", f"{npz}#b"} <= keys
    assert not any(k.count("#") > 1 for k in keys)


def test_populate_strips_existing_npz_key(viewer, qtbot, tmp_path):
    # Unit-level: open_file() stores the base path, and no current UI flow
    # leaves a keyed current_file_path in single mode, so this exercises the
    # defensive strip in ThumbnailPane.populate directly.
    npz = _write_npz(tmp_path / "m.npz")
    viewer.show()
    viewer.open_file(f"{npz}#a")
    viewer.current_file_path = f"{npz}#a"

    viewer.thumbnail_pane.populate()

    keys = set(viewer.thumbnail_pane.gallery_images)
    assert f"{npz}#a" in keys and f"{npz}#b" in keys
    assert not any(k.count("#") > 1 for k in keys)
