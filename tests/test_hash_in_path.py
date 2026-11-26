"""Regression tests: '#' in a filesystem path must not be treated as an NPZ key
separator unless the path really is ``file.npz#key``."""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from utils import split_npz_key


def _png(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.full((5, 7), 9, np.uint8)).save(path)
    return path


class TestSplitNpzKey:
    def test_plain_path(self):
        assert split_npz_key("/a/b.png") == ("/a/b.png", None)

    def test_hash_in_directory_is_kept(self, tmp_path):
        p = _png(str(tmp_path / "C#proj" / "x.png"))
        assert split_npz_key(p) == (p, None)

    def test_hash_in_nonexistent_non_npz_path_is_kept(self):
        assert split_npz_key("/nope/C#proj/x.png") == ("/nope/C#proj/x.png", None)

    def test_npz_key_split(self, tmp_path):
        npz = str(tmp_path / "C#proj" / "d.npz")
        os.makedirs(os.path.dirname(npz))
        np.savez(npz, a=np.zeros((2, 2)))
        assert split_npz_key(npz + "#a") == (npz, "a")

    def test_npz_file_with_hash_in_name(self, tmp_path):
        npz = str(tmp_path / "x#1.npz")
        np.savez(npz, a=np.zeros((2, 2)))
        assert split_npz_key(npz) == (npz, None)


def test_open_file_with_hash_in_dir(viewer, tmp_path):
    p = _png(str(tmp_path / "C#proj" / "x.png"))
    viewer.open_file(p)
    assert viewer.image_label.file_path == p
    assert viewer.image_handler.original_image_data.shape[:2] == (5, 7)


def test_montage_with_hash_in_dir(viewer, tmp_path):
    p1 = _png(str(tmp_path / "C#proj" / "x.png"))
    p2 = _png(str(tmp_path / "plain" / "y.png"))
    viewer.display_montage([p1, p2])
    assert [lbl.file_path for lbl in viewer.montage_labels] == [p1, p2]
