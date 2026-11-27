"""Regression tests for I21 (optical flow NaN/unknown values, truncated .flo)
and I29 (NPZ color_format, single-key validity, clean state on failure)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from PIL import Image

from image_handler import ImageHandler
from utils import flow_to_color, read_flow

FLO_MAGIC = np.float32(202021.25)


def _write_flo(path, w, h, data):
    with open(path, "wb") as f:
        np.array([FLO_MAGIC], np.float32).tofile(f)
        np.array([w, h], np.int32).tofile(f)
        np.asarray(data, np.float32).tofile(f)


def _ramp_flow(h=4, w=5):
    flow = np.zeros((h, w, 2), np.float32)
    flow[..., 0] = np.arange(w)[None, :] - w / 2
    flow[..., 1] = np.arange(h)[:, None] - h / 2
    return flow


# ---- I21: flow_to_color -------------------------------------------------

@pytest.mark.parametrize("bad", [np.nan, 1e10, -np.inf])
def test_flow_invalid_vectors_do_not_blacken_image(bad):
    flow = _ramp_flow()
    flow[0, 0] = (bad, 0.0)
    flow[1, 1] = (0.0, bad)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no 'invalid value in cast' etc.
        img = flow_to_color(flow)
    assert img.dtype == np.uint8 and img.shape == (4, 5, 3)
    # Invalid pixels are painted black.
    assert (img[0, 0] == 0).all() and (img[1, 1] == 0).all()
    # All other pixels match a flow where the invalid entries are simply
    # zero, i.e. the magnitude normalization ignored them.
    ref_flow = _ramp_flow()
    ref_flow[0, 0] = 0
    ref_flow[1, 1] = 0
    ref = flow_to_color(ref_flow)
    mask = np.ones((4, 5), bool)
    mask[0, 0] = mask[1, 1] = False
    assert np.array_equal(img[mask], ref[mask])
    assert img[mask].max() > 0


def test_flow_all_invalid_is_black():
    flow = np.full((2, 3, 2), np.nan, np.float32)
    img = flow_to_color(flow)
    assert img.shape == (2, 3, 3) and not img.any()


def test_flow_nan_max_flow_falls_back_to_autoscale():
    flow = _ramp_flow()
    assert np.array_equal(flow_to_color(flow, max_flow=float("nan")), flow_to_color(flow))


# ---- I21: read_flow -----------------------------------------------------

def test_read_flow_roundtrip(tmp_path):
    flow = _ramp_flow(3, 4)
    p = tmp_path / "ok.flo"
    _write_flo(p, 4, 3, flow)
    assert np.array_equal(read_flow(str(p)), flow)


def test_read_flow_truncated_raises(tmp_path):
    p = tmp_path / "trunc.flo"
    _write_flo(p, 4, 3, np.zeros(10, np.float32))  # needs 24 floats
    with pytest.raises(ValueError, match="Truncated"):
        read_flow(str(p))


def test_read_flow_empty_raises(tmp_path):
    p = tmp_path / "empty.flo"
    p.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        read_flow(str(p))


def test_read_flow_bad_magic_raises(tmp_path):
    p = tmp_path / "bad.flo"
    np.zeros(8, np.float32).tofile(p)
    with pytest.raises(ValueError, match="magic"):
        read_flow(str(p))


def test_handler_loads_flo_with_unknown_values(tmp_path):
    flow = _ramp_flow()
    flow[2, 2] = (np.nan, np.nan)
    flow[3, 4] = (1e10, 1e10)
    p = tmp_path / "unknown.flo"
    _write_flo(p, 5, 4, flow)
    h = ImageHandler()
    h.load_image(str(p))
    img = h.original_image_data
    assert h.color_format == "RGB (Flow)"
    assert (img[2, 2] == 0).all() and (img[3, 4] == 0).all()
    assert img[0, 0].any()


# ---- I29: NPZ -----------------------------------------------------------

@pytest.mark.parametrize("shape, fmt", [
    ((5, 6), "Grayscale"),
    ((5, 6, 3), "RGB"),
    ((5, 6, 4), "RGBA"),
    ((5, 6, 2), "RGB (Flow)"),
])
def test_npz_sets_color_format(tmp_path, shape, fmt):
    png = str(tmp_path / "rgb.png")
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(png)
    npz = str(tmp_path / "a.npz")
    np.savez(npz, img=np.ones(shape, np.float32))
    h = ImageHandler()
    h.load_image(png)
    assert h.color_format == "RGB"
    h.load_image(npz)
    assert h.color_format == fmt
    assert (h.height, h.width) == (5, 6)


def test_npz_multi_key_skips_invalid_and_sets_format(tmp_path):
    npz = str(tmp_path / "m.npz")
    np.savez(npz, vec=np.arange(10), rgba=np.zeros((3, 4, 4), np.uint8))
    h = ImageHandler()
    h.load_image(npz)
    assert h.npz_keys == {"vec": False, "rgba": True}
    assert h.current_npz_key == "rgba"
    assert h.color_format == "RGBA"


@pytest.mark.parametrize("arr", [
    np.arange(10),
    np.zeros((2, 3, 4, 5)),
    np.array([{"a": 1}, None], dtype=object),
    np.array(["x", "y"]),
])
def test_single_key_npz_non_image_raises_cleanly(tmp_path, arr):
    good = str(tmp_path / "good.npz")
    np.savez(good, a=np.zeros((5, 6)), b=np.ones((5, 6, 3)))
    bad = str(tmp_path / "bad.npz")
    np.savez(bad, only=arr)

    h = ImageHandler()
    h.load_image(good)
    old = h.npz_data
    with pytest.raises(ValueError, match="not an image array"):
        h.load_image(bad)
    assert old.fid is None
    assert h.npz_keys == {}
    assert h.npz_data is None
    assert h.current_npz_key is None
    assert h.npz_file_path is None


def test_failed_npz_load_closes_handle(tmp_path, monkeypatch):
    bad = str(tmp_path / "bad.npz")
    np.savez(bad, x=np.arange(3), y=np.arange(4))
    opened = []
    real_load = np.load

    def spy(*a, **k):
        r = real_load(*a, **k)
        opened.append(r)
        return r

    monkeypatch.setattr(np, "load", spy)
    h = ImageHandler()
    with pytest.raises(ValueError, match="No valid image arrays"):
        h.load_image(bad)
    assert opened and opened[0].fid is None
    assert h.npz_keys == {} and h.npz_data is None
    assert h.current_npz_key is None and h.npz_file_path is None
