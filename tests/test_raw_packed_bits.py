"""Regression tests for I17: N-bit samples stored in a wider container.

.u10/.u12/.u14 files and 'uintN'/'intN' overrides must drop the unused high
bits (and sign-extend signed data) on every single-plane path: grayscale with
an explicit size, grayscale with a guessed size, overrides, and Bayer.
"""
from __future__ import annotations

import numpy as np
import pytest

from image_handler import ImageHandler

# Words with garbage in the bits above the sample width.
WORDS = np.array([0xF123, 0x0FFF, 0xA800, 0x0001, 0xFFFF, 0x5000], np.uint16)


def _expected(bits, signed):
    v = WORDS.astype(np.int64) & ((1 << bits) - 1)
    if signed:
        v = np.where(v >= 1 << (bits - 1), v - (1 << bits), v)
    return v


def _write(path, words=WORDS):
    words.astype("<u2").tofile(path)
    return str(path)


@pytest.mark.parametrize("bits", [10, 12, 14])
def test_uN_extension_masks_explicit_resolution(tmp_path, bits):
    f = _write(tmp_path / f"img_3x2.u{bits}")
    h = ImageHandler()
    h.load_image(f)
    data = h.original_image_data
    assert data.shape == (2, 3)
    assert data.dtype == np.uint16
    np.testing.assert_array_equal(data.ravel(), _expected(bits, False))
    assert data.max() <= (1 << bits) - 1


def test_u12_example_from_issue(tmp_path):
    f = _write(tmp_path / "one_1x1.u12", np.array([0xF123], np.uint16))
    h = ImageHandler()
    h.load_image(f)
    assert int(h.original_image_data[0, 0]) == 0x123


@pytest.mark.parametrize("bits", [10, 12, 14])
def test_uN_extension_masks_guessed_resolution(tmp_path, bits):
    f = _write(tmp_path / f"noname.u{bits}")
    h = ImageHandler()
    h.load_image(f)
    data = h.original_image_data
    assert data.size == WORDS.size
    np.testing.assert_array_equal(np.sort(data.ravel()), np.sort(_expected(bits, False)))


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("bits", [10, 12, 14])
def test_override_masks_and_sign_extends(tmp_path, bits, signed):
    f = _write(tmp_path / "data.raw")
    h = ImageHandler()
    dtype = f"{'int' if signed else 'uint'}{bits}"
    h.load_image(f, {"width": 3, "height": 2, "dtype": dtype, "color_format": "Grayscale"})
    data = h.original_image_data
    assert data.dtype == (np.int16 if signed else np.uint16)
    np.testing.assert_array_equal(data.ravel(), _expected(bits, signed))
    if signed:
        assert data.min() >= -(1 << (bits - 1))
        assert data.max() <= (1 << (bits - 1)) - 1


def test_int12_example_from_issue(tmp_path):
    f = _write(tmp_path / "data.raw", np.array([0xF123, 0x0800, 0x07FF, 0xFFFF], np.uint16))
    h = ImageHandler()
    h.load_image(f, {"width": 4, "height": 1, "dtype": "int12"})
    assert h.original_image_data.ravel().tolist() == [291, -2048, 2047, -1]


def test_full_width_types_untouched(tmp_path):
    f = _write(tmp_path / "data.raw")
    h = ImageHandler()
    h.load_image(f, {"width": 3, "height": 2, "dtype": "uint16"})
    np.testing.assert_array_equal(h.original_image_data.ravel(), WORDS)
    h.load_image(f, {"width": 3, "height": 2, "dtype": "int16"})
    np.testing.assert_array_equal(h.original_image_data.ravel(), WORDS.view(np.int16))


@pytest.mark.parametrize("fmt", ["Bayer RGGB", "Bayer GRBG"])
def test_bayer_path_masks(tmp_path, fmt):
    words = np.full(8 * 8, 0xF123, np.uint16)
    f = _write(tmp_path / "bayer.raw", words)
    h = ImageHandler()
    h.load_image(f, {"width": 8, "height": 8, "dtype": "uint12", "color_format": fmt})
    rgb = h.original_image_data
    assert rgb.shape == (8, 8, 3)
    assert rgb.max() <= 4095
    np.testing.assert_allclose(rgb[2:-2, 2:-2], 0x123)


def test_bayer_signed_path_sign_extends(tmp_path):
    words = np.full(8 * 8, 0xFFFF, np.uint16)  # int12 -1 with garbage bits
    f = _write(tmp_path / "bayer.raw", words)
    h = ImageHandler()
    h.load_image(f, {"width": 8, "height": 8, "dtype": "int12", "color_format": "Bayer RGGB"})
    np.testing.assert_allclose(h.original_image_data[2:-2, 2:-2], -1)
