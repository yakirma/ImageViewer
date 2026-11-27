"""Regression tests for I19 (EXIF orientation ignored for JPEG/HEIC) and I22
(16-bit RGB(A) PNG/TIFF truncated to 8 bits; LA/PA alpha dropped) in
ImageHandler._load_standard_image."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from PyQt6.QtGui import QImage

from image_handler import ImageHandler


def _load(path):
    h = ImageHandler()
    h.load_image(str(path))
    return h


# ------------------------------------------------------------------ I19 EXIF

def _quadrant_rgb(h=20, w=40):
    """Distinct colors in each quadrant so orientation is checkable after JPEG."""
    a = np.zeros((h, w, 3), np.uint8)
    a[: h // 2, : w // 2] = (255, 0, 0)    # top-left red
    a[: h // 2, w // 2:] = (0, 255, 0)     # top-right green
    a[h // 2:, : w // 2] = (0, 0, 255)     # bottom-left blue
    a[h // 2:, w // 2:] = (255, 255, 255)  # bottom-right white
    return a


def _save_jpeg_with_orientation(path, arr, orientation):
    img = Image.fromarray(arr)
    exif = Image.Exif()
    exif[0x0112] = orientation
    img.save(path, quality=95, exif=exif.tobytes())


def _corner_colors(a):
    m = 3
    return [tuple(int(v) for v in a[y, x])
            for y, x in ((m, m), (m, -m - 1), (-m - 1, m), (-m - 1, -m - 1))]


def _expected(arr, orientation):
    # Ground truth: PIL's own transpose of the stored pixels.
    from PIL import ImageOps
    img = Image.fromarray(arr)
    exif = img.getexif()
    exif[0x0112] = orientation
    img.info["exif"] = exif.tobytes()
    return np.array(ImageOps.exif_transpose(img))


@pytest.mark.parametrize("orientation", [1, 3, 6, 8])
def test_jpeg_exif_orientation_applied(tmp_path, orientation):
    arr = _quadrant_rgb()
    p = tmp_path / f"o{orientation}.jpg"
    _save_jpeg_with_orientation(p, arr, orientation)
    h = _load(p)
    data = h.original_image_data
    exp = _expected(arr, orientation)
    assert data.shape == exp.shape
    assert (h.height, h.width) == exp.shape[:2]
    if orientation in (6, 8):
        assert data.shape[:2] == (40, 20)
    for got, want in zip(_corner_colors(data), _corner_colors(exp)):
        assert max(abs(g - w) for g, w in zip(got, want)) < 40


@pytest.mark.parametrize("orientation", range(1, 9))
def test_numpy_orientation_matches_pil(orientation):
    arr = np.arange(3 * 5 * 3, dtype=np.uint16).reshape(3, 5, 3)
    got = ImageHandler._apply_exif_orientation(arr, orientation)
    # Compare per-channel against PIL on an 8-bit-safe copy.
    exp = _expected((arr % 256).astype(np.uint8), orientation)
    assert got.shape == exp.shape
    assert np.array_equal(got % 256, exp)


_HEIC_SCRIPT = r"""
import sys, numpy as np
from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()
from image_handler import ImageHandler
a = np.zeros((32, 64, 3), np.uint8)
a[:16, :32] = (255, 0, 0)          # stored top-left red
im = Image.fromarray(a)
e = Image.Exif(); e[0x0112] = 6
try:
    im.save(sys.argv[1], quality=95, exif=e.tobytes())
except Exception:
    sys.exit(77)
h = ImageHandler(); h.load_image(sys.argv[1])
d = h.original_image_data
assert d.shape == (64, 32, 3), d.shape
assert (h.height, h.width) == (64, 32)
assert d[3, -4, 0] > 200 and d[3, 3, 0] < 50   # rotated CW: red at top-right
"""


def test_heic_exif_orientation_applied(tmp_path):
    # Run in a subprocess: the HEIC encoder can segfault when loaded next to
    # the Qt/OpenGL stack in the pytest process.
    import subprocess, sys
    from pathlib import Path
    pytest.importorskip("pillow_heif")
    root = str(Path(__file__).resolve().parents[1])
    r = subprocess.run([sys.executable, "-c", _HEIC_SCRIPT, str(tmp_path / "o6.heic")],
                       cwd=root, capture_output=True, text=True, timeout=60)
    if r.returncode in (77,) or r.returncode < 0:
        pytest.skip(f"HEIC encode unavailable (rc={r.returncode})")
    assert r.returncode == 0, r.stderr


# ------------------------------------------------------------- I22 16-bit RGB

def _rgb16():
    a = np.zeros((4, 6, 3), np.uint16)
    a[..., 0] = 60000
    a[..., 1] = np.arange(6, dtype=np.uint16) * 1000 + 3
    a[..., 2] = 7
    return a


@pytest.mark.parametrize("ext", [".png", ".tif"])
def test_16bit_rgb_keeps_uint16(tmp_path, ext):
    import cv2
    a = _rgb16()
    p = str(tmp_path / f"c16{ext}")
    assert cv2.imwrite(p, a[..., ::-1].copy())  # cv2 writes BGR
    h = _load(p)
    assert h.original_image_data.dtype == np.uint16
    assert np.array_equal(h.original_image_data, a)
    assert h.color_format == "RGB"
    assert (h.height, h.width) == (4, 6)


@pytest.mark.parametrize("ext", [".png", ".tif"])
def test_16bit_rgba_keeps_uint16_and_alpha(tmp_path, ext):
    import cv2
    a = np.concatenate([_rgb16(), np.full((4, 6, 1), 30000, np.uint16)], axis=2)
    p = str(tmp_path / f"c16a{ext}")
    assert cv2.imwrite(p, a[..., [2, 1, 0, 3]].copy())
    h = _load(p)
    assert h.original_image_data.dtype == np.uint16
    assert np.array_equal(h.original_image_data, a)
    assert h.color_format == "RGBA"


def test_16bit_rgb_tiff_honors_orientation(tmp_path):
    a = _rgb16()
    p = str(tmp_path / "o6.tif")
    import cv2
    cv2.imwrite(p, a[..., ::-1].copy())
    # Add an Orientation tag via PIL's tag editing on re-save is lossy for
    # 16-bit RGB, so check the helper path directly through a monkeypatch.
    h = ImageHandler()
    orig = Image.Image.getexif

    def fake_getexif(self):
        e = orig(self)
        e[0x0112] = 6
        return e

    import unittest.mock as um
    with um.patch.object(Image.Image, "getexif", fake_getexif):
        h.load_image(p)
    assert h.original_image_data.dtype == np.uint16
    assert np.array_equal(h.original_image_data, np.rot90(a, -1))
    assert (h.height, h.width) == (6, 4)


# --------------------------------------------------------------- I22 LA / PA

def test_la_keeps_alpha(tmp_path):
    L = np.arange(25, dtype=np.uint8).reshape(5, 5) * 10
    A = np.full((5, 5), 77, np.uint8)
    p = tmp_path / "la.png"
    Image.merge("LA", [Image.fromarray(L), Image.fromarray(A)]).save(p)
    h = _load(p)
    d = h.original_image_data
    assert d.shape == (5, 5, 4) and d.dtype == np.uint8
    assert np.array_equal(d[..., 0], L) and np.array_equal(d[..., 3], A)
    assert h.color_format == "RGBA"


def test_pa_and_p_transparency_keep_alpha(tmp_path):
    pal = Image.new("P", (4, 3))
    pal.putpalette([255, 0, 0, 0, 255, 0] + [0] * 762)
    pal.putdata([0, 1] * 6)
    p = tmp_path / "ptrans.png"
    pal.save(p, transparency=1)
    d = _load(p).original_image_data
    assert d.shape == (3, 4, 4)
    assert tuple(d[0, 0]) == (255, 0, 0, 255)
    assert d[0, 1, 3] == 0

    pa = pal.convert("PA")
    p2 = tmp_path / "pa.tif"
    try:
        pa.save(p2)
        with Image.open(p2) as chk:
            assert chk.mode == "PA"
    except Exception:
        pytest.skip("PA round-trip not supported by this Pillow build")
    d2 = _load(p2).original_image_data
    assert d2.shape == (3, 4, 4)


# ------------------------------------------------------------- unchanged paths

def test_plain_8bit_rgb_and_gray_unchanged(tmp_path):
    rgb = (np.arange(4 * 5 * 3) % 256).astype(np.uint8).reshape(4, 5, 3)
    gray = (np.arange(4 * 5) * 9 % 256).astype(np.uint8).reshape(4, 5)
    Image.fromarray(rgb).save(tmp_path / "rgb.png")
    Image.fromarray(gray).save(tmp_path / "g.png")
    g16 = (np.arange(20, dtype=np.uint16) * 3000).reshape(4, 5)
    Image.fromarray(g16).save(tmp_path / "g16.png")
    Image.new("P", (3, 2)).save(tmp_path / "p.png")  # no transparency

    h = _load(tmp_path / "rgb.png")
    assert h.original_image_data.dtype == np.uint8
    assert np.array_equal(h.original_image_data, rgb) and h.color_format == "RGB"
    h = _load(tmp_path / "g.png")
    assert np.array_equal(h.original_image_data, gray) and h.color_format == "Grayscale"
    h = _load(tmp_path / "g16.png")
    assert np.array_equal(h.original_image_data.astype(np.uint16), g16)
    h = _load(tmp_path / "p.png")
    assert h.original_image_data.shape == (2, 3, 3)


# ------------------------------------------------------- end-to-end display

def test_16bit_rgb_displays_and_reports_true_values(viewer, tmp_path):
    import cv2
    a = np.zeros((8, 64, 3), np.uint16)
    a[..., 0] = np.linspace(0, 65535, 64).astype(np.uint16)
    a[..., 1] = 30000
    p = str(tmp_path / "ramp16.png")
    cv2.imwrite(p, a[..., ::-1].copy())
    viewer.open_file(p)

    lab = viewer.image_label
    assert lab.original_data.dtype == np.uint16
    data = getattr(lab, "inspection_data", None)
    if data is None:
        data = lab.original_data
    assert tuple(int(v) for v in data[0, 63]) == (65535, 30000, 0)

    img = lab.current_pixmap.toImage().convertToFormat(QImage.Format.Format_RGB32)
    buf = np.frombuffer(img.constBits().asstring(img.sizeInBytes()), np.uint8)
    out = buf.reshape(img.height(), img.bytesPerLine() // 4, 4)[:, : img.width(), [2, 1, 0]].astype(int)
    # Red ramp spans the display range rather than wrapping or saturating.
    assert out[0, 0, 0] < 20 and out[0, -1, 0] > 235
    assert np.all(np.diff(out[0, :, 0]) >= 0)
    assert 60 < out[0, 0, 1] < 200  # mid-level green, not clipped
