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


# ------------------------------------- reviewer follow-ups (16-bit alpha etc.)

def _pixmap_rgba(label):
    img = label.current_pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    buf = np.frombuffer(img.constBits().asstring(img.sizeInBytes()), np.uint8)
    return buf.reshape(img.height(), img.bytesPerLine() // 4, 4)[:, : img.width()].astype(int)


def _painted_rgb(label):
    from PyQt6.QtGui import QPainter
    img = QImage(label.size(), QImage.Format.Format_RGB32)
    img.fill(0)
    p = QPainter(img)
    label._paint_contents(p)
    p.end()
    buf = np.frombuffer(img.constBits().asstring(img.sizeInBytes()), np.uint8)
    arr = buf.reshape(img.height(), img.bytesPerLine() // 4, 4)[:, : img.width()]
    return arr[..., [2, 1, 0]].astype(int)


def _rgba16_partial_alpha():
    a = np.zeros((16, 16, 4), np.uint16)
    a[..., 0] = 65535
    a[..., 3] = 16384  # 25 % opaque
    return a


def test_uint16_rgba_partial_alpha_with_contrast_limits(qtbot):
    from widgets import ZoomableDraggableLabel
    lab = ZoomableDraggableLabel()
    qtbot.addWidget(lab)
    lab.resize(64, 64)
    lab.set_data(_rgba16_partial_alpha())
    lab.set_contrast_limits(0.0, 65535.0)
    px = _pixmap_rgba(lab)
    # Blended onto the dark-gray (40) background at 25 % -> ~94, not 255.
    expected = 255 * 0.25 + 40 * 0.75
    assert abs(px[8, 8, 0] - expected) <= 3
    assert abs(px[8, 8, 1] - 40 * 0.75) <= 3
    painted = _painted_rgb(lab)
    cy, cx = painted.shape[0] // 2, painted.shape[1] // 2
    assert painted[cy, cx, 0] < 150


def test_uint16_rgba_partial_alpha_auto_path(qtbot):
    from widgets import ZoomableDraggableLabel
    lab = ZoomableDraggableLabel()
    qtbot.addWidget(lab)
    lab.resize(64, 64)
    lab.set_data(_rgba16_partial_alpha())
    if lab.contrast_limits:
        pytest.skip("label applied contrast limits by default")
    px = _pixmap_rgba(lab)
    assert abs(px[8, 8, 3] - 64) <= 2  # 16384/65535 * 255


def test_overlay_display_array_normalizes():
    from image_viewer import _overlay_display_array
    rgb16 = np.zeros((2, 3, 3), np.uint16)
    rgb16[..., 0] = [[0, 30000, 60000]] * 2
    out = _overlay_display_array(rgb16)
    assert out.dtype == np.uint8 and out.shape == (2, 3, 3)
    assert out[0, 0, 0] == 0 and out[0, 2, 0] == 255 and 120 < out[0, 1, 0] < 135

    rgba16 = np.dstack([rgb16, np.full((2, 3), 16384, np.uint16)])
    out = _overlay_display_array(rgba16)
    assert out.shape == (2, 3, 4) and abs(int(out[0, 0, 3]) - 64) <= 1

    g16 = np.array([[1000, 2000, 3000]], np.uint16)
    assert _overlay_display_array(g16).tolist() == [[0, 127, 255]]
    gf = np.array([[np.nan, 0.0, 0.5, np.inf]], np.float32)
    out = _overlay_display_array(gf)
    assert out[0, 0] == 0 and out[0, 1] == 0 and out[0, 2] == 255
    assert not _overlay_display_array(np.full((2, 2), 7, np.uint16)).any()  # constant
    g8 = np.array([[5, 9]], np.uint8)
    assert _overlay_display_array(g8).tolist() == [[5, 9]]
    assert _overlay_display_array(np.zeros((2, 2, 5))) is None


def test_overlay_fallback_uses_normalized_16bit(viewer, tmp_path):
    """End-to-end: an overlay built from a 16-bit RGB file that is not open in
    any window is stretched to 8 bits, not reinterpreted as RGB888 bytes."""
    import cv2
    a = np.zeros((8, 8, 3), np.uint16)
    a[:, :4, 0] = 65535
    src = str(tmp_path / "ov16.png")
    cv2.imwrite(src, a[..., ::-1].copy())
    tgt = str(tmp_path / "target.png")
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(tgt)
    viewer.open_file(tgt)
    active = viewer.active_label.file_path
    pair = (src, active)
    viewer.overlay_alphas[pair] = 0.5
    viewer._update_overlays()
    assert pair in viewer.overlay_cache
    qimg = viewer.overlay_cache[pair].toImage().convertToFormat(QImage.Format.Format_RGB888)
    assert (qimg.width(), qimg.height()) == (8, 8)
    c = qimg.pixelColor(0, 0)
    assert (c.red(), c.green(), c.blue()) == (255, 0, 0)
    c = qimg.pixelColor(7, 0)
    assert (c.red(), c.green(), c.blue()) == (0, 0, 0)


def test_exif_transpose_matches_loader_for_depth_input(tmp_path):
    """DA3Worker feeds ImageOps.exif_transpose(img).convert('RGB'); it must have
    the same shape as what the viewer displays."""
    from PIL import ImageOps
    arr = _quadrant_rgb()
    p = tmp_path / "o6.jpg"
    _save_jpeg_with_orientation(p, arr, 6)
    with Image.open(p) as raw:
        depth_in = np.array(ImageOps.exif_transpose(raw).convert("RGB"))
    assert depth_in.shape == _load(p).original_image_data.shape
