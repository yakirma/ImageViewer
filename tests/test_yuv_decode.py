"""Regression tests for I03: YUV raw layouts (NV12/NV21/I420/YUYV/UYVY) must be
decoded to RGB instead of being caught by the generic channel heuristic."""
from __future__ import annotations

import numpy as np
import pytest

from image_handler import ImageHandler

W, H = 8, 4
FORMATS = ["YUV NV12", "YUV NV21", "YUV I420", "YUV YUYV", "YUV UYVY"]


def _chroma_420():
    """Distinct (U, V) per 2x2 block so plane/order mix-ups are detectable."""
    cw, ch = W // 2, H // 2
    U = (40 + 20 * np.arange(cw * ch)).reshape(ch, cw).astype(np.uint8)
    V = (200 - 15 * np.arange(cw * ch)).reshape(ch, cw).astype(np.uint8)
    return U, V


def _y_plane():
    return (16 + 7 * np.arange(W * H) % 220).reshape(H, W).astype(np.uint8)


def _encode(fmt):
    """Return (file bytes, full-res Y, U, V) for a synthetic frame."""
    Y = _y_plane()
    if fmt in ("YUV NV12", "YUV NV21", "YUV I420"):
        U, V = _chroma_420()
        if fmt == "YUV I420":
            payload = np.concatenate([Y.ravel(), U.ravel(), V.ravel()])
        else:
            first, second = (U, V) if fmt == "YUV NV12" else (V, U)
            payload = np.concatenate([Y.ravel(), np.stack([first, second], -1).ravel()])
        Uf = np.repeat(np.repeat(U, 2, 0), 2, 1)
        Vf = np.repeat(np.repeat(V, 2, 0), 2, 1)
    else:  # 4:2:2 packed
        U = (30 + 25 * np.arange(H * W // 2)).reshape(H, W // 2).astype(np.uint8)
        V = (220 - 20 * np.arange(H * W // 2)).reshape(H, W // 2).astype(np.uint8)
        Y0, Y1 = Y[:, 0::2], Y[:, 1::2]
        order = (Y0, U, Y1, V) if fmt == "YUV YUYV" else (U, Y0, V, Y1)
        payload = np.stack(order, -1).ravel()
        Uf = np.repeat(U, 2, 1)
        Vf = np.repeat(V, 2, 1)
    return payload.astype(np.uint8), Y, Uf, Vf


def _reference_rgb(Y, U, V):
    """Full-range BT.601 reference (same convention as ImageHandler._yuv2rgb)."""
    try:
        import cv2
        # cv2's COLOR_YUV2RGB uses analog-YUV coefficients; YCrCb is the
        # full-range BT.601 (JPEG) transform the handler implements.
        return cv2.cvtColor(np.stack([Y, V, U], -1), cv2.COLOR_YCrCb2RGB)
    except ImportError:
        y, u, v = (a.astype(np.float64) for a in (Y, U, V))
        rgb = np.stack([y + 1.402 * (v - 128),
                        y - 0.344136 * (u - 128) - 0.714136 * (v - 128),
                        y + 1.772 * (u - 128)], -1)
        return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("fmt", FORMATS)
def test_yuv_override_decodes_to_rgb(tmp_path, fmt):
    payload, Y, U, V = _encode(fmt)
    path = tmp_path / f"frame_{W}x{H}.raw"
    payload.tofile(path)

    h = ImageHandler()
    h.load_image(str(path), override_settings={
        "width": W, "height": H, "dtype": "uint8", "color_format": fmt})

    out = h.original_image_data
    assert out.shape == (H, W, 3)
    assert out.dtype == np.uint8
    assert h.color_format == fmt  # kept so it can be inherited by the next file
    diff = np.abs(out.astype(int) - _reference_rgb(Y, U, V).astype(int))
    assert diff.max() <= 1, f"{fmt} max diff {diff.max()}"


@pytest.mark.parametrize("ext,fmt", [(".nv12", "YUV NV12"), (".nv21", "YUV NV21"),
                                     (".yuyv", "YUV YUYV"), (".uyvy", "YUV UYVY")])
def test_yuv_extension_autoselects_format(tmp_path, ext, fmt):
    payload, Y, U, V = _encode(fmt)
    path = tmp_path / f"frame_{W}x{H}{ext}"
    payload.tofile(path)

    h = ImageHandler()
    h.load_image(str(path))

    assert h.color_format == fmt
    assert h.original_image_data.shape == (H, W, 3)
    diff = np.abs(h.original_image_data.astype(int) - _reference_rgb(Y, U, V).astype(int))
    assert diff.max() <= 1


def test_yuv_truncated_file_raises(tmp_path):
    path = tmp_path / "short.raw"
    np.zeros(W * H, np.uint8).tofile(path)  # NV12 needs 1.5*W*H
    with pytest.raises(ValueError):
        ImageHandler().load_image(str(path), override_settings={
            "width": W, "height": H, "dtype": "uint8", "color_format": "YUV NV12"})


def test_nv12_odd_dimensions(tmp_path):
    w, h = 5, 3
    cw, ch = 3, 2
    payload = np.concatenate([np.full(w * h, 100, np.uint8), np.full(cw * ch * 2, 128, np.uint8)])
    path = tmp_path / "odd.raw"
    payload.tofile(path)
    hd = ImageHandler()
    hd.load_image(str(path), override_settings={
        "width": w, "height": h, "dtype": "uint8", "color_format": "YUV NV12"})
    assert hd.original_image_data.shape == (h, w, 3)
    assert (hd.original_image_data == 100).all()
