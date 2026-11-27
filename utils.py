import numpy as np
import os

def read_flow(filename):
    """
    Read .flo file in Middlebury format.

    Raises ValueError with a descriptive message when the file is empty, has a
    bad magic number, invalid dimensions, or is truncated.
    """
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic.size != 1:
            raise ValueError(f"Invalid .flo file (empty or too short): {filename}")
        if magic[0] != np.float32(202021.25):
            raise ValueError(f"Invalid .flo file (bad magic number {magic[0]!r}): {filename}")

        dims = np.fromfile(f, np.int32, count=2)
        if dims.size != 2:
            raise ValueError(f"Invalid .flo file (missing width/height header): {filename}")
        w, h = int(dims[0]), int(dims[1])
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid .flo file (bad dimensions {w}x{h}): {filename}")

        expected = 2 * w * h
        data = np.fromfile(f, np.float32, count=expected)
        if data.size != expected:
            raise ValueError(
                f"Truncated .flo file: expected {expected} floats for {w}x{h} flow, "
                f"found {data.size}: {filename}")

        # Reshape data into 3D array (height, width, 2)
        return data.reshape((h, w, 2))

def make_color_wheel():
    """
    Generate color wheel according to Middlebury color code
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros((ncols, 3))

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY, 1)/RY)
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0, YG, 1)/YG)
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0, GC, 1)/GC)
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(0, CB, 1)/CB)
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0, BM, 1)/BM)
    col += BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(0, MR, 1)/MR)
    colorwheel[col:col+MR, 0] = 255
    
    return colorwheel

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    colorwheel = make_color_wheel()
    ncols = colorwheel.shape[0]

    rad = np.sqrt(u**2 + v**2)
    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    img = np.zeros((u.shape[0], u.shape[1], 3))

    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f)*col0 + f*col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.floor(255 * col).astype(int)

    return img

# Middlebury convention: flow components above this magnitude are 'unknown'.
UNKNOWN_FLOW_THRESH = 1e9


def flow_magnitude(flow_uv):
    """Per-pixel magnitude of an [H,W,2] flow, NaN where the vector is unknown.

    Unknown follows flow_to_color: a non-finite component or
    |component| > UNKNOWN_FLOW_THRESH. Callers deriving contrast / max_flow
    limits filter non-finite values, so unknown vectors cannot inflate them.
    """
    u = np.asarray(flow_uv[..., 0], dtype=np.float64)
    v = np.asarray(flow_uv[..., 1], dtype=np.float64)
    with np.errstate(invalid='ignore', over='ignore'):
        invalid = ~(np.isfinite(u) & np.isfinite(v)
                    & (np.abs(u) <= UNKNOWN_FLOW_THRESH)
                    & (np.abs(v) <= UNKNOWN_FLOW_THRESH))
        mag = np.sqrt(u ** 2 + v ** 2)
    mag[invalid] = np.nan
    return mag


def flow_to_color(flow_uv, max_flow=None):
    """
    Expects a 2D flow image of shape.
    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        max_flow (float, optional): Float to normalized max flow magnitude. Defaults to None (autoscale).
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have shape [H,W,2]'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'

    u = np.asarray(flow_uv[:, :, 0], dtype=np.float64)
    v = np.asarray(flow_uv[:, :, 1], dtype=np.float64)

    # Middlebury marks unknown flow with NaN or |component| > 1e9. Exclude
    # such pixels from the magnitude normalization and paint them black.
    with np.errstate(invalid='ignore'):
        invalid = ~(np.isfinite(u) & np.isfinite(v)
                    & (np.abs(u) <= UNKNOWN_FLOW_THRESH)
                    & (np.abs(v) <= UNKNOWN_FLOW_THRESH))
    if invalid.any():
        u = np.where(invalid, 0.0, u)
        v = np.where(invalid, 0.0, v)

    rad = np.sqrt(u ** 2 + v ** 2)

    if max_flow is not None and np.isfinite(max_flow) and max_flow > 0:
        rad_max = max_flow
    else:
        rad_max = float(np.max(rad)) if rad.size else 0.0

    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    img = compute_color(u, v).astype(np.uint8)
    img[invalid] = 0
    return img


def split_npz_key(path):
    """Split an NPZ key path of the form ``"file.npz#key"``.

    Returns ``(file_path, key)``; ``key`` is ``None`` when ``path`` is not an
    NPZ key path. Only splits when the part before the last ``'#'`` ends with
    ``.npz`` and either that file exists or the full path does not, so that
    ordinary paths containing ``'#'`` (e.g. ``/work/C#proj/x.png``) are kept
    intact.
    """
    if not path or '#' not in path:
        return path, None
    base, key = path.rsplit('#', 1)
    if base.lower().endswith('.npz') and (os.path.exists(base) or not os.path.exists(path)):
        return base, key
    return path, None
