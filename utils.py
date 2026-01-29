import numpy as np
import os

def read_flow(filename):
    """
    Read .flo file in Middlebury format
    """
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        
        data = np.fromfile(f, np.float32, count=2*w*h)
        
        # Reshape data into 3D array (height, width, 2)
        flow = np.resize(data, (h, w, 2))
        return flow

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

    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]

    rad = np.sqrt(u ** 2 + v ** 2)
    
    if max_flow is not None and max_flow > 0:
        rad_max = max_flow
    else:
        rad_max = np.max(rad)

    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    return compute_color(u, v).astype(np.uint8)
