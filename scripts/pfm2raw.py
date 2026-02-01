import numpy as np
import argparse
import math

from matplotlib import pyplot as plt


# ----------------------------
# PFM reader
# ----------------------------
def read_pfm(filename):
    with open(filename, "rb") as f:
        header = f.readline().decode().rstrip()
        if header not in ("PF", "Pf"):
            raise ValueError("Not a PFM file")

        color = header == "PF"

        width, height = map(int, f.readline().decode().split())
        scale = float(f.readline().decode())

        endian = "<" if scale < 0 else ">"
        scale = abs(scale)

        data = np.fromfile(f, endian + "f")
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)

        plt.imshow(data)
        plt.show()

        return data


# ----------------------------
# Read dmin/dmax from calib
# ----------------------------
def read_viz_range(calibfile):
    dmin = math.inf
    dmax = math.inf

    with open(calibfile, "r") as f:
        for line in f:
            if "vmin=" in line:
                dmin = float(line.split("=")[1])
            if "vmax=" in line:
                dmax = float(line.split("=")[1])

    if math.isinf(dmin) or math.isinf(dmax):
        raise RuntimeError("Cannot extract vmin/vmax from calib file")

    return dmin, dmax


# ----------------------------
# Jet colormap (faithful port)
# ----------------------------
def jet(x):
    if x < 0:
        x = -0.05
    if x > 1:
        x = 1.05

    x = x / 1.15 + 0.1

    r = int(round(255 * (1.5 - 4 * abs(x - 0.75))))
    g = int(round(255 * (1.5 - 4 * abs(x - 0.50))))
    b = int(round(255 * (1.5 - 4 * abs(x - 0.25))))

    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    return r, g, b


# ----------------------------
# Min / max ignoring INF
# ----------------------------
def get_min_max(img):
    valid = img[np.isfinite(img)]
    return float(valid.min()), float(valid.max())


# ----------------------------
# Float disparity → color
# ----------------------------
def float_to_color(fimg, dmin, dmax):
    h, w = fimg.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    scale = 1.0 / (dmax - dmin)

    for y in range(h):
        for x in range(w):
            f = fimg[y, x]
            if np.isfinite(f):
                val = scale * (f - dmin)
                r, g, b = jet(val)
                out[y, x] = [r, g, b]  # RGB
            else:
                out[y, x] = [0, 0, 0]

    return out


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)

    args = parser.parse_args()

    disp = read_pfm(args.input)

    disp[~np.isfinite(disp)] = 0.
    disp = disp.astype(np.float32)
    h, w = disp.shape

    out_path = args.input.replace(".pfm", f"_{w}x{h}.f32")
    disp.tofile(out_path)
    print(f"output is written to {out_path}")


if __name__ == "__main__":
    main()
