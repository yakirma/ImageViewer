import numpy as np

import numpy as np

def read_10bit_raw(buffer_path, width, height):
    """
    Read a 10-bit RAW image stored as 16-bit little-endian values.

    Assumptions:
    - Each pixel stored in uint16
    - Only lower 10 bits are valid
    - Little-endian byte order

    Returns:
    - image: np.ndarray of shape (height, width), dtype=np.uint16
             values range: 0–1023
    """
    expected_pixels = width * height

    raw = np.fromfile(buffer_path, dtype=np.uint16)
    if raw.size != expected_pixels:
        raise ValueError(
            f"File size mismatch: expected {expected_pixels} pixels, got {raw.size}"
        )

    image = raw.reshape((height, width))

    # Mask to 10 bits just in case upper bits are non-zero
    image &= 0x03FF

    return image


if __name__ == '__main__':
    read_10bit_raw('/Users/ymatari/Git/ImageViewer/output.u10', 1920, 1080)