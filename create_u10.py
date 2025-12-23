import argparse

import numpy as np

def create_u10_image(width: int, height: int) -> np.ndarray:
    """
    Create a 10-bit (U10) gradient image.
    Horizontal gradient from 0 to 1023.

    Returns:
        np.ndarray of shape (height, width), dtype=np.uint16
    """
    if width <= 1 or height <= 0:
        raise ValueError("width must be > 1 and height > 0")

    # Create 1D gradient [0..1023]
    grad = np.linspace(0, 1023, width, dtype=np.uint16)

    # Tile vertically
    image = np.tile(grad, (height, 1))

    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a U10 gradient image")
    parser.add_argument("--width", type=int, default=1920, help="Image width")
    parser.add_argument("--height", type=int, default=1080, help="Image height")
    parser.add_argument("--output", type=str, default="output.u10", help="Output file")

    args = parser.parse_args()

    img = create_u10_image(args.width, args.height)

    # Ensure little-endian uint16 on disk
    img.astype("<u2").tofile(args.output)

    print(
        f"Wrote {args.output}: "
        f"{args.width}x{args.height}, "
        f"uint16 U10, "
        f"range [{img.min()}, {img.max()}]"
    )