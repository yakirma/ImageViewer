from image_handler import ImageHandler
import numpy as np

ih = ImageHandler()
try:
    ret = ih.parse_resolution("test_10x10.raw")
    print(f"Return: {ret}")
    print(f"Length: {len(ret)}")
except Exception as e:
    print(f"Error: {e}")
