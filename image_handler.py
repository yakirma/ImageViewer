import os
import re
import numpy as np
from PIL import Image


class ImageHandler:
    def __init__(self):
        self.original_image_data = None
        self.image_data = None
        self.width = 0
        self.height = 0
        self.dtype = None
        self.is_raw = False
        self.raw_extensions = [".raw", ".f32", ".f16", ".uint8"]
        self.dtype_map = {".f32": np.float32, ".f16": np.float16, ".uint8": np.uint8, ".raw": np.uint8}

    def load_image(self, file_path, override_settings=None):
        _, ext = os.path.splitext(file_path)
        self.is_raw = ext.lower() in self.raw_extensions

        if self.is_raw:
            self._load_raw_image(file_path, override_settings)
        else:
            self._load_standard_image(file_path)

    def _load_standard_image(self, file_name):
        with Image.open(file_name) as img:
            self.original_image_data = np.array(img.convert("L"))
            self.image_data = self.original_image_data.copy()
            self.width, self.height = img.width, img.height
            self.dtype = self.original_image_data.dtype

    def _load_raw_image(self, file_name, override_settings=None):
        if override_settings:
            width, height, dtype = override_settings['width'], override_settings['height'], override_settings['dtype']
        else:
            width, height, dtype = self._parse_raw_filename(file_name)

        self.width = width
        self.height = height
        self.dtype = dtype

        data = np.fromfile(file_name, dtype=dtype)
        expected_size = width * height
        if data.size != expected_size:
            raise ValueError(f"For {width}x{height}, expected {expected_size} data points, but found {data.size}")

        self.original_image_data = data.reshape((height, width))
        self.image_data = self.original_image_data.copy()

    def _parse_raw_filename(self, file_name):
        basename = os.path.basename(file_name)
        match = re.search(r"_(\d+)x(\d+)", basename)
        if not match:
            raise ValueError("Resolution (_WxH) not found in filename.")
        width, height = int(match.group(1)), int(match.group(2))
        _, ext = os.path.splitext(file_name)
        return width, height, self.dtype_map.get(ext.lower(), np.uint8)

    def apply_math_transform(self, expression):
        if self.original_image_data is None:
            raise ValueError("No image loaded to transform.")

        safe_dict = {
            'x': self.original_image_data,
            'np': np,
            'log': np.log, 'log10': np.log10, 'exp': np.exp,
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'sqrt': np.sqrt, 'abs': np.abs,
        }

        transformed_data = eval(expression, {"__builtins__": {}}, safe_dict)

        if not isinstance(transformed_data, np.ndarray) or transformed_data.shape != self.original_image_data.shape:
            raise ValueError("Expression must return a NumPy array of the same shape as the original image.")

        self.image_data = transformed_data
