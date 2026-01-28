import os
import re
import numpy as np
from PIL import Image
import cv2

class ImageHandler:
    def __init__(self):
        self.original_image_data = None
        self.width = 0
        self.height = 0
        self.dtype = None
        self.color_format = None
        self.is_raw = False
        self.raw_extensions = [
            ".raw", ".bin", ".dat", 
            ".f32", ".f16", 
            ".uint8", ".u8", 
            ".uint16", ".u16", 
            ".u10", ".u12", ".u14", 
            ".yuv", ".nv12", ".nv21", ".y",
            ".rgb", ".rgba", ".bgr", ".bgra",
            ".yuyv", ".uyvy"
        ]
        self.video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        self.is_video = False
        self.video_cap = None
        self.video_fps = 30
        self.video_frame_count = 0
        self.current_frame_index = -1
        self.dtype_map = {
            ".f32": np.float32, 
            ".f16": np.float16, 
            ".uint8": np.uint8, ".u8": np.uint8, ".raw": np.uint8, ".yuv": np.uint8, ".nv12": np.uint8, ".nv21": np.uint8, ".y": np.uint8,
             ".yuyv": np.uint8, ".uyvy": np.uint8,
            ".uint16": np.uint16, ".u16": np.uint16,
            ".u10": "uint10", ".u12": "uint12", ".u14": "uint14", # Often stored in 16-bit
            ".rgb": np.uint8, ".rgba": np.uint8, ".bgr": np.uint8, ".bgra": np.uint8
        }

    def is_single_channel_image(self):
        if self.original_image_data is None:
            return True
        return self.original_image_data.ndim == 2

    def load_image(self, file_path, override_settings=None):
        _, ext = os.path.splitext(file_path)
        ext_lower = ext.lower()
        self.is_raw = ext_lower in self.raw_extensions or (override_settings is not None)

        self.is_video = False
        if self.video_cap:
             self.video_cap.release()
             self.video_cap = None

        if ext_lower in self.video_extensions:
             self._load_video(file_path)
        elif self.is_raw or override_settings:
            # If we have settings, we treat it as raw even if extension is png
            self._load_raw_image(file_path, override_settings)
        else:
            self._load_standard_image(file_path)

    def _load_video(self, file_path):
        self.video_cap = cv2.VideoCapture(file_path)
        if not self.video_cap.isOpened():
             raise Exception(f"Could not open video file: {file_path}")
        
        self.is_video = True
        fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        self.video_fps = fps if fps > 0 else 30.0
        self.video_frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Load first frame
        self.seek_frame(0)

    def seek_frame(self, index):
        if not self.is_video or not self.video_cap: return False
        
        # Clamp
        if self.video_frame_count > 0:
            index = max(0, min(index, self.video_frame_count - 1))
        
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        return self._update_from_current_frame(expected_index=index)

    def get_next_frame(self):
        if not self.is_video or not self.video_cap: return False
        return self._update_from_current_frame()

    def _update_from_current_frame(self, expected_index=None):
        ret, frame = self.video_cap.read()
        if ret:
             if expected_index is not None:
                 self.current_frame_index = expected_index
             else:
                 self.current_frame_index = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
             
             # Convert BGR to RGB
             self.original_image_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
             self.dtype = self.original_image_data.dtype
             return True
        return False

    def _load_standard_image(self, file_name):
        try:
            with Image.open(file_name) as img:
                if img.mode.startswith("I;16") or img.mode in ("I", "F"):
                    self.original_image_data = np.array(img)
                elif img.mode in ("RGB", "RGBA"):
                    self.original_image_data = np.array(img.convert("RGB"))
                else:
                    self.original_image_data = np.array(img.convert("L"))

                self.width, self.height = img.width, img.height
                self.dtype = self.original_image_data.dtype
        except Exception:
             raise

    def _load_raw_image(self, file_name, override_settings=None):
        color_format = "Grayscale"
        
        if override_settings:
            width = override_settings['width']
            height = override_settings['height']
            dtype_str = override_settings['dtype']
            color_format = override_settings.get('color_format', "Grayscale")
            
            if isinstance(dtype_str, type) or isinstance(dtype_str, np.dtype):
                dtype_val = np.dtype(dtype_str)
                bits = dtype_val.itemsize * 8
                is_signed = dtype_val.kind == 'i'
                is_float = dtype_val.kind == 'f'
                dtype = dtype_val # Logical and container are same
                container = dtype_val
            else:
                container, bits, is_signed, is_float = self._parse_dtype_string(str(dtype_str))
                dtype = str(dtype_str) # logical dtype is string
        else:
            width, height, dtype_raw = self.parse_resolution(file_name)
            
            if width == 0 or height == 0:
                 raise ValueError("Resolution (_WxH) not found in filename.")

            if isinstance(dtype_raw, str):
                 container, bits, is_signed, is_float = self._parse_dtype_string(dtype_raw)
                 dtype = dtype_raw # Keep logical name
            else:
                 container = dtype_raw
                 dtype = dtype_raw
                 bits = np.dtype(container).itemsize * 8
                 is_signed = np.dtype(container).kind == 'i'
                 is_float = np.dtype(container).kind == 'f'

        self.width = width
        self.height = height
        self.dtype = dtype
        self.color_format = color_format

        # raw file reading
        raw_data = np.fromfile(file_name, dtype=container)
        
        # Determine expected size based on format
        expected_size = width * height
        
        if "YUV" in color_format:
            if "NV12" in color_format or "NV21" in color_format or "I420" in color_format:
                 expected_size = int(width * height * 1.5)
            elif "YUYV" in color_format or "UYVY" in color_format:
                 expected_size = width * height * 2 # 2 bytes per pixel
        
        if raw_data.size < expected_size:
            # Handle YUV reading mismatch if dtype was guessed wrong (e.g. 16bit but YUV is 8bit stream)
            if "YUV" in color_format and dtype != np.uint8:
                 raw_data = np.fromfile(file_name, dtype=np.uint8)
            
            # If still mismatch
            if raw_data.size < expected_size:
                 # Truncate request or fail? Fail for now
                 pass 
                 # However, let's proceed to allow partial reads if valid
    
        if raw_data.size != expected_size:
             if raw_data.size > expected_size:
                 raw_data = raw_data[:expected_size]
             elif raw_data.size < expected_size:
                 raise ValueError(f"For {width}x{height} and format {color_format}, expected minimum {expected_size} elements, found {raw_data.size}")

        # Post-Processing: Masking for non-standard bits
        if not is_float and not "YUV" in color_format:
             container_bits = np.dtype(container).itemsize * 8
             if bits < container_bits:
                 mask = (1 << bits) - 1
                 raw_data = raw_data & mask
                 if is_signed:
                     # Sign extension
                     sign_bit = 1 << (bits - 1)
                     raw_data = (raw_data ^ sign_bit) - sign_bit

        # Reshaping
        if "YUV" in color_format:
            self.original_image_data = self._process_yuv(raw_data, width, height, color_format)
        elif "Bayer" in color_format:
             image = raw_data.reshape((height, width))
             self.original_image_data = self._debayer(image, color_format)
        else:
             self.original_image_data = raw_data.reshape((height, width))

    def _parse_dtype_string(self, type_str):
        type_str = type_str.lower().strip()
        
        # Handle "uintN" or "intN"
        is_signed = type_str.startswith("int")
        is_float = "float" in type_str
        
        if is_float:
            if "64" in type_str: return np.float64, 64, True, True
            if "16" in type_str: return np.float16, 16, True, True
            return np.float32, 32, True, True
            
        match = re.search(r'\d+', type_str)
        if match:
            bits = int(match.group(0))
        else:
            bits = 8
            
        if bits <= 8: container = np.int8 if is_signed else np.uint8
        elif bits <= 16: container = np.int16 if is_signed else np.uint16
        elif bits <= 32: container = np.int32 if is_signed else np.uint32
        else: container = np.int64 if is_signed else np.uint64
        
        return container, bits, is_signed, False

    def parse_resolution(self, file_name):
        basename = os.path.basename(file_name)
        match = re.search(r"_(\d+)x(\d+)", basename)
        width, height = (int(match.group(1)), int(match.group(2))) if match else (0, 0)
        
        if width == 0:
             size = os.path.getsize(file_name)
             side = int(np.sqrt(size))
             if side * side == size:
                 width, height = side, side
        
        # Return 0,0 if not found instead of raising
        # if width == 0:
        #      raise ValueError("Resolution (_WxH) not found in filename.")

        _, ext = os.path.splitext(file_name)
        dtype = self.dtype_map.get(ext.lower(), np.uint8)
        return width, height, dtype

    def _debayer(self, img, format_str):
        # format_str: "Bayer GRBG", etc.
        pattern = format_str.split(" ")[1].upper() # GRBG
        h, w = img.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        
        # Pattern mapping
        bp = {
            'RGGB': {'R': (0,0), 'G1': (0,1), 'G2': (1,0), 'B': (1,1)},
            'GRBG': {'G1': (0,0), 'R': (0,1), 'B': (1,0), 'G2': (1,1)},
            'GBRG': {'G1': (0,0), 'B': (0,1), 'R': (1,0), 'G2': (1,1)},
            'BGGR': {'B': (0,0), 'G1': (0,1), 'G2': (1,0), 'R': (1,1)},
        }
        p = bp.get(pattern, bp['GRBG'])
        
        def get_mask(offset_y, offset_x):
            mask = np.zeros((h, w), dtype=bool)
            mask[offset_y::2, offset_x::2] = True
            return mask

        mask_r = get_mask(*p['R'])
        mask_b = get_mask(*p['B'])
        mask_g1 = get_mask(*p['G1'])
        mask_g2 = get_mask(*p['G2'])
        mask_g = mask_g1 | mask_g2
        
        # Helper for shifting with padding
        def shift(arr, dy, dx):
            res = np.zeros_like(arr)
            src_y_start = max(0, -dy); src_y_end = min(h, h-dy)
            src_x_start = max(0, -dx); src_x_end = min(w, w-dx)
            dst_y_start = max(0, dy); dst_y_end = min(h, h+dy)
            dst_x_start = max(0, dx); dst_x_end = min(w, w+dx)
            res[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = arr[src_y_start:src_y_end, src_x_start:src_x_end]
            return res

        img_float = img.astype(np.float32)
        
        # Averages
        H_avg = (shift(img_float, 0, -1) + shift(img_float, 0, 1)) / 2
        V_avg = (shift(img_float, -1, 0) + shift(img_float, 1, 0)) / 2
        X_avg = (shift(img_float, -1, -1) + shift(img_float, -1, 1) + shift(img_float, 1, -1) + shift(img_float, 1, 1)) / 4
        PLUS_avg = (shift(img_float, 0, -1) + shift(img_float, 0, 1) + shift(img_float, -1, 0) + shift(img_float, 1, 0)) / 4

        # Red
        rgb[..., 0] = img * mask_r
        rgb[..., 0] += X_avg * mask_b
        R_h = (shift(mask_r.astype(int), 0, 1) + shift(mask_r.astype(int), 0, -1)) > 0
        R_v = (shift(mask_r.astype(int), 1, 0) + shift(mask_r.astype(int), -1, 0)) > 0
        rgb[..., 0] += H_avg * (R_h & mask_g)
        rgb[..., 0] += V_avg * (R_v & mask_g)
        
        # Blue
        rgb[..., 2] = img * mask_b
        rgb[..., 2] += X_avg * mask_r
        B_h = (shift(mask_b.astype(int), 0, 1) + shift(mask_b.astype(int), 0, -1)) > 0
        B_v = (shift(mask_b.astype(int), 1, 0) + shift(mask_b.astype(int), -1, 0)) > 0
        rgb[..., 2] += H_avg * (B_h & mask_g)
        rgb[..., 2] += V_avg * (B_v & mask_g)
        
        # Green
        rgb[..., 1] = img * mask_g
        rgb[..., 1] += PLUS_avg * (mask_r | mask_b)
        
        return rgb.astype(img.dtype)
        
    def _process_yuv(self, data, width, height, format_str):
        data = data.astype(np.float32) # Using float for calculation
        
        Y, U, V = None, None, None
        
        if "NV12" in format_str or "NV21" in format_str:
            y_end = width * height
            Y = data[:y_end].reshape((height, width))
            uv = data[y_end:].reshape((height // 2, width // 2, 2))
            uv_full = np.repeat(np.repeat(uv, 2, axis=0), 2, axis=1)
            
            if "NV12" in format_str:
                U = uv_full[..., 0]; V = uv_full[..., 1]
            else:
                V = uv_full[..., 0]; U = uv_full[..., 1]
                
        elif "YUYV" in format_str or "UYVY" in format_str:
             half_w = width // 2
             macro = data.reshape((height, half_w, 4))
             
             if "YUYV" in format_str: # Y0 U Y1 V
                 Y0 = macro[..., 0]; U_sub = macro[..., 1]; Y1 = macro[..., 2]; V_sub = macro[..., 3]
             else: # U Y0 V Y1
                 U_sub = macro[..., 0]; Y0 = macro[..., 1]; V_sub = macro[..., 2]; Y1 = macro[..., 3]
                 
             U = np.repeat(U_sub, 2, axis=1)
             V = np.repeat(V_sub, 2, axis=1)
             Y = np.empty((height, width), dtype=np.float32)
             Y[:, 0::2] = Y0
             Y[:, 1::2] = Y1
             
        if Y is None: return np.zeros((height, width, 3), dtype=np.uint8)
        return self._yuv2rgb(Y, U, V)

    def _yuv2rgb(self, Y, U, V):
        R = Y + 1.402 * (V - 128)
        G = Y - 0.344136 * (U - 128) - 0.714136 * (V - 128)
        B = Y + 1.772 * (U - 128)
        rgb = np.stack([R, G, B], axis=-1)
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def apply_math_transform(self, expression, context_dict=None):
        if self.original_image_data is None and context_dict is None:
            raise ValueError("No image loaded to transform.")

        safe_dict = {
            'x': self.original_image_data.astype(np.float64) if self.original_image_data is not None else None,
            'np': np,
            'log': np.log, 'log10': np.log10, 'exp': np.exp,
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'sqrt': np.sqrt, 'abs': np.abs,
        }
        
        if context_dict:
            safe_dict.update(context_dict)
            
        transformed_data = eval(expression, {"__builtins__": {}}, safe_dict)

        if not isinstance(transformed_data, np.ndarray):
            raise ValueError("Expression must return a NumPy array.")

        return transformed_data