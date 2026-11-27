import math
import os
import re
import numpy as np
from PIL import Image
from utils import read_flow, flow_to_color


def as_displayable(arr):
    """Cast boolean arrays to uint8 0/1 so display/stat code (np.percentile,
    histograms, colormaps) never sees numpy bool, which breaks subtraction.
    Other arrays are returned unchanged."""
    if isinstance(arr, np.ndarray) and arr.dtype == np.bool_:
        return arr.astype(np.uint8)
    return arr


class _DisplayableNpz:
    """Thin proxy over np.lib.npyio.NpzFile whose item access returns
    display-safe arrays (bool -> uint8). Every other attribute (files, close,
    fid, ...) is forwarded, so callers indexing ``handler.npz_data[key]``
    (key switching, montage loads, thumbnails) get uint8 for bool keys."""

    def __init__(self, npz_file):
        self._npz = npz_file

    def __getitem__(self, key):
        return as_displayable(self._npz[key])

    def __contains__(self, key):
        return key in self._npz

    def __iter__(self):
        return iter(self._npz)

    def __len__(self):
        return len(self._npz)

    def __getattr__(self, name):
        return getattr(self._npz, name)


# cv2 and pillow_heif are imported lazily — they're heavy native libs only
# needed when opening videos / HEIC files. Deferring saves ~80 ms in dev
# and several seconds of perceived startup in the frozen bundle.

_HEIF_REGISTERED = False

def _ensure_heif_opener():
    """Register pillow_heif's PIL plugin on first use. Safe to call repeatedly."""
    global _HEIF_REGISTERED
    if _HEIF_REGISTERED:
        return
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    _HEIF_REGISTERED = True


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
        self.video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.gif']
        self.flow_extensions = ['.flo']
        self.npz_extensions = ['.npz']
        # Extensions handled by the standard PIL loader. Any extension not in
        # raw/video/flow/npz/standard is treated as raw — scientific binary
        # blobs often have ad-hoc extensions.
        self.standard_extensions = [
            '.png', '.jpg', '.jpeg', '.bmp',
            '.tiff', '.tif', '.gif', '.webp',
            '.heic', '.heif',
        ]
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
        
        # Extensions that unambiguously identify a YUV layout (.yuv does not).
        self.yuv_extension_formats = {
            ".nv12": "YUV NV12", ".nv21": "YUV NV21",
            ".yuyv": "YUV YUYV", ".uyvy": "YUV UYVY",
        }

        # NPZ file support
        self.npz_keys = {}  # {key: is_valid_image}
        self.current_npz_key = None
        self.npz_file_path = None
        self.npz_data = None

    def is_single_channel_image(self):
        if self.original_image_data is None:
            return True
        return self.original_image_data.ndim == 2

    def is_raw_path(self, file_path):
        """Treat known raw extensions and any unknown extension as raw."""
        _, ext = os.path.splitext(file_path)
        ext_lower = ext.lower()
        if ext_lower in self.raw_extensions:
            return True
        # Known non-raw types are not raw.
        if (ext_lower in self.standard_extensions
                or ext_lower in self.video_extensions
                or ext_lower in self.flow_extensions
                or ext_lower in self.npz_extensions):
            return False
        # Unknown extension — treat as raw.
        return True

    def load_image(self, file_path, override_settings=None):
        _, ext = os.path.splitext(file_path)
        ext_lower = ext.lower()

        self.is_video = False
        if self.video_cap:
             self.video_cap.release()
             self.video_cap = None

        # Drop any state left over from a previously opened NPZ so that a
        # non-NPZ file doesn't inherit its keys/arrays (and close the handle).
        self._reset_npz_state()

        if ext_lower in self.npz_extensions:
            self.is_raw = False
            self._load_npz_file(file_path)
        elif ext_lower in self.video_extensions:
            self.is_raw = False
            self._load_video(file_path)
        elif ext_lower in self.flow_extensions:
            self.is_raw = False
            self._load_optical_flow(file_path)
        elif self.is_raw_path(file_path) or override_settings:
            # Known raw extension, unknown extension, or explicit override.
            self.is_raw = True
            self._load_raw_image(file_path, override_settings)
        else:
            self.is_raw = False
            self._load_standard_image(file_path)
        return True

    def _reset_npz_state(self):
        """Close any open NpzFile and clear all NPZ bookkeeping."""
        if self.npz_data is not None:
            try:
                self.npz_data.close()
            except Exception:
                pass
        self.npz_keys = {}
        self.npz_data = None
        self.current_npz_key = None
        self.npz_file_path = None

    def _load_video(self, file_path):
        import cv2
        self.video_cap = cv2.VideoCapture(file_path)
        if not self.video_cap.isOpened():
             raise Exception(f"Could not open video file: {file_path}")

        self.is_video = True
        fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        self.video_fps = fps if fps > 0 else 30.0
        self.video_frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.color_format = "RGB" # Videos opened as RGB via OpenCV usually


        # Load first frame
        self.seek_frame(0)

    def seek_frame(self, index):
        if not self.is_video or not self.video_cap: return False
        import cv2

        # Clamp
        if self.video_frame_count > 0:
            index = max(0, min(index, self.video_frame_count - 1))

        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        return self._update_from_current_frame(expected_index=index)

    def get_next_frame(self):
        if not self.is_video or not self.video_cap: return False
        return self._update_from_current_frame()

    def _update_from_current_frame(self, expected_index=None):
        import cv2
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

    def _load_optical_flow(self, file_name):
        flow_data = read_flow(file_name)
        if flow_data is not None:
            # Convert flow to RGB for visualization
            self.original_image_data = flow_to_color(flow_data)
            self.height, self.width, _ = self.original_image_data.shape
            self.dtype = self.original_image_data.dtype
            self.color_format = "RGB (Flow)"
        else:
            raise ValueError("Failed to read flow file")

    def _load_npz_file(self, file_path):
        """Load NPZ file and extract image arrays."""
        npz_file = _DisplayableNpz(np.load(file_path))
        try:
            # Store available keys and their validity
            keys = {}
            for key in npz_file.files:
                try:
                    arr = npz_file[key]
                except Exception:
                    # e.g. object arrays (allow_pickle=False) or corrupt members
                    keys[key] = False
                    continue
                keys[key] = self._is_valid_image_array(arr)

            # Load the first valid key (also covers the single-key case, which
            # must pass the same validity checks).
            first_valid = next((k for k in npz_file.files if keys[k]), None)
            if first_valid is None:
                if len(npz_file.files) == 1:
                    raise ValueError(
                        f"Array '{npz_file.files[0]}' in NPZ file is not an image array: {file_path}")
                raise ValueError(f"No valid image arrays found in NPZ file: {file_path}")
            data = npz_file[first_valid]
        except Exception:
            npz_file.close()
            self._reset_npz_state()
            raise

        self.npz_keys = keys
        self.original_image_data = data
        self.current_npz_key = first_valid

        # Store reference for switching keys later
        self.npz_file_path = file_path
        self.npz_data = npz_file

        # Set dimensions
        if data.ndim == 2:
            self.height, self.width = data.shape
        else:
            self.height, self.width = data.shape[:2]
        self.dtype = data.dtype
        self.color_format = self._color_format_for_array(data)

    @staticmethod
    def _color_format_for_array(arr):
        """Describe an in-memory image array's layout for the Info pane."""
        if arr.ndim == 3:
            return {2: "RGB (Flow)", 3: "RGB", 4: "RGBA"}.get(arr.shape[2], "Grayscale")
        return "Grayscale"

    def _is_valid_image_array(self, arr):
        """Check if array can be displayed as an image."""
        if not isinstance(arr, np.ndarray):
            return False
        # Object/string/void arrays cannot be rendered as pixels.
        if arr.dtype.kind not in "biuf":
            return False
        if arr.ndim not in [2, 3]:
            return False
        if arr.ndim == 3 and arr.shape[2] not in [2, 3, 4]:
            return False
        # Check that array has reasonable dimensions
        if arr.shape[0] < 1 or arr.shape[1] < 1:
            return False
        return True

    @staticmethod
    def _is_16bit_color(img):
        """True if PIL opened a 16-bit-per-channel RGB(A) file as 8-bit RGB(A).

        PIL has no 16-bit multi-channel modes, so it silently truncates such
        PNG/TIFF files to 8 bits; the decoder's raw mode (e.g. 'RGB;16B')
        still reveals the true depth.
        """
        if img.mode not in ("RGB", "RGBA"):
            return False
        try:
            tile = img.tile
        except Exception:
            return False
        if not tile:
            return False
        args = tile[0][3] if len(tile[0]) > 3 else None
        rawmode = args if isinstance(args, str) else (args[0] if args and isinstance(args[0], str) else "")
        return ";16" in rawmode

    @staticmethod
    def _apply_exif_orientation(arr, orientation):
        """Apply an EXIF Orientation (1-8) to an HxW[xC] array."""
        if orientation == 2:
            arr = arr[:, ::-1]
        elif orientation == 3:
            arr = arr[::-1, ::-1]
        elif orientation == 4:
            arr = arr[::-1]
        elif orientation == 5:
            arr = np.swapaxes(arr, 0, 1)
        elif orientation == 6:
            arr = np.rot90(arr, -1)
        elif orientation == 7:
            arr = np.swapaxes(arr[::-1, ::-1], 0, 1)
        elif orientation == 8:
            arr = np.rot90(arr, 1)
        return np.ascontiguousarray(arr)

    @staticmethod
    def _read_16bit_color_cv2(file_name):
        """Decode a 16-bit RGB(A) file with OpenCV, returning RGB(A) order."""
        import cv2
        buf = np.fromfile(file_name, dtype=np.uint8)  # unicode-safe on Windows
        arr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if arr is None or arr.dtype != np.uint16 or arr.ndim != 3 or arr.shape[2] not in (3, 4):
            return None
        if arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    def _load_standard_image(self, file_name):
        from PIL import ImageOps
        if file_name.lower().endswith((".heic", ".heif")):
            _ensure_heif_opener()
        with Image.open(file_name) as img:
            data = None
            if self._is_16bit_color(img):
                try:
                    data = self._read_16bit_color_cv2(file_name)
                except Exception:
                    data = None
                if data is not None:
                    try:
                        orientation = img.getexif().get(0x0112, 1)
                    except Exception:
                        orientation = 1
                    data = self._apply_exif_orientation(data, orientation)

            if data is None:
                # Honor EXIF Orientation (phone JPEG/HEIC); no-op without EXIF.
                try:
                    img = ImageOps.exif_transpose(img)
                except Exception:
                    pass
                if img.mode.startswith("I;16") or img.mode in ("I", "F", "L", "RGB", "RGBA"):
                    data = np.array(img)
                elif img.mode in ("LA", "La", "PA", "RGBa") or (
                        img.mode == "P" and "transparency" in img.info):
                    # Keep the alpha channel instead of dropping it.
                    data = np.array(img.convert("RGBA"))
                else:
                    # Fallback for P (Palette), CMYK, YCbCr, etc. -> Convert to RGB
                    data = np.array(img.convert("RGB"))
            self.original_image_data = data

            self.height, self.width = data.shape[:2]
            self.dtype = self.original_image_data.dtype
            
            # Set color format to display in info pane
            if self.original_image_data.ndim == 3:
                if self.original_image_data.shape[2] == 4:
                    self.color_format = "RGBA"
                else:
                    self.color_format = "RGB"
            else:
                self.color_format = "Grayscale"

    def _load_raw_image(self, file_name, override_settings=None):
        self.color_format = "Grayscale" # Default for raw
        color_format = "Grayscale"
        
        if override_settings:
            width = override_settings['width']
            height = override_settings['height']
            dtype_str = override_settings['dtype']
            color_format = override_settings.get('color_format', "Grayscale")
            self.color_format = color_format
            
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
                 # No size in the name: use an estimate that fits the file
                 # exactly (e.g. for thumbnails), else give up.
                 width, height, dtype_raw, guess_fmt, _ = self.guess_raw_layout(file_name)
                 if width == 0 or height == 0:
                     raise ValueError("Resolution (_WxH) not found in filename.")
                 color_format = guess_fmt

            if isinstance(dtype_raw, str):
                 container, bits, is_signed, is_float = self._parse_dtype_string(dtype_raw)
                 dtype = dtype_raw # Keep logical name
            else:
                 container = dtype_raw
                 dtype = dtype_raw
                 bits = np.dtype(container).itemsize * 8
                 is_signed = np.dtype(container).kind == 'i'
                 is_float = np.dtype(container).kind == 'f'

            # Auto-select a YUV layout from unambiguous extensions.
            ext = os.path.splitext(file_name)[1].lower()
            ext_format = self.yuv_extension_formats.get(ext)
            if ext_format:
                color_format = ext_format
            elif ext == ".yuv" and os.path.getsize(file_name) == self._yuv_expected_size(width, height, "YUV I420"):
                # A .yuv frame of exactly 1.5*W*H bytes is 4:2:0; default to
                # planar I420 (ffmpeg yuv420p / test-sequence convention).
                color_format = "YUV I420"

        self.width = width
        self.height = height
        self.dtype = dtype
        self.color_format = color_format

        # YUV layouts are byte-packed planes, not W*H*C arrays: decode them
        # before the generic channel heuristic (which would misread them).
        if "YUV" in color_format:
            raw_data = np.fromfile(file_name, dtype=np.uint8)
            expected_size = self._yuv_expected_size(width, height, color_format)
            if raw_data.size < expected_size:
                raise ValueError(f"For {width}x{height} and format {color_format}, expected minimum {expected_size} bytes, found {raw_data.size}")
            self.original_image_data = self._process_yuv(raw_data[:expected_size], width, height, color_format)
            return

        # raw file reading
        raw_data = np.fromfile(file_name, dtype=container)
        # N-bit samples in a wider container (u10/u12/u14, intN): drop the
        # unused high bits and sign-extend, for every plane layout.
        raw_data = self._mask_packed_bits(raw_data, container, bits, is_signed, is_float)
        total_elements = raw_data.size
        expected_pixels = width * height
        
        channels = 1
        if total_elements > expected_pixels:
             if total_elements % expected_pixels == 0:
                 channels = int(total_elements // expected_pixels)
        


        # Heuristic: If we detected 16 channels of uint8, it's very likely float32 RGBA (4 bytes * 4 channels = 16 bytes)
        # This fixes loading .raw files that are actually float32 RGBA without explicit extension/settings
        if channels == 16 and container == np.uint8:
            try:
                raw_data_f32 = raw_data.view(np.float32)
                # Check if size matches
                if raw_data_f32.size == width * height * 4:
                    raw_data = raw_data_f32
                    channels = 4
                    self.dtype = np.float32
                    self.color_format = "RGBA"
                    print("Heuristic: Re-interpreted 16-channel uint8 as 4-channel float32 RGBA")
            except Exception as heuristic_e:
                print(f"Heuristic failed: {heuristic_e}")

        if channels > 1:
             # Reshape to (H, W, C)
             raw_data = raw_data.reshape((height, width, channels))
             
             # Set Color Format
             if channels == 3: self.color_format = "RGB"
             elif channels == 4: self.color_format = "RGBA"
             elif channels == 2: pass # Flow or 2-channel
             
             self.original_image_data = raw_data
             return # Success

        # Single Channel / Fallback Path
        if channels <= 1: # Only try this if we think it's 1 channel
            raw_data = raw_data.reshape((height, width))
        else:
            raise ValueError(f"Cannot reshape {total_elements} into ({height}, {width}, {channels})")
             
        self.original_image_data = raw_data
        
        # For non-Bayer single-channel images, we're done
        if "Bayer" not in color_format:
            return

        self.original_image_data = self._debayer(raw_data, color_format)

    @staticmethod
    def _mask_packed_bits(raw_data, container, bits, is_signed, is_float):
        """Keep the low `bits` of each sample; sign-extend if signed."""
        if is_float or bits <= 0:
            return raw_data
        container = np.dtype(container)
        container_bits = container.itemsize * 8
        if bits >= container_bits:
            return raw_data
        # Work on the unsigned view so masking ignores the container's sign.
        u = raw_data.view(np.dtype(f"u{container.itemsize}"))
        u = u & np.array((1 << bits) - 1, dtype=u.dtype)
        if not is_signed:
            return u.view(container)
        sign_bit = np.array(1 << (bits - 1), dtype=u.dtype)
        # (x ^ s) - s in the signed container gives the two's-complement value.
        return (u ^ sign_bit).view(container) - sign_bit.astype(container)

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

    # Resolutions tried first (exact pixel-count match) when a raw file has no
    # size in its name; plain factor searches get 16:9 sizes wrong.
    COMMON_RAW_RESOLUTIONS = [
        (1920, 1080), (1280, 720), (3840, 2160), (4096, 2160), (2560, 1440),
        (640, 480), (1280, 960), (1280, 800), (1920, 1200), (2048, 1536),
        (1600, 1200), (4000, 3000), (7680, 4320), (2048, 1080), (1440, 1080),
        (1280, 1024), (1024, 768), (800, 600), (320, 240), (752, 480),
        (2592, 1944), (3264, 2448), (4032, 3024), (4056, 3040), (4208, 3120),
    ]

    @classmethod
    def guess_exact_dims(cls, file_size, bpp, color_format="Grayscale"):
        """(width, height) with width*height*bpp == file_size, or None.

        Tries, in order: a common resolution, a square, then the factor pair
        closest to 4:3 (aspect within 4:1, preferring 16-aligned widths).
        """
        pixels = file_size / bpp if bpp > 0 else 0
        if pixels < 1 or pixels != int(pixels):
            return None
        pixels = int(pixels)
        need_even = "YUV" in color_format

        def ok(w, h):
            return w * h == pixels and not (need_even and (w % 2 or h % 2))

        for w, h in cls.COMMON_RAW_RESOLUTIONS:
            if ok(w, h):
                return w, h
        side = math.isqrt(pixels)
        if ok(side, side):
            return side, side
        best, best_score = None, None
        for h in range(1, side + 1):
            if pixels % h:
                continue
            w = pixels // h
            if w > 4 * h or not ok(w, h):
                continue
            score = (abs(math.log((w / h) / (4.0 / 3.0))), w % 16 != 0)
            if best_score is None or score < best_score:
                best, best_score = (w, h), score
        return best

    def guess_raw_layout(self, file_name, file_size=None):
        """Estimate parameters for a raw file without a resolution in its name.

        Returns (width, height, dtype, color_format, bytes_per_pixel); width and
        height are 0 when no size uses the whole file exactly.
        """
        ext = os.path.splitext(file_name)[1].lower()
        if file_size is None:
            file_size = os.path.getsize(file_name)
        dtype = self.dtype_map.get(ext, np.uint8)
        try:
            container = self._parse_dtype_string(dtype)[0] if isinstance(dtype, str) else dtype
            itemsize = np.dtype(container).itemsize
        except Exception:
            itemsize = 1
        color_format = self.yuv_extension_formats.get(ext, "Grayscale")
        channels = {".rgb": 3, ".bgr": 3, ".rgba": 4, ".bgra": 4}.get(ext, 1)
        if channels > 1:
            color_format = "RGB" if channels == 3 else "RGBA"
        bpp = float(itemsize * channels)
        if "YUV" in color_format:
            bpp = 1.5 if ("NV12" in color_format or "NV21" in color_format) else 2.0
        if ext == ".yuv":
            # A bare .yuv is conventionally planar 4:2:0 (ffmpeg yuv420p /
            # test sequences): prefer I420 when it uses the whole file, else
            # fall back to a single-plane (Grayscale) layout if that fits.
            dims = self.guess_exact_dims(file_size, 1.5, "YUV I420")
            if dims:
                return dims[0], dims[1], dtype, "YUV I420", 1.5
            dims = self.guess_exact_dims(file_size, bpp, color_format)
            if dims:
                return dims[0], dims[1], dtype, color_format, bpp
            return 0, 0, dtype, "YUV I420", 1.5
        dims = self.guess_exact_dims(file_size, bpp, color_format) or (0, 0)
        return dims[0], dims[1], dtype, color_format, bpp

    def parse_resolution(self, file_name):
        """Resolution from a `_WxH`/`-WxH` filename tag, else (0, 0)."""
        basename = os.path.basename(file_name)
        match = re.search(r"[\-_](\d+)x(\d+)", basename)
        width, height = (int(match.group(1)), int(match.group(2))) if match else (0, 0)

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
        
    @staticmethod
    def _yuv_expected_size(width, height, format_str):
        """Number of bytes a YUV frame of the given layout occupies."""
        if "NV12" in format_str or "NV21" in format_str or "I420" in format_str:
            # 4:2:0 — full-res Y plus two quarter-res chroma planes (rounded up).
            return width * height + 2 * ((width + 1) // 2) * ((height + 1) // 2)
        if "YUYV" in format_str or "UYVY" in format_str:
            if width % 2:
                raise ValueError(f"{format_str} requires an even width, got {width}")
            return width * height * 2
        raise ValueError(f"Unsupported YUV format: {format_str}")

    def _process_yuv(self, data, width, height, format_str):
        data = data.astype(np.float32) # Using float for calculation
        
        Y, U, V = None, None, None
        
        if "NV12" in format_str or "NV21" in format_str or "I420" in format_str:
            y_end = width * height
            cw, ch = (width + 1) // 2, (height + 1) // 2
            Y = data[:y_end].reshape((height, width))
            if "I420" in format_str:
                # Planar: U plane then V plane.
                U_sub = data[y_end:y_end + cw * ch].reshape((ch, cw))
                V_sub = data[y_end + cw * ch:y_end + 2 * cw * ch].reshape((ch, cw))
            else:
                # Semi-planar: interleaved UV (NV12) or VU (NV21).
                uv = data[y_end:y_end + 2 * cw * ch].reshape((ch, cw, 2))
                if "NV12" in format_str:
                    U_sub = uv[..., 0]; V_sub = uv[..., 1]
                else:
                    V_sub = uv[..., 0]; U_sub = uv[..., 1]
            U = np.repeat(np.repeat(U_sub, 2, axis=0), 2, axis=1)[:height, :width]
            V = np.repeat(np.repeat(V_sub, 2, axis=0), 2, axis=1)[:height, :width]
                
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
            
        with np.errstate(divide='ignore', invalid='ignore'):
            transformed_data = eval(expression, {"__builtins__": {}}, safe_dict)
            
            # If the result contains NaNs (e.g. from 0/0), convert them to Inf as requested
            if isinstance(transformed_data, np.ndarray) and np.issubdtype(transformed_data.dtype, np.floating):
                transformed_data[np.isnan(transformed_data)] = np.inf

        if not isinstance(transformed_data, np.ndarray):
            raise ValueError("Expression must return a NumPy array.")

        # Comparisons (e.g. "x > 5") yield bool masks; show them as 0/1.
        return as_displayable(transformed_data)