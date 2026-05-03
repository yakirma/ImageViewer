import sys
import os
from PyInstaller.utils.hooks import copy_metadata, collect_data_files, collect_dynamic_libs

IS_WINDOWS = sys.platform == 'win32'
IS_MACOS   = sys.platform == 'darwin'

APP_ICON = 'assets/icons/icon.ico' if IS_WINDOWS else 'assets/ImageViewer.icns'

# DA3 package data — graceful fallback so the build works without DA3 installed.
da3_datas = []
try:
    import depth_anything_3
    da3_datas = [(depth_anything_3.__path__[0], 'depth_anything_3')]
except ImportError:
    print("WARNING: depth_anything_3 not installed — DA3 depth generation will be unavailable in the bundle.")

a = Analysis(
    ['main.py'],
    pathex=[],
    # collect_dynamic_libs('numpy') is kept for Windows where the numpy DLL
    # discovery hook is unreliable across PyInstaller versions.
    binaries=collect_dynamic_libs('numpy'),
    datas=[
        ('assets/icons', 'assets/icons'),
    ] + da3_datas
      + copy_metadata('imageio')
      + copy_metadata('safetensors')
      + copy_metadata('huggingface_hub')
      + collect_data_files('numpy'),
    hiddenimports=[
        # DA3 inference stack (torch/torchvision/timm are NOT bundled —
        # they are downloaded at runtime into ~/.imageviewer/ml_packages)
        'depth_anything_3',
        # Image I/O
        'PIL', 'PIL.Image', 'PIL.ImageOps',
        'tifffile',
        'pillow_heif',
        'imageio', 'imageio.plugins.tifffile_v3',
        # Video
        'cv2',
        # Scientific config / metadata
        'omegaconf', 'addict', 'evo',
        # HuggingFace download
        'requests', 'safetensors', 'huggingface_hub',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # ── Unused Qt6 modules ──────────────────────────────────────────────
        # These add up to ~25-40 MB in the bundle and are never imported.
        'PyQt6.QtBluetooth',
        'PyQt6.QtLocation',
        'PyQt6.QtPositioning',
        'PyQt6.QtSensors',
        'PyQt6.QtSerialPort',
        'PyQt6.QtSerialBus',
        'PyQt6.QtWebEngine',
        'PyQt6.QtWebEngineCore',
        'PyQt6.QtWebEngineWidgets',
        'PyQt6.QtWebChannel',
        'PyQt6.QtMultimedia',
        'PyQt6.QtMultimediaWidgets',
        'PyQt6.QtRemoteObjects',
        'PyQt6.QtSql',
        'PyQt6.QtXml',
        'PyQt6.QtTest',
        'PyQt6.QtDesigner',
        'PyQt6.QtPrintSupport',
        'PyQt6.QtPdf',
        'PyQt6.QtPdfWidgets',
        'PyQt6.QtCharts',
        'PyQt6.QtDataVisualization',
        'PyQt6.QtNetworkAuth',
        'PyQt6.QtQuick',
        'PyQt6.QtQuickWidgets',
        'PyQt6.QtQml',
        # Qt C++ plugin names (macOS)
        'Qt6Location', 'Qt6Positioning', 'Qt6Sensors', 'Qt6WebEngine',
        'qdarwinpermissionplugin',
        # ── Torch / torchvision / timm — NOT bundled ───────────────────────
        # Downloaded at runtime into ~/.imageviewer/ml_packages on first use.
        # Excluding them here prevents PyInstaller from pulling them in through
        # depth_anything_3's static import graph.
        'torch', 'torchvision', 'timm',
        # ── Unused ML / scientific packages ────────────────────────────────
        'sklearn', 'IPython', 'notebook', 'pandas', 'sympy',
        'triton',               # GPU kernel compiler — not available on macOS
        # ── Unused stdlib modules ───────────────────────────────────────────
        'doctest', 'xmlrpc', 'ftplib',
        'email', 'html.server', 'http.server', 'imaplib', 'poplib', 'smtplib',
        'tkinter', 'turtle',
        'setuptools', 'pkg_resources',
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure, optimize=1)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ImageViewer',
    debug=False,
    bootloader_ignore_signals=False,
    # strip removes debug symbols from shared libraries on macOS/Linux,
    # cutting dylib size by ~20-30%.  Disabled on Windows (tool unavailable).
    strip=not IS_WINDOWS,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=APP_ICON,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=not IS_WINDOWS,
    upx=True,
    # UPX can silently corrupt large compiled libraries.  Exclude them so UPX
    # only compresses smaller Python extension modules where it is safe.
    upx_exclude=[
        # Qt frameworks / DLLs
        'Qt6Core.dll',   'Qt6Gui.dll',   'Qt6Widgets.dll',
        'Qt6OpenGL.dll', 'Qt6Network.dll',
        # Torch / BLAS
        'libtorch_cpu.dylib', 'libtorch_cpu.so', 'torch_cpu.dll',
        'libopenblas.dylib',  'libopenblas.so',  'libopenblas.dll',
        # OpenCV
        'libopencv_core.dylib', 'libopencv_videoio.dylib',
        'opencv_world*.dll',
    ],
    name='ImageViewer',
)

# BUNDLE is macOS-only.  PyInstaller silently ignores it on Windows, but
# keeping it conditional makes the spec easier to read and reason about.
if IS_MACOS:
    app = BUNDLE(
        coll,
        name='ImageViewer.app',
        icon='assets/ImageViewer.icns',
        bundle_identifier='com.yakirma.ImageViewer',
        info_plist={
            'CFBundleName': 'ImageViewer',
            'CFBundleDisplayName': 'ImageViewer',
            'CFBundleShortVersionString': '1.1.3',
            'CFBundleVersion': '1.1.3',
            'NSHighResolutionCapable': 'True',
            'NSRequiresAquaSystemAppearance': 'False',  # Supports dark mode
            'CFBundleDocumentTypes': [
                {
                    'CFBundleTypeName': 'Image File',
                    'CFBundleTypeRole': 'Viewer',
                    'LSHandlerRank': 'Alternate',
                    'LSItemContentTypes': [
                        'public.image', 'public.png', 'public.jpeg',
                        'public.tiff', 'com.adobe.raw-image',
                    ],
                    'CFBundleTypeExtensions': [
                        'png', 'jpg', 'jpeg', 'tif', 'tiff',
                        'raw', 'u16', 'f32', 'uint8', 'uint16', 'float32', 'bin',
                        'flo', 'npz', 'npy', 'heic', 'heif', 'webp', 'gif',
                    ],
                },
                {
                    'CFBundleTypeName': 'Video File',
                    'CFBundleTypeRole': 'Viewer',
                    'LSHandlerRank': 'Alternate',
                    'LSItemContentTypes': [
                        'public.movie', 'public.mpeg-4',
                        'com.apple.quicktime-movie', 'public.avi',
                    ],
                    'CFBundleTypeExtensions': [
                        'mp4', 'avi', 'mov', 'mkv', 'webm',
                    ],
                },
            ],
        },
    )
