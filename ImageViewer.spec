from PyInstaller.utils.hooks import copy_metadata
import os
import depth_anything_3

da3_path = depth_anything_3.__path__[0]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('assets/icons', 'assets/icons'),
        (da3_path, 'depth_anything_3')
    ] + copy_metadata('imageio') + copy_metadata('safetensors') + copy_metadata('huggingface_hub'),
    hiddenimports=['requests', 'torch', 'torchvision', 'timm', 'imageio', 'omegaconf', 'addict', 'evo', 'depth_anything_3', 'PIL', 'tifffile', 'safetensors', 'huggingface_hub'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['Qt6Location', 'Qt6Positioning', 'qdarwinpermissionplugin', 'PyQt6.QtLocation', 'PyQt6.QtPositioning'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ImageViewer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets/ImageViewer.icns'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ImageViewer',
)
app = BUNDLE(
    coll,
    name='ImageViewer.app',
    icon='assets/ImageViewer.icns',
    bundle_identifier='com.yakirma.ImageViewer',
    info_plist={
        'CFBundleName': 'ImageViewer',
        'CFBundleDisplayName': 'ImageViewer',
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleVersion': '1.0.0',
        'NSHighResolutionCapable': 'True',
        'CFBundleDocumentTypes': [
            {
                'CFBundleTypeName': 'Image File',
                'CFBundleTypeRole': 'Viewer',
                'LSHandlerRank': 'Owner',
                'LSItemContentTypes': ['public.image', 'public.png', 'public.jpeg', 'public.tiff', 'com.adobe.raw-image'],
                'CFBundleTypeExtensions': ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'raw', 'u16', 'f32', 'uint8', 'uint16', 'float32', 'bin']
            },
             {
                'CFBundleTypeName': 'Video File',
                'CFBundleTypeRole': 'Viewer',
                'LSHandlerRank': 'Alternate',
                'LSItemContentTypes': ['public.movie', 'public.mpeg-4', 'com.apple.quicktime-movie'],
            }
        ]
    },
)
