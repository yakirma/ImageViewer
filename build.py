import PyInstaller.__main__
import os
import platform
import sys
from PyQt6.QtCore import QLibraryInfo

# --- Configuration ---
APP_NAME = "ImageViewer"
ENTRY_POINT = "main.py"
ICON_MACOS = "assets/icons/icon.icns"
ICON_WINDOWS = "assets/icons/icon.ico"

def get_pyqtgraph_hooks():
    """
    Finds the path to the pyqtgraph hooks directory.
    This is often necessary to ensure pyqtgraph works correctly when bundled.
    """
    try:
        import pyqtgraph
        pg_path = os.path.dirname(pyqtgraph.__file__)
        hooks_path = os.path.join(pg_path, 'meta', 'pyinstaller-hooks')
        if os.path.exists(hooks_path):
            return hooks_path
    except ImportError:
        pass
    print("Warning: pyqtgraph hooks not found. The executable might not run correctly.")
    return None

def main():
    # --- Base PyInstaller command ---
    command = [
        '--name', APP_NAME,
        '--windowed',
        '--clean',
        '--noconfirm',
    ]

    # --- Platform-specific arguments ---
    system = platform.system()
    if system == "Windows":
        if os.path.exists(ICON_WINDOWS):
            command.extend(['--icon', ICON_WINDOWS])
    elif system == "Darwin":  # macOS
        if os.path.exists(ICON_MACOS):
            command.extend(['--icon', ICON_MACOS])

    # --- Add data and hooks for libraries ---
    
    # Correctly find and add Qt plugins
    plugins_path = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
    command.extend(['--add-data', f'{plugins_path}{os.pathsep}PyQt6/plugins'])

    # Add pyqtgraph hooks
    hooks_path = get_pyqtgraph_hooks()
    if hooks_path:
        command.extend(['--additional-hooks-dir', hooks_path])

    # --- Add the entry point script ---
    command.append(ENTRY_POINT)

    # --- Run PyInstaller ---
    print(f"Running PyInstaller with command: {' '.join(command)}")
    try:
        PyInstaller.__main__.run(command)
        print(f"\\nBuild complete. Look for the '{APP_NAME}' application in the 'dist' directory.")
    except Exception as e:
        print(f"An error occurred during the build process: {e}")

if __name__ == "__main__":
    main()
