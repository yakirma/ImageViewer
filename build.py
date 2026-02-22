import PyInstaller.__main__
import os
import platform
import sys
from PyQt6.QtCore import QLibraryInfo

# --- Configuration ---
APP_NAME = "ImageViewer"
ENTRY_POINT = "main.py"
ICON_MACOS = "assets/ImageViewer.icns"
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
    # --- Run PyInstaller with the spec file ---
    # The spec file now contains all configuration (icon, data, plist, etc.)
    command = ['ImageViewer.spec', '--noconfirm', '--clean']
    
    print(f"Running PyInstaller with command: {' '.join(command)}")
    try:
        PyInstaller.__main__.run(command)
        print(f"\\nBuild complete. Look for the '{APP_NAME}' application in the 'dist' directory.")
    except Exception as e:
        print(f"An error occurred during the build process: {e}")
        sys.exit(1)

    # --- Packaging Step ---
    print(f"\\nStarting Packaging for {platform.system()}...")
    
    if platform.system() == "Windows":
        # Check for makensis (NSIS)
        nsis_found = False
        try:
            # Check if makensis is in PATH
            import subprocess
            subprocess.run(["makensis", "/VERSION"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            nsis_found = True
        except FileNotFoundError:
            nsis_found = False
            
        if nsis_found:
             print("Building Windows Installer with NSIS...")
             try:
                 subprocess.run(["makensis", "installer.nsi"], check=True)
                 print(f"Packaging Complete: dist/{APP_NAME}_Setup.exe")
             except subprocess.CalledProcessError as e:
                 print(f"Error building installer: {e}")
        else:
             print("Warning: 'makensis' not found. Falling back to .zip archive.")
             import shutil
             dist_dir = os.path.join("dist", APP_NAME)
             archive_name = os.path.join("dist", f"{APP_NAME}_Windows")
             if os.path.exists(dist_dir):
                 shutil.make_archive(archive_name, 'zip', root_dir='dist', base_dir=APP_NAME)
                 print(f"Packaging Complete: {archive_name}.zip")
             else:
                 print(f"Error: {dist_dir} not found, cannot zip.")

    else:
        print("Packaging not implemented in Python for this OS yet (use build.sh for Mac/Linux).")

if __name__ == "__main__":
    main()
