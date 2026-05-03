import PyInstaller.__main__
import os
import platform
import sys
import subprocess
import shutil

APP_NAME    = "ImageViewer"
ENTRY_POINT = "main.py"
VERSION     = "1.1.2"

def run_pyinstaller():
    command = ['ImageViewer.spec', '--noconfirm', '--clean']
    print(f"Running PyInstaller: {' '.join(command)}")
    try:
        PyInstaller.__main__.run(command)
        print(f"\nBuild complete. Output is in the 'dist' directory.")
    except Exception as e:
        print(f"PyInstaller failed: {e}")
        sys.exit(1)

def package_windows():
    dist_dir = os.path.join("dist", APP_NAME)
    if not os.path.exists(dist_dir):
        print(f"Error: {dist_dir} not found — did the build succeed?")
        sys.exit(1)

    # Prefer NSIS installer; fall back to zip.
    try:
        subprocess.run(["makensis", "/VERSION"], check=False,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Building Windows installer with NSIS...")
        result = subprocess.run(["makensis", "installer.nsi"], check=False)
        if result.returncode == 0:
            print(f"Packaging complete: dist/{APP_NAME}_Setup.exe")
        else:
            print("NSIS returned an error — falling back to .zip")
            _make_zip(dist_dir)
    except FileNotFoundError:
        print("'makensis' not found — falling back to .zip archive.")
        _make_zip(dist_dir)

def _make_zip(dist_dir):
    archive = os.path.join("dist", f"{APP_NAME}_Windows")
    shutil.make_archive(archive, 'zip', root_dir='dist', base_dir=APP_NAME)
    print(f"Packaging complete: {archive}.zip")

def package_macos():
    app_path = os.path.join("dist", f"{APP_NAME}.app")
    if not os.path.exists(app_path):
        print(f"Error: {app_path} not found — did the build succeed?")
        sys.exit(1)

    dmg_name   = f"{APP_NAME}_Installer.dmg"
    dmg_path   = os.path.join("dist", dmg_name)
    tmp_dir    = os.path.join("dist", "dmg_root")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    if os.path.exists(dmg_path):
        os.remove(dmg_path)

    os.makedirs(tmp_dir)
    print("Copying app bundle...")
    shutil.copytree(app_path, os.path.join(tmp_dir, f"{APP_NAME}.app"),
                    symlinks=True)
    os.symlink("/Applications", os.path.join(tmp_dir, "Applications"))

    print("Creating DMG...")
    result = subprocess.run([
        "hdiutil", "create",
        "-volname", APP_NAME,
        "-srcfolder", tmp_dir,
        "-ov",
        "-format", "UDZO",   # zlib-compressed, widely compatible
        dmg_path,
    ], check=False)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    if result.returncode == 0:
        print(f"Packaging complete: {dmg_path}")
    else:
        print("hdiutil failed — the .app is still in dist/ and usable.")
        sys.exit(1)

def main():
    run_pyinstaller()

    print(f"\nStarting packaging for {platform.system()}...")
    os_name = platform.system()
    if os_name == "Windows":
        package_windows()
    elif os_name == "Darwin":
        package_macos()
    else:
        print("Linux packaging: use build.sh")

if __name__ == "__main__":
    main()
