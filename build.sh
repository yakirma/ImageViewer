#!/bin/bash

APP_NAME="ImageViewer"

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     OS_TYPE=Linux;;
    Darwin*)    OS_TYPE=Mac;;
    CYGWIN*)    OS_TYPE=Windows;;
    MINGW*)     OS_TYPE=Windows;;
    MSYS*)      OS_TYPE=Windows;;
    *)          OS_TYPE="UNKNOWN:${OS}"
esac

echo "Detected OS: $OS_TYPE"

# 1. Build the Application
echo "Starting Application Build..."
python build.py
if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

# 2. Package based on OS
echo "Packaging for $OS_TYPE..."

if [ "$OS_TYPE" == "Mac" ]; then
    # MacOS Packaging (.dmg)
    DMG_NAME="${APP_NAME}_Installer.dmg"
    APP_PATH="dist/${APP_NAME}.app"
    TMP_DIR="dist/dmg_root"

    # Clean up
    rm -rf "$TMP_DIR"
    rm -f "dist/$DMG_NAME"

    mkdir -p "$TMP_DIR"
    
    if [ ! -d "$APP_PATH" ]; then
        echo "Error: $APP_PATH not found."
        exit 1
    fi

    echo "Copying app bundle..."
    cp -a "$APP_PATH" "$TMP_DIR/"
    
    echo "Creating /Applications link..."
    ln -s /Applications "$TMP_DIR/Applications"

    echo "Creating DMG..."
    hdiutil create -volname "$APP_NAME" -srcfolder "$TMP_DIR" -ov -format UDZO "dist/$DMG_NAME"
    
    # Cleanup temp
    rm -rf "$TMP_DIR"
    
    echo "Packaging Complete: dist/$DMG_NAME"

elif [ "$OS_TYPE" == "Linux" ]; then
    # Linux Packaging (.deb)
    DEB_NAME="${APP_NAME}_Linux.deb"
    BUILD_DIR="dist/${APP_NAME}"
    PKG_ROOT="dist/package_root"
    VERSION="1.0.0"
    ARCH="amd64" # Assuming amd64 for now, could detect with uname -m
    
    if [ ! -d "$BUILD_DIR" ]; then
        echo "Error: $BUILD_DIR not found."
        exit 1
    fi
    
    echo "Creating Debian package structure..."
    rm -rf "$PKG_ROOT"
    mkdir -p "$PKG_ROOT/DEBIAN"
    mkdir -p "$PKG_ROOT/usr/bin"
    mkdir -p "$PKG_ROOT/usr/share/applications"
    mkdir -p "$PKG_ROOT/usr/share/icons/hicolor/256x256/apps"
    
    # 1. Control File
    cat > "$PKG_ROOT/DEBIAN/control" <<EOF
Package: imageviewer
Version: $VERSION
Section: graphics
Priority: optional
Architecture: $ARCH
Maintainer: Yakirma <user@example.com>
Description: Professional Image Analysis Tool
 ImageViewer is a powerful, cross-platform desktop application for viewing and analyzing scientific and standard image formats.
EOF

    # 2. Copy Binary (Directory)
    # We copy the entire PyInstaller folder to /usr/lib/imageviewer and link the executable
    mkdir -p "$PKG_ROOT/usr/lib/$APP_NAME"
    cp -r "$BUILD_DIR/"* "$PKG_ROOT/usr/lib/$APP_NAME/"
    
    # 3. Create Launcher Script
    cat > "$PKG_ROOT/usr/bin/$APP_NAME" <<EOF
#!/bin/sh
exec /usr/lib/$APP_NAME/$APP_NAME "\$@"
EOF
    chmod +x "$PKG_ROOT/usr/bin/$APP_NAME"
    
    # 4. Desktop File & Icon
    cp "assets/ImageViewer.desktop" "$PKG_ROOT/usr/share/applications/"
    # Convert png to icon if needed, or use existing. distinct prompt implies we have app_icon.png
    # Let's assume we use app_icon.png as the icon
    cp "assets/app_icon.png" "$PKG_ROOT/usr/share/icons/hicolor/256x256/apps/${APP_NAME}.png"
    
    # 5. Build .deb
    if command -v dpkg-deb >/dev/null 2>&1; then
        echo "Building .deb package..."
        dpkg-deb --build "$PKG_ROOT" "dist/$DEB_NAME"
        echo "Packaging Complete: dist/$DEB_NAME"
    else
        echo "Warning: 'dpkg-deb' not found. Falling back to .tar.gz"
        ARCHIVE_NAME="${APP_NAME}_Linux.tar.gz"
        tar -czf "dist/$ARCHIVE_NAME" -C dist "$APP_NAME"
        echo "Packaging Complete: dist/$ARCHIVE_NAME"
    fi
    
    # Cleanup
    rm -rf "$PKG_ROOT"

elif [ "$OS_TYPE" == "Windows" ]; then
    # Windows Packaging (Installer)
    
    if command -v makensis >/dev/null 2>&1; then
        echo "Building Windows Installer..."
        makensis installer.nsi
        echo "Packaging Complete: dist/${APP_NAME}_Setup.exe"
    else
        echo "Warning: 'makensis' not found. Falling back to .zip"
        ARCHIVE_NAME="${APP_NAME}_Windows.zip"
        
        if command -v zip >/dev/null 2>&1; then
            cd dist
            zip -r "$ARCHIVE_NAME" "$APP_NAME"
            cd ..
            echo "Packaging Complete: dist/$ARCHIVE_NAME"
        else
            echo "Error: 'zip' command not found. Please install zip or use PowerShell to compress."
            exit 1
        fi
    fi

else
    echo "Unknown OS: $OS_TYPE. Skipping packaging."
    exit 1
fi

echo "Build and Packaging Finished Successfully!"
