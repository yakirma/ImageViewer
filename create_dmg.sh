#!/bin/bash
APP_NAME="ImageViewer"
DMG_NAME="ImageViewer_Installer.dmg"
APP_PATH="dist/ImageViewer.app"
TMP_DIR="dist/dmg_root"

# Clean up previous builds
rm -rf "$TMP_DIR"
rm -f "$DMG_NAME"
rm -f "dist/$DMG_NAME"

# Create temp directory structure
mkdir -p "$TMP_DIR"

# Check if App exists
if [ ! -d "$APP_PATH" ]; then
    echo "Error: $APP_PATH not found! Please run 'python build.py' first."
    exit 1
fi

# Copy App to temp dir
echo "Copying $APP_NAME.app to temporary directory..."
cp -r "$APP_PATH" "$TMP_DIR/"

# Create symlink to /Applications for easy drag-and-drop
echo "Creating link to /Applications..."
ln -s /Applications "$TMP_DIR/Applications"

# Create DMG using hdiutil
echo "Creating DMG..."
hdiutil create -volname "$APP_NAME" -srcfolder "$TMP_DIR" -ov -format UDZO "dist/$DMG_NAME"

# Cleanup
rm -rf "$TMP_DIR"

echo "Success! DMG created at dist/$DMG_NAME"
