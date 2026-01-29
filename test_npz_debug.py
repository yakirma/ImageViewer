#!/usr/bin/env python3
"""
Quick test to verify NPZ key paths in montage labels
"""
import sys
sys.path.insert(0, '/Users/ymatari/Git/ImageViewer')

from image_handler import ImageHandler

# Load the multi_valid NPZ
handler = ImageHandler()
handler.load_image('/Users/ymatari/Git/ImageViewer/test_npz/multi_valid.npz')

print("NPZ keys:", handler.npz_keys)
print("Current key:", handler.current_npz_key)
print("Current file path would be:", f"/Users/ymatari/Git/ImageViewer/test_npz/multi_valid.npz#{handler.current_npz_key}")

# Test loading specific keys
for key in handler.npz_keys.keys():
    if handler.npz_keys[key]:
        print(f"\nKey '{key}' - valid, shape:", handler.npz_data[key].shape)
