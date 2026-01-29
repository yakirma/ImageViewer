#!/usr/bin/env python3
"""
Create sample NPZ files for testing NPZ support in ImageViewer
"""
import numpy as np

# Create test directory
import os
os.makedirs('test_npz', exist_ok=True)

# Test 1: Single array NPZ
print("Creating single array NPZ...")
img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
np.savez('test_npz/single_array.npz', image=img)

# Test 2: Multiple valid arrays
print("Creating multi-array NPZ with all valid images...")
rgb_img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
gray_img = np.random.randint(0, 255, (100, 150), dtype=np.uint8)
depth_map = np.random.rand(100, 150).astype(np.float32)
np.savez('test_npz/multi_valid.npz', rgb=rgb_img, gray=gray_img, depth=depth_map)

# Test 3: Mixed valid and invalid arrays
print("Creating mixed NPZ with some invalid arrays...")
valid_img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
invalid_scalar = np.array(42)  # 0D array - invalid
invalid_1d = np.array([1, 2, 3, 4, 5])  # 1D array - invalid
invalid_4d = np.random.rand(10, 10, 10, 10)  # 4D array - invalid
invalid_wrong_channels = np.random.rand(50, 50, 7)  # 7 channels - invalid
np.savez('test_npz/mixed.npz', 
         valid_image=valid_img,
         scalar=invalid_scalar,
         vector=invalid_1d,
         tensor_4d=invalid_4d,
         weird_channels=invalid_wrong_channels)

print("\nNPZ test files created in test_npz/ directory:")
print("1. single_array.npz - One RGB image")
print("2. multi_valid.npz - Three valid image arrays")
print("3. mixed.npz - One valid image + four invalid arrays (should be grayed out)")
print("\nTest by opening these files in ImageViewer!")
