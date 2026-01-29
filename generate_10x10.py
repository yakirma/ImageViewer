import numpy as np
import os

width = 10
height = 10
channels = 4

# Create 10x10x4 float32 array
# Pattern: 
# R: Horizontal Gradient 0-1
# G: Vertical Gradient 0-1
# B: Diagonal
# A: Checkerboard (0.0 or 1.0)

data = np.zeros((height, width, channels), dtype=np.float32)

for y in range(height):
    for x in range(width):
        data[y, x, 0] = x / (width - 1)      # R
        data[y, x, 1] = y / (height - 1)     # G
        data[y, x, 2] = (x+y) / (width+height-2) # B
        
        # Alpha Checkerboard
        if (x + y) % 2 == 0:
            data[y, x, 3] = 1.0 # Opaque
        else:
            data[y, x, 3] = 0.0 # Transparent

# Save as RAW
output_file = "test_10x10_rgba_float32.raw"
data.tofile(output_file)

print(f"Generated {output_file}: {width}x{height}x{channels} float32")
