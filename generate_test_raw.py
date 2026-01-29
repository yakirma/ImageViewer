
import numpy as np
import os

width = 256
height = 256
channels = 4
dtype = np.float32

# Create a gradient pattern
x = np.linspace(0, 1, width)
y = np.linspace(0, 1, height)
xv, yv = np.meshgrid(x, y)

# Channel 1: Horizontal Gradient (Red)
c1 = xv
# Channel 2: Vertical Gradient (Green)
c2 = yv
# Channel 3: Circle (Blue)
c3 = np.sqrt((xv - 0.5)**2 + (yv - 0.5)**2)
c3 = 1 - c3/np.max(c3)
# Channel 4: Noise (Alpha/Data)
c4 = np.random.rand(height, width)

# Stack channels
img = np.stack([c1, c2, c3, c4], axis=-1).astype(dtype)

# Naming convention: _WxH_dtype.raw
filename = f"test_{width}x{height}_float32.raw"

# Save
img.tofile(filename)
print(f"Generated {filename} with shape {img.shape} and dtype {dtype}")
