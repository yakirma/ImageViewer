import numpy as np
from PIL import Image
import os

def generate_map():
    width, height = 256, 256
    x = np.linspace(-3, 3, width)
    y = np.linspace(-3, 3, height)
    X, Y = np.meshgrid(x, y)
    
    # Generate peaks function (like Matlab's peaks)
    Z = 3 * (1-X)**2 * np.exp(-(X**2) - (Y+1)**2) \
      - 10 * (X/5 - X**3 - Y**5) * np.exp(-X**2 - Y**2) \
      - 1/3 * np.exp(-(X+1)**2 - Y**2)
      
    # Normalize to 0-255
    Z_norm = ((Z - Z.min()) / (Z.max() - Z.min()) * 255).astype(np.uint8)
    
    img = Image.fromarray(Z_norm)
    filename = "test_map_256x256.png"
    img.save(filename)
    print(f"Generated {filename}")

if __name__ == "__main__":
    generate_map()
