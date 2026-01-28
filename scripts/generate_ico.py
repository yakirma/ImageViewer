from PIL import Image
import os

def create_ico():
    png_path = "assets/app_icon.png"
    ico_path = "assets/icons/icon.ico"
    
    if not os.path.exists(png_path):
        print(f"Error: {png_path} not found.")
        return

    try:
        img = Image.open(png_path)
        # Create icon directory if needed (it exists but good practice)
        os.makedirs(os.path.dirname(ico_path), exist_ok=True)
        
        # Save as ICO (Pillow handles resizing/formatting)
        img.save(ico_path, format='ICO', sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
        print(f"Successfully created {ico_path}")
    except Exception as e:
        print(f"Failed to create icon: {e}")

if __name__ == "__main__":
    create_ico()
