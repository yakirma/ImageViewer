# ImageViewer: A Professional Image Analysis Tool

ImageViewer is a powerful, cross-platform desktop application for viewing and analyzing scientific and standard image formats. Built with Python and PyQt6, it provides a fluid, multi-window interface packed with features designed for in-depth image comparison and analysis.

![ImageViewer Screenshot](https://via.placeholder.com/800x500.png?text=ImageViewer+Application+Screenshot)

*(Screenshot placeholder)*

---

## Dominant Features

### 1. Advanced Image Viewing
- **Wide Format Support:** Natively opens standard formats (PNG, JPEG, TIFF) and specialized raw data formats (`.raw`, `.f32`, `.uint8`, etc.).
- **Interactive Zoom & Pan:** Smooth, mouse-based zooming and panning allows for effortless exploration of large images.
- **Multi-Window Interface:** Open multiple images in separate, independent windows (`Ctrl+N`) for flexible workspace management.

### 2. Multi-Image Montage & Synchronized Analysis
- **Thumbnail Pane:** A dockable pane shows thumbnails of all open images across all windows.
- **Montage View:** Select multiple thumbnails (with `Shift` or `Ctrl/Cmd`) to display them side-by-side in a synchronized grid.
- **Synchronized Zoom & Pan:** When in montage view, zooming or panning one image instantly applies the same view transformation to all other images in the grid.
- **Synchronized Crosshair:** Press `c` to toggle a synchronized crosshair across all images in the montage. The status bar displays the pixel coordinates and values for each image under the cursor, enabling direct, precise comparison.

### 3. Powerful Analysis Tools
- **Live Histogram:** A dockable histogram updates in real-time to show the pixel distribution of the active image.
- **Interactive Contrast Stretching:** Drag the histogram bars or input absolute pixel values to perform non-destructive contrast stretching on both grayscale and color (per-channel) images.
- **Math Transforms:** Apply complex mathematical operations to image data using NumPy expressions (e.g., `np.log(x)`).
- **Colormaps:** Instantly apply standard colormaps like `viridis` or `turbo` to single-channel (grayscale or raw) images.

### 4. Modern & Intuitive UI
- **Drag and Drop:** Drag image files directly from your file explorer onto the application to open them.
- **Dockable Panes:** All tool panes (Histogram, Thumbnails, etc.) can be moved, resized, or hidden to customize your workspace.
- **Keyboard Shortcuts:** Designed for power users, with intuitive shortcuts for common actions.

---

## Installation & Usage

1.  **Clone the repository:**
    ```sh
    git clone <your-repository-url>
    cd ImageViewer
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```sh
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```sh
    python main.py
    ```

---

## Building the Application

A build script is included to create a standalone, distributable application using PyInstaller.
```sh
python build.py
```
The final executable will be located in the `dist/` directory.

---

## Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| `Ctrl+N` | Open a new, empty ImageViewer window. |
| `Ctrl+O` | Open the file dialog to load an image. |
| `M` | Apply a Min-Max contrast stretch to the active image. |
| `Shift+M` | Apply a 5%-95% percentile contrast stretch. |
| `c` | Toggle the synchronized crosshair in montage view. |
| `Arrow Up/Down` | Navigate thumbnails in the selection pane. |
| `Space` | Select the focused thumbnail. |
