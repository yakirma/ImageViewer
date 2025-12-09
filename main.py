import sys
from PyQt6.QtWidgets import QApplication
from image_viewer import ImageViewer

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setOrganizationName("ImageViewer")
    app.setApplicationName("ImageViewer")

    # Central list to manage open windows
    open_windows = []

    # Create the first window and pass it the list
    viewer = ImageViewer(window_list=open_windows)
    viewer.show()

    sys.exit(app.exec())