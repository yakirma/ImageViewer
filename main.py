import sys
from PyQt6.QtWidgets import QApplication
from image_viewer import ImageViewer

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setOrganizationName("ImageViewer")
    app.setApplicationName("ImageViewer")
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec())
