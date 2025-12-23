import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt
from image_viewer import ImageViewer

if __name__ == "__main__":
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    app.setOrganizationName("ImageViewer")
    app.setApplicationName("ImageViewer")
    app.setWindowIcon(QIcon("assets/app_icon.png"))

    # Central list to manage open windows
    open_windows = []

    # Create the first window and pass it the list
    viewer = ImageViewer(window_list=open_windows)
    viewer.show()

    # Check for file arguments (CLI support)
    if len(sys.argv) > 1:
        # On macOS, if app is launched via Finder, the FileOpen event handles it.
        # But if launched from terminal, sys.argv is used.
        # If the argument is not a flag (doesn't start with -), try to open it.
        file_path = sys.argv[1]
        if not file_path.startswith("-"):
             from PyQt6.QtCore import QTimer
             # Delay slightly to ensure UI is ready
             QTimer.singleShot(100, lambda: viewer.open_file(file_path))

    sys.exit(app.exec())    