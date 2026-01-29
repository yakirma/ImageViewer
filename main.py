import sys
import os
import platform
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QFileDialog, QMessageBox, QSplitter
from PyQt6.QtCore import Qt, QTimer, QEvent, QSize, pyqtSignal, QPoint
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QColor, QPalette
from image_viewer import ImageViewer

# Define Custom Application Class at module level
class ImageViewerApp(QApplication):
    def event(self, event):
        if event.type() == QEvent.Type.FileOpen:
            # Handle macOS FileOpen event (Open With...)
            file_path = event.file()
            if file_path:
                # Delay to ensure main loop is running/window is ready
                QTimer.singleShot(100, lambda: self.open_file_event(file_path))
            return True
        return super().event(event)
        
    def open_file_event(self, file_path):
        # Access the global viewer instance or window list
        # Since we are in the App class, we need a reference to the windows.
        # We can find top level widgets.
        top_widgets = self.topLevelWidgets()
        for widget in top_widgets:
            if isinstance(widget, ImageViewer):
                widget.open_file(file_path)
                widget.raise_()
                widget.activateWindow()
                return
        
        # If no window found (unlikely if app is running), we could create one,
        # but for this specific flow 'viewer' object usually exists.

if __name__ == "__main__":
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    
    # Initialize Application ONCE
    app = ImageViewerApp(sys.argv)
    app.setOrganizationName("ImageViewer")
    app.setApplicationName("ImageViewer")
    app.setWindowIcon(QIcon("assets/app_icon.png"))

    # Central list to manage open windows (optional, but kept for consistency)
    open_windows = []

    # Create the first window
    viewer = ImageViewer(window_list=open_windows)
    viewer.show()

    # Check for file arguments (CLI support)
    # Note: On macOS "Open With", FileOpen event handles it.
    # Terminal args are handled here.
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if not file_path.startswith("-"):
             QTimer.singleShot(100, lambda: viewer.open_file(file_path))

    sys.exit(app.exec())