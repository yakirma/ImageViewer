import sys
import os
import platform
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QFileDialog, QMessageBox, QSplitter
from PyQt6.QtCore import Qt, QTimer, QEvent, QSize, pyqtSignal, QPoint
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QColor, QPalette
from image_viewer import ImageViewer
from single_instance import SingleInstance

# Define Custom Application Class at module level
class ImageViewerApp(QApplication):
    def event(self, event):
        if event.type() == QEvent.Type.FileOpen:
            # Handle macOS FileOpen event (Open With...)
            file_path = event.file()
            if file_path:
                # Open immediately without delay to avoid flashes
                self.open_file_event(file_path)
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

    # Single instance check
    single_instance = SingleInstance('ImageViewer')
    
    if not single_instance.check_and_connect():
        # Secondary instance: send files to primary and exit
        if len(sys.argv) > 1:
            single_instance.send_files(sys.argv[1:])
        sys.exit(0)
    
    # Primary instance: continue as normal
    # Central list to manage open windows (optional, but kept for consistency)
    open_windows = []

    # Create the first window
    viewer = ImageViewer(window_list=open_windows)
    viewer.show()
    
    # Connect signal to handle files from secondary instances
    def handle_received_files(files):
        """Open files received from secondary instances in new windows"""
        for file_path in files:
            if os.path.isfile(file_path):
                # Create new window for each file
                new_viewer = ImageViewer(window_list=open_windows)
                new_viewer.show()
                QTimer.singleShot(100, lambda f=file_path, v=new_viewer: v.open_file(f))
    
    single_instance.files_received.connect(handle_received_files)

    # Check for file arguments (CLI support)
    # Note: On macOS "Open With", FileOpen event handles it.
    # Terminal args are handled here.
    if len(sys.argv) > 1:
        file_paths = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
        if file_paths:
            # Open first file in primary window
            viewer.open_file(file_paths[0])
            
            # Open additional files in new windows
            for additional_file in file_paths[1:]:
                if os.path.exists(additional_file):
                    new_viewer = ImageViewer(window_list=open_windows)
                    new_viewer.show()
                    new_viewer.open_file(additional_file)

    sys.exit(app.exec())