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
    def __init__(self, argv):
        super().__init__(argv)
        self.file_buffer = []
        self.open_timer = QTimer()
        self.open_timer.setSingleShot(True)
        self.open_timer.timeout.connect(self._flush_file_buffer)

    def event(self, event):
        if event.type() == QEvent.Type.FileOpen:
            file_path = event.file()
            if file_path:
                self.file_buffer.append(file_path)
                self.open_timer.start(50)  # Wait 50ms for more files
            return True
        return super().event(event)
        
    def _flush_file_buffer(self):
        if not self.file_buffer:
            return
            
        files = list(self.file_buffer)
        self.file_buffer.clear()
        
        top_widgets = self.topLevelWidgets()
        for widget in top_widgets:
            if isinstance(widget, ImageViewer):
                widget.open_files(files)
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
        """Open files received from secondary instances in the primary window"""
        if files:
            viewer.open_files(files)
            viewer.raise_()
            viewer.activateWindow()
    
    single_instance.files_received.connect(handle_received_files)

    # Check for file arguments (CLI support)
    if len(sys.argv) > 1:
        file_paths = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
        if file_paths:
            viewer.open_files(file_paths)

    sys.exit(app.exec())