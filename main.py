import sys
import os
import platform

# Fix for WinError 1114 on Windows when loading torch
if platform.system() == "Windows":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QFileDialog, QMessageBox, QSplitter
from PyQt6.QtCore import Qt, QTimer, QEvent, QSize, pyqtSignal, QPoint
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QColor, QPalette
from image_viewer import ImageViewer
from single_instance import SingleInstance, absolute_paths

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

def _default_log_path():
    """Per-user log file location (e.g. ~/Library/Application Support/ImageViewer
    on macOS, %LOCALAPPDATA%/ImageViewer on Windows, ~/.local/share/... on Linux)."""
    from PyQt6.QtCore import QStandardPaths
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".imageviewer")
    return os.path.join(base, "imageviewer.log")


def install_exception_hook(app, log_path=None, show_notice=True):
    """Install a sys.excepthook that reports uncaught exceptions instead of
    letting PyQt6 abort the process (PyQt calls qFatal() for exceptions
    escaping a slot when no custom excepthook is set).

    Each exception is printed to stderr and appended to a rotating log file
    (``log_path``; default: per-user app-data dir, None-safe). When
    ``show_notice`` is True, a status-bar message is shown on the active
    window and a non-modal QMessageBox is shown at most once per distinct
    error (type + message), so a slot failing on every repaint doesn't spam
    dialogs. Not installed under pytest: only the __main__ block calls this.

    Returns the hook function (also exposed as ``hook.seen`` for the set of
    already-notified error signatures).
    """
    import logging
    import traceback
    from logging.handlers import RotatingFileHandler

    logger = logging.getLogger("ImageViewer.uncaught")
    logger.propagate = False
    for old_handler in list(logger.handlers):
        logger.removeHandler(old_handler)
        old_handler.close()
    try:
        path = log_path or _default_log_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=2, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)
    except Exception:
        pass  # Logging to file is best effort; stderr still gets the traceback.

    seen = set()
    boxes = []

    def _notify(summary, details, first_time):
        try:
            win = app.activeWindow()
            if win is None:
                win = next((w for w in app.topLevelWidgets()
                            if isinstance(w, QMainWindow) and w.isVisible()), None)
            if isinstance(win, QMainWindow) and win.statusBar() is not None:
                win.statusBar().showMessage(f"Error: {summary}", 10000)
            if not first_time:
                return
            # One reusable, parentless (Python-owned) box; never stack dialogs.
            if boxes and boxes[0].isVisible():
                return
            if not boxes:
                box = QMessageBox()
                box.setIcon(QMessageBox.Icon.Warning)
                box.setWindowTitle("Unexpected Error")
                box.setStandardButtons(QMessageBox.StandardButton.Ok)
                box.setWindowModality(Qt.WindowModality.NonModal)
                boxes.append(box)
            box = boxes[0]
            box.setText("An unexpected error occurred. The application will keep running, "
                        "but the last action may not have completed.\n\n" + summary)
            box.setDetailedText(details)
            box.show()
        except Exception:
            pass

    def hook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        details = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        try:
            sys.stderr.write("Uncaught exception:\n" + details)
            sys.stderr.flush()
        except Exception:
            pass
        try:
            logger.error("Uncaught exception:\n%s", details.rstrip())
        except Exception:
            pass
        if not show_notice:
            return
        summary = f"{exc_type.__name__}: {exc_value}"
        signature = (exc_type, str(exc_value))
        first_time = signature not in seen
        seen.add(signature)
        # Defer UI work: the hook may run mid-paint/mid-event.
        QTimer.singleShot(0, lambda: _notify(summary, details, first_time))

    hook.seen = seen
    sys.excepthook = hook
    return hook


if __name__ == "__main__":
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    
    # Initialize Application ONCE
    app = ImageViewerApp(sys.argv)
    app.setOrganizationName("ImageViewer")
    app.setApplicationName("ImageViewer")
    app.setWindowIcon(QIcon("assets/app_icon.png"))

    # Report uncaught exceptions (incl. from Qt slots) instead of aborting.
    install_exception_hook(app)

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
        file_paths = absolute_paths(sys.argv[1:])
        if file_paths:
            viewer.open_files(file_paths)

    sys.exit(app.exec())