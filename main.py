import sys
import os
import platform

# Fix for WinError 1114 on Windows when loading torch
if platform.system() == "Windows":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QFileDialog, QMessageBox, QSplitter
from PyQt6.QtCore import Qt, QTimer, QEvent, QSize, pyqtSignal, QPoint, QObject
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QColor, QPalette
from image_viewer import ImageViewer
from single_instance import SingleInstance, absolute_paths

class FileOpenBatcher(QObject):
    """Collects paths from macOS FileOpen events and opens them together.

    Finder delivers one FileOpen event per selected file, in quick
    succession. Paths are buffered and, once no new path has arrived for
    ``delay_ms``, handed to a single ``ImageViewer.open_files()`` call so
    2-3 files become one montage instead of one window per file. If no
    ImageViewer window exists yet (FileOpen events can arrive during a cold
    start), the paths are kept and delivery is retried until one does.
    """

    RETRY_MS = 100

    def __init__(self, delay_ms=50, parent=None):
        super().__init__(parent)
        self.delay_ms = delay_ms
        self.file_buffer = []
        self.open_timer = QTimer(self)
        self.open_timer.setSingleShot(True)
        self.open_timer.timeout.connect(self.flush)

    def add(self, file_path):
        if not file_path:
            return
        self.file_buffer.append(file_path)
        self.open_timer.start(self.delay_ms)  # Wait for more files

    @staticmethod
    def _target_window():
        app = QApplication.instance()
        if app is None:
            return None
        active = app.activeWindow()
        if isinstance(active, ImageViewer):
            return active
        viewers = [w for w in app.topLevelWidgets() if isinstance(w, ImageViewer)]
        visible = [w for w in viewers if w.isVisible()]
        return (visible or viewers or [None])[0]

    def flush(self):
        if not self.file_buffer:
            return
        target = self._target_window()
        if target is None:
            # Cold start: no window yet. Keep the files and try again.
            self.open_timer.start(self.RETRY_MS)
            return
        files = list(self.file_buffer)
        self.file_buffer.clear()
        target.open_files(files)
        target.raise_()
        target.activateWindow()


# Define Custom Application Class at module level
class ImageViewerApp(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.file_opener = FileOpenBatcher(parent=self)

    def event(self, event):
        if event.type() == QEvent.Type.FileOpen:
            self.file_opener.add(event.file())
            return True
        return super().event(event)


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