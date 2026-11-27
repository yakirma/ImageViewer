import sys
import os
import platform

# Fix for WinError 1114 on Windows when loading torch
if platform.system() == "Windows":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QFileDialog, QMessageBox, QSplitter
from PyQt6.QtCore import Qt, QTimer, QEvent, QSize, pyqtSignal, QPoint, QObject
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QColor, QPalette
from PyQt6 import sip
from image_viewer import ImageViewer
from single_instance import SingleInstance, absolute_paths

def _is_live(window):
    """True for an ImageViewer whose C++ side exists and that is not closed/hidden."""
    try:
        return (isinstance(window, ImageViewer) and not sip.isdeleted(window)
                and window.isVisible())
    except RuntimeError:
        return False


def _is_empty(window):
    """True if the window shows no image (single view or montage)."""
    if getattr(window, 'current_file_path', None):
        return False
    return not any(getattr(label, 'file_path', None)
                   for label in getattr(window, 'montage_labels', []) or [])


class FileOpenBatcher(QObject):
    """Opens files from outside the app (macOS FileOpen events, the
    single-instance hand-off) without replacing what the user is viewing.

    Finder delivers one FileOpen event per selected file, in quick
    succession. Paths are buffered and, once no new path has arrived for
    ``delay_ms``, the whole batch goes to ONE destination window through a
    single ``open_files()`` call (2-3 files -> montage). The destination is
    the most recently active ImageViewer if it is empty, else any visible
    empty ImageViewer in ``window_list``, else a new window -- an occupied
    window is never overwritten (see "Fix Open With to spawn new windows").

    Files that arrive before ``attach()`` (cold start, before the first
    window exists) are kept and delivered once it has been called.
    """

    RETRY_MS = 100

    def __init__(self, delay_ms=50, parent=None, window_list=None):
        super().__init__(parent)
        self.delay_ms = delay_ms
        self.window_list = window_list
        self.file_buffer = []
        self.last_active = None
        self.open_timer = QTimer(self)
        self.open_timer.setSingleShot(True)
        self.open_timer.timeout.connect(self.flush)
        app = QApplication.instance()
        if app is not None:
            # Track the most recently activated ImageViewer: when Finder
            # triggers the open, activeWindow() is usually None.
            app.installEventFilter(self)

    def attach(self, window_list):
        """Start delivering into ``window_list`` (the app's window list)."""
        self.window_list = window_list
        if self.file_buffer:
            self.open_timer.start(0)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.WindowActivate and isinstance(obj, ImageViewer):
            self.last_active = obj
        return False

    def add(self, file_path):
        if not file_path:
            return
        self.file_buffer.append(file_path)
        self.open_timer.start(self.delay_ms)  # Wait for more files

    def _destination(self):
        """Existing window to open into, or None to create a new one."""
        app = QApplication.instance()
        active = app.activeWindow() if app is not None else None
        if not _is_live(active):
            active = self.last_active
        if _is_live(active) and active in self.window_list and _is_empty(active):
            return active
        for win in list(self.window_list):
            if _is_live(win) and _is_empty(win):
                return win
        return None

    def open_now(self, files):
        """Open ``files`` together in one destination window; returns it."""
        files = list(files)
        if not files:
            return None
        target = self._destination()
        if target is None:
            reference = self.last_active if _is_live(self.last_active) else None
            target = ImageViewer(window_list=self.window_list)
            if reference is not None:
                target.move(reference.pos() + QPoint(30, 30))
        target.show()
        target.raise_()
        target.activateWindow()
        self.last_active = target
        target.open_files(files)
        return target

    def flush(self):
        if not self.file_buffer:
            return
        if self.window_list is None:
            # Cold start: the first window is not set up yet. Keep the files.
            self.open_timer.start(self.RETRY_MS)
            return
        files = list(self.file_buffer)
        self.file_buffer.clear()
        self.open_now(files)


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
    
    # Check for file arguments (CLI support)
    if len(sys.argv) > 1:
        file_paths = absolute_paths(sys.argv[1:])
        if file_paths:
            viewer.open_files(file_paths)

    # Finder opens (incl. ones queued during startup) and files from
    # secondary instances share one rule: open in an empty window, never
    # over the image the user is viewing.
    app.file_opener.attach(open_windows)

    def handle_received_files(files):
        """Open files received from secondary instances."""
        if files:
            app.file_opener.open_now(files)

    single_instance.files_received.connect(handle_received_files)

    sys.exit(app.exec())