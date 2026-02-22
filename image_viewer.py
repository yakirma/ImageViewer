import os
import sys
import re
import shutil
import json
import tempfile
from datetime import datetime

import numpy as np
from PyQt6.QtCore import Qt, QEvent, QPoint, QPointF, QTimer, QThread, pyqtSignal, QUrl, QMimeData, QDir
from PyQt6.QtGui import QAction, QPixmap, QImage, QIcon, QDesktopServices, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QWidget,
    QFrame,
    QFileDialog,
    QStatusBar,
    QPushButton,
    QStackedWidget,
    QMessageBox,
    QComboBox,
    QToolBar,
    QSizePolicy,
    QSlider,
    QGridLayout,
    QHBoxLayout,
    QSpinBox,
    QToolButton,
    QProgressBar,
    QDialog,
)
import matplotlib.cm as cm

from widgets import ZoomableDraggableLabel, InfoPane, MathTransformPane, ZoomSettingsDialog, HistogramWidget, \
    ThumbnailPane, SharedViewState, FileExplorerPane, PointCloudViewer
from image_handler import ImageHandler
import settings

try:
    import requests
except ImportError:
    requests = None

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (QPoint, QPointF)):
            return [obj.x(), obj.y()]
        return super(NumpyEncoder, self).default(obj)

__version__ = "1.1.0"

class CheckForUpdates(QThread):
    update_available = pyqtSignal(str, str) # version, url

    def run(self):
        if requests is None:
            return

        try:
            repo = "yakirma/ImageViewer"
            url = f"https://api.github.com/repos/{repo}/releases/latest"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                latest_version_str = data['tag_name'].lstrip('v')
                html_url = data['html_url']
                
                # Semantic version comparison
                # Parse "1.0.4" -> (1, 0, 4)
                def parse_version(v):
                    return tuple(map(int, (v.split("."))))

                local_ver = parse_version(__version__)
                remote_ver = parse_version(latest_version_str)

                if remote_ver > local_ver:
                    self.update_available.emit(latest_version_str, html_url)
        except Exception as e:
            print(f"Update check failed: {e}")


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)



class DA3Worker(QThread):
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, int) # current_frame, total_frames
    download_progress = pyqtSignal(str, float) # filename, percentage
    
    class QtTqdm:
        def __init__(self, iterable=None, total=None, desc=None, unit=None, **kwargs):
            self.total = total
            self.n = 0
            self.desc = desc or ""
            self.worker = kwargs.get('worker')
            
        def update(self, n=1):
            self.n += n
            if self.total and self.worker:
                percent = (self.n / self.total) * 100
                self.worker.download_progress.emit(self.desc, percent)
                
        def close(self):
            if self.worker:
                self.worker.download_progress.emit(self.desc, 100.0)
                
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()

    def __init__(self, image_path, model_name):
        super().__init__()
        self.image_path = image_path
        self.model_name = model_name
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            import torch
            from depth_anything_3.api import DepthAnything3
            from PIL import Image
            import numpy as np
            import tifffile
            import cv2
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
            
            # Load model with download progress tracking
            model = DepthAnything3.from_pretrained(
                self.model_name, 
                tqdm_class=lambda **kwargs: self.QtTqdm(worker=self, **kwargs)
            )
            model.eval().to(device)
            
            base, ext = os.path.splitext(self.image_path)
            ext = ext.lower()
            video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.gif']
            
            if ext in video_exts:
                # Video Mode
                cap = cv2.VideoCapture(self.image_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                out_dir = f"{base}_DEPTH"
                os.makedirs(out_dir, exist_ok=True)
                
                frame_idx = 0
                while cap.isOpened() and not self._is_cancelled:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Convert BGR (OpenCV) to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Run inference
                    pred = model.inference([frame_rgb])
                    depth_map = pred.depth[0]
                    
                    # Save with 1-based indexing for user readability
                    out_path = os.path.join(out_dir, f"frame_{frame_idx+1}.tiff")
                    tifffile.imwrite(out_path, depth_map)
                    
                    frame_idx += 1
                    self.progress.emit(frame_idx, total_frames)
                
                cap.release()
                if self._is_cancelled:
                    self.failed.emit("Cancelled by user")
                else:
                    self.finished.emit(out_dir)
            else:
                # Single Image Mode
                img = Image.open(self.image_path).convert('RGB')
                img_np = np.array(img)
                
                # Run inference
                pred = model.inference([img_np])
                depth_map = pred.depth[0]
                
                # Save depth map next to original file
                out_path = f"{base}_DEPTH.tiff"
                
                tifffile.imwrite(out_path, depth_map)
                self.finished.emit(out_path)
        except Exception as e:
            self.failed.emit(str(e))

class DA3ModelDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Depth Anything V3 Model")
        self.layout = QVBoxLayout(self)
        
        self.layout.addWidget(QLabel("Select a model to generate depth map:"))
        self.combo = QComboBox()
        self.combo.addItems([
            "depth-anything/DA3-SMALL",
            "depth-anything/DA3-BASE",
            "depth-anything/DA3-LARGE",
            "depth-anything/DA3-GIANT"
        ])
        self.layout.addWidget(self.combo)
        
        self.warning_label = QLabel("Note: First run will download the model weights (may take a while).")
        self.warning_label.setStyleSheet("color: gray; font-style: italic;")
        self.warning_label.setWordWrap(True)
        self.layout.addWidget(self.warning_label)
        
        btn_layout = QHBoxLayout()
        self.ok_btn = QPushButton("Generate")
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)
        self.layout.addLayout(btn_layout)
        
    def get_model(self):
        return self.combo.currentText()

class ImageViewer(QMainWindow):
    view_clipboard = None  # Class-level clipboard for cross-window sharing

    def __init__(self, window_list):
        super().__init__()
        self.setWindowTitle("Image Viewer")
        self.window_list = window_list
        self.window_list.append(self)
        
        # Enable Drag and Drop
        self.setAcceptDrops(True)

        self.image_handler = ImageHandler()
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self._video_timer_timeout)
        self.is_slider_pressed = False
        
        self.shared_state = SharedViewState()

        self.current_math_expression = None # Persist math across video frames
        self.recent_files = settings.load_recent_files()
        self.last_raw_settings = None
        self.current_file_path = None
        self.montage_shared_state = None
        self.montage_labels = []
        self.active_label = None

        screen_geometry = self.screen().geometry()
        self.resize(screen_geometry.width() // 2, screen_geometry.height() // 2)
        self.move(screen_geometry.center() - self.rect().center())

        self.stacked_widget = QStackedWidget()
        # Give the central widget priority in horizontal space distribution (Stretch=1)
        policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        policy.setHorizontalStretch(1)
        self.stacked_widget.setSizePolicy(policy)
        self.stacked_widget.setAcceptDrops(True)
        self.setCentralWidget(self.stacked_widget)
        self._last_center_width = 0
        self._updating_from_thumbnail = False  # Flag to prevent circular refresh
        self._dock_sizes = {}  # Track dock sizes to preserve layout
        # Ensure Left Dock takes precedence on the left side
        self.setCorner(Qt.Corner.TopLeftCorner, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.setCorner(Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.setCorner(Qt.Corner.TopRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)
        self.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)
        
        # Optimize Layout Behavior
        self.setDockOptions(self.dockOptions() | QMainWindow.DockOption.AnimatedDocks | QMainWindow.DockOption.AllowNestedDocks)

        self.current_colormap = "gray"
        self.overlay_mode = 0 # 0=Hidden, 1=Basename, 2=Full Path
        self.zoom_settings = {"zoom_speed": 1.1, "zoom_in_interp": "Nearest", "zoom_out_interp": "Smooth"}

        self._create_welcome_screen()
        self._create_image_display()
        self._create_montage_view()

        # Start on welcome screen
        self.stacked_widget.setCurrentIndex(0)



        self._create_menus_and_toolbar()
        self._create_file_explorer_pane()
        self._create_info_pane()
        self._create_math_transform_pane()
        self._create_histogram_window()
        self._create_thumbnail_pane()

        # Force initial docking ratio if possible
        self.resizeDocks([self.file_explorer_pane], [225], Qt.Orientation.Horizontal)

        # Install event filter at the very end. 
        # By only catching Drop here and allowing DragEnter/Move to propagate,
        # we ensure side panes can show their own cursors/acceptances while central area
        # remains orchestrated.
        QApplication.instance().installEventFilter(self)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0) # Indeterminate
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)


        self.zoom_status_label = QLabel("Zoom: 100%")
        self.status_bar.addPermanentWidget(self.zoom_status_label)
        
        self._check_for_updates()

    def _check_for_updates(self):
        self.update_checker = CheckForUpdates()
        self.update_checker.update_available.connect(self._on_update_available)
        self.update_checker.start()

    def _on_update_available(self, version, url):
        # Create a clickable message
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Update Available")
        msg_box.setText(f"A new version ({version}) is available!")
        msg_box.setInformativeText(f"Current version: {__version__}\n\nDo you want to view the release page?")
        msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)
        
        if msg_box.exec() == QMessageBox.StandardButton.Yes:
            QDesktopServices.openUrl(QUrl(url))


    def _get_percentiles_from_limits(self, data, limits):
        if data is None or limits is None:
            return (0.0, 100.0)
        
        # Use a flattened view for percentile calculation
        flat_data = data.ravel()
        
        # Use full data for precise min/max detection to preserve "Min-Max" intent
        img_min = flat_data.min()
        img_max = flat_data.max()
        
        # Handle anchors for perfect Min-Max inheritance
        if limits[0] <= img_min:
            p_min = 0.0
        elif limits[0] >= img_max:
            p_min = 100.0
        else:
            # Sampling for performance on very large images in the middle range
            sample = np.random.choice(flat_data, 1_000_000, replace=False) if flat_data.size > 1_000_000 else flat_data
            # Using < for the lower end matches np.percentile logic (percent strictly below)
            p_min = np.mean(sample < limits[0]) * 100.0

        if limits[1] >= img_max:
            p_max = 100.0
        elif limits[1] <= img_min:
            p_max = 0.0
        else:
            sample = np.random.choice(flat_data, 1_000_000, replace=False) if flat_data.size > 1_000_000 else flat_data
            # Using <= for the upper end matches np.percentile logic (percent below or at)
            p_max = np.mean(sample <= limits[1]) * 100.0
            
        return (p_min, p_max)

    def _get_limits_from_percentiles(self, data, percentiles):
        if data is None or percentiles is None:
            return (0.0, 1.0)
            
        flat_data = data.ravel()
        
        # Sampling for performance on very large images
        if flat_data.size > 1_000_000:
            sample = np.random.choice(flat_data, 1_000_000, replace=False)
        else:
            sample = flat_data
            
        try:
            limits = np.nanpercentile(sample, percentiles)
            return (float(limits[0]), float(limits[1]))
        except Exception:
            return (0.0, 1.0)

    def _create_welcome_screen(self):
        self.welcome_widget = QWidget()
        welcome_layout = QVBoxLayout(self.welcome_widget)
        welcome_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label = QLabel("Image Viewer")
        font = title_label.font()
        font.setPointSize(24)
        title_label.setFont(font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        open_button = QPushButton("Open Image")
        open_button.clicked.connect(self.open_image_dialog)

        drag_drop_label = QLabel("(Drag & Drop files here or Paste [Ctrl+V] from Clipboard)")
        drag_drop_label.setStyleSheet("color: gray; font-style: italic;")
        drag_drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        welcome_layout.addWidget(title_label)
        welcome_layout.addWidget(open_button)
        welcome_layout.addWidget(drag_drop_label)
        self.stacked_widget.addWidget(self.welcome_widget)

    def _create_image_display(self):
        self.image_label = ZoomableDraggableLabel()
        self.image_label.open_companion_depth.connect(self._on_open_companion_depth)
        self.image_display_container = QWidget()
        policy = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Expanding)
        policy.setHorizontalStretch(100)
        self.image_display_container.setSizePolicy(policy)
        image_display_layout = QVBoxLayout(self.image_display_container)
        image_display_layout.addWidget(self.image_label)
        self.image_label.setSizePolicy(policy) # Also on the label itself


        self.stacked_widget.addWidget(self.image_display_container)
        self.apply_zoom_settings()  # Apply default settings immediately

    def _create_montage_view(self):
        self.montage_widget = QWidget()
        policy = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Expanding)
        policy.setHorizontalStretch(100)
        self.montage_widget.setSizePolicy(policy)
        self.montage_widget.setAcceptDrops(True)
        
        # Container Layout: Label + Grid
        container_layout = QVBoxLayout(self.montage_widget)
        container_layout.setContentsMargins(10, 10, 10, 10)
        container_layout.setSpacing(10)
        
        # Instruction Label
        self.montage_instruction_label = QLabel("Drag additional image files here to add them to the view")
        self.montage_instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.montage_instruction_label.setStyleSheet("color: gray; font-style: italic; margin-bottom: 5px;")
        container_layout.addWidget(self.montage_instruction_label)
        
        # Inner Grid Widget for actual images
        self.montage_grid_widget = QWidget()
        self.montage_grid_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.montage_layout = QGridLayout(self.montage_grid_widget) # Use inner widget's layout
        container_layout.addWidget(self.montage_grid_widget)
        
        self.stacked_widget.addWidget(self.montage_widget)

    def _create_menus_and_toolbar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        open_action = QAction(QIcon.fromTheme("document-open"), "&Open", self)
        open_action.triggered.connect(self.open_image_dialog)
        file_menu.addAction(open_action)
        
        paste_action = QAction("&Paste", self)
        paste_action.setShortcut(QKeySequence.StandardKey.Paste)
        paste_action.triggered.connect(self.paste_from_clipboard)
        file_menu.addAction(paste_action)

        export_action = QAction("&Export Current View to...", self)
        export_action.triggered.connect(self.export_current_view)
        file_menu.addAction(export_action)
        file_menu.addSeparator()
        self.recent_files_menu = file_menu.addMenu("Recent Files")
        self._update_recent_files_menu()

        help_menu = menu_bar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        self.explorer_action = QAction(QIcon.fromTheme("folder"), "File Explorer", self)
        self.explorer_action.triggered.connect(self.toggle_file_explorer_pane)
        toolbar.addAction(self.explorer_action)
        
        toolbar.addAction(open_action)
        
        settings_action = QAction(QIcon.fromTheme("preferences-system"), "Settings", self)
        settings_action.triggered.connect(self.open_zoom_settings)
        toolbar.addAction(settings_action)
        restore_action = QAction(QIcon(resource_path("assets/icons/expand.png")), "Restore View", self)
        restore_action.triggered.connect(self.restore_image_view)
        toolbar.addAction(restore_action)
        
        reset_action = QAction(QIcon(resource_path("assets/icons/redo.png")), "Reset Image", self)
        reset_action.triggered.connect(self.reset_image_full)
        toolbar.addAction(reset_action)

        self.colormap_combo = QComboBox(self)
        self.colormap_combo.addItems([
            "gray", "turbo", "viridis", "magma", "inferno", "plasma",  # Standard / Uniform
            "bone", "hot", "cool", "copper", "pink",                   # Classic Sequential
            "seismic", "coolwarm", "bwr", "spectral",                 # Diverging
            "hsv", "twilight",                                        # Cyclic
            "flow",                                                   # Optical Flow
            "jet", "rainbow", "ocean", "terrain"                      # Misc
        ])
        self.colormap_combo.setCurrentText(self.current_colormap)
        self.colormap_combo.currentTextChanged.connect(self.set_colormap)
        toolbar.addWidget(QLabel("Colormap:", self))
        toolbar.addWidget(self.colormap_combo)

        # Show Colorbar Button
        self.colorbar_action = QAction(QIcon(resource_path("assets/icons/colorbar.png")), "Colorbar", self)
        self.colorbar_action.setToolTip("Show Colorbar Legend")
        self.colorbar_action.setCheckable(True)
        self.colorbar_action.toggled.connect(self.toggle_colorbar)
        toolbar.addAction(self.colorbar_action)
        
        # Channel Selector
        self.channel_combo = QComboBox(self)
        self.channel_combo.addItem("Default")
        self.channel_combo.currentTextChanged.connect(lambda: self.update_image_display(reset_view=False))
        toolbar.addWidget(QLabel("  Channel:", self))
        toolbar.addWidget(self.channel_combo)
        
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        
        self._create_video_toolbar()

        self.histogram_action = QAction(QIcon(resource_path("assets/icons/histogram.png")), "Histogram", self)
        self.histogram_action.triggered.connect(self.toggle_histogram_window)
        self.histogram_action.setEnabled(False)
        toolbar.addAction(self.histogram_action)

        self.da3_action = QAction(QIcon(resource_path("assets/icons/layers.png")), "DA3 Depth", self)
        self.da3_action.setToolTip("Generate Depth Map using Depth Anything V3")
        self.da3_action.triggered.connect(self.generate_da3_depth_map)
        toolbar.addAction(self.da3_action)

        # Custom "3D" Button with separate colors for '3' (Green) and 'D' (Red)
        self.threed_button = QWidget()
        self.threed_button.setFixedSize(40, 32)
        self.threed_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.threed_button.setToolTip("Open 3D View")
        
        container_layout = QHBoxLayout(self.threed_button)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        self.label_3d_3 = QLabel("3") # This line was part of the original code, but the snippet places it here.
                                      # I'm following the snippet's placement.
        self.label_3d_3.setStyleSheet("color: #00E676; font-weight: 900; font-size: 16px; font-family: 'Arial'; margin-right: -5px;")
        self.label_3d_3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.label_3d_d = QLabel("D")
        self.label_3d_d.setStyleSheet("color: #FF1744; font-weight: 900; font-size: 16px; font-family: 'Arial'; margin-left: -5px;")
        self.label_3d_d.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        container_layout.addWidget(self.label_3d_3)
        container_layout.addWidget(self.label_3d_d)
        
        # Make the whole container clickable by filtering events
        class ClickableFilter(QWidget):
            def __init__(self, parent_obj, callback):
                super().__init__()
                self.parent_obj = parent_obj
                self.callback = callback
            def eventFilter(self, source, event):
                if event.type() == QEvent.Type.MouseButtonPress:
                    # Respect parent enablement
                    if source.isEnabled():
                        self.callback()
                        return True
                return False
        
        self.threed_click_filter = ClickableFilter(self, self.open_3d_view)
        self.threed_button.installEventFilter(self.threed_click_filter)
        
        # Styling the container background
        self.threed_button.setStyleSheet("""
            QWidget {
                background-color: transparent;
                border-radius: 6px;
                border: none;
            }
            QWidget:hover {
                background-color: rgba(255, 255, 255, 30);
                border: none;
            }
            QWidget:disabled {
                opacity: 0.5;
            }
        """)
        
        toolbar.addWidget(self.threed_button)

        self.math_transform_action = QAction(QIcon.fromTheme("accessories-calculator"), "Math Transform", self)
        self.math_transform_action.triggered.connect(self.toggle_math_transform_pane)
        self.math_transform_action.setEnabled(False)
        toolbar.addAction(self.math_transform_action)
        self.info_action = QAction(QIcon.fromTheme("dialog-information"), "Image Info", self)
        self.info_action.triggered.connect(self.toggle_info_pane)
        self.info_action.setEnabled(False)
        toolbar.addAction(self.info_action)

        thumbnail_action = QAction(QIcon(resource_path("assets/icons/opened_images.png")), "Opened Images", self)
        thumbnail_action.triggered.connect(self.toggle_thumbnail_pane)
        toolbar.addAction(thumbnail_action)

    def update_channel_options(self):
        """Populate channel selector based on current image channels."""
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        
        image = None
        # Prefer active label data if available
        if self.active_label and self.active_label.original_data is not None:
             image = self.active_label.original_data
        elif self.image_handler.original_image_data is not None:
             image = self.image_handler.original_image_data
        
        if image is None:
            self.channel_combo.addItem("Default")
            self.channel_combo.blockSignals(False)
            return

        # Check if this is an NPZ file with multiple keys (FIXME: Need to track NPZ keys per label)
        # For now, fallback to image handler if active label matches handler
        is_handler_source = (image is self.image_handler.original_image_data)
        
        if is_handler_source and hasattr(self.image_handler, 'npz_keys') and len(self.image_handler.npz_keys) > 1:
            from PyQt6.QtGui import QStandardItemModel, QStandardItem, QColor
            model = QStandardItemModel()
            for key in self.image_handler.npz_keys.keys():
                item = QStandardItem(key)
                is_valid = self.image_handler.npz_keys[key]
                if not is_valid:
                    item.setEnabled(False)
                    item.setForeground(QColor('gray'))  # Gray out invalid keys
                model.appendRow(item)
            self.channel_combo.setModel(model)
            
            # Select current key
            if hasattr(self.image_handler, 'current_npz_key'):
                try:
                    index = list(self.image_handler.npz_keys.keys()).index(self.image_handler.current_npz_key)
                    self.channel_combo.setCurrentIndex(index)
                except ValueError:
                    pass
        else:
            # Standard channel options for regular images
            if image.ndim == 2:
                self.channel_combo.addItems(["Gray"])
            elif image.ndim == 3:
                channels = image.shape[2]
                if channels == 3:
                    # Standard RGB
                    self.channel_combo.addItems(["RGB", "R", "G", "B", "RG"])
                elif channels == 4:
                    # RGBA
                    self.channel_combo.addItems(["RGBA", "RGB", "RG", "R", "G", "B", "A"])
                elif channels == 2:
                    # RG
                    self.channel_combo.addItems(["RG", "R", "G"])
                else:
                    self.channel_combo.addItem(f"{channels} Channels")
        
        self.channel_combo.blockSignals(False)

    def apply_channel_selection(self, image_data):
        """Slice image data based on selected channel option."""
        selection = self.channel_combo.currentText()
        
        # Handle NPZ key selection
        if hasattr(self.image_handler, 'npz_keys') and selection in self.image_handler.npz_keys:
            if self.image_handler.npz_keys[selection]:  # Valid key
                self.image_handler.original_image_data = self.image_handler.npz_data[selection]
                self.image_handler.current_npz_key = selection
                # Update dimensions
                if self.image_handler.original_image_data.ndim == 2:
                    self.image_handler.height, self.image_handler.width = self.image_handler.original_image_data.shape
                elif self.image_handler.original_image_data.ndim == 3:
                    self.image_handler.height, self.image_handler.width = self.image_handler.original_image_data.shape[:2]
                return self.image_handler.original_image_data
        
        if image_data is None or image_data.ndim < 3:
            return image_data
            
        if selection == "Default" or selection == "RGB" or selection == "RGBA":
            # Return full image (handle RGBA->RGB if needed elsewhere, but here we keep original)
             # Actually, if selection is RGB but image is RGBA, we might want to slice alpha out?
             # For now let's assume 'RGB' means full color view or explicit 3 channels
             if selection == "RGB" and image_data.shape[2] == 4:
                 return image_data[:, :, :3]
             return image_data
             
        # Single Channels
        if selection == "R":
            return image_data[:, :, 0]
        elif selection == "G":
            return image_data[:, :, 1]
        elif selection == "B":
            return image_data[:, :, 2]
        elif selection == "A" and image_data.shape[2] >= 4:
            return image_data[:, :, 3]
            
        # Multi-Channel combinations
        if selection == "RG":
            # Return true 2-channel data to enable flow visualization auto-detection
            return image_data[:, :, :2]
            
        return image_data

    def update_image_display(self, reset_view=False):
        """Centralized method to update image display with channel selection and transforms."""
        if not self.active_label:
            return
        
        # 1. Get the base data (original or current video frame)
        is_montage = (self.stacked_widget.currentWidget() == self.montage_widget)
        
        if is_montage and self.active_label:
             # If the active label IS the main video being played, use image_handler data
             if self.active_label.file_path == self.image_label.file_path and self.image_handler.is_video:
                 data = self.image_handler.original_image_data
             else:
                 data = self.active_label.pristine_data
        else:
             data = self.image_handler.original_image_data
        
        if data is None:
            return
        
        # 2. Apply Channel Selection
        sliced_data = self.apply_channel_selection(data)
        
        # Enable 3D View for single channel images or RGB images with a depth companion
        is_single_channel = (sliced_data.ndim == 2) or (sliced_data.ndim == 3 and sliced_data.shape[2] == 1)
        
        has_depth_companion = False
        if not is_single_channel and self.active_label and getattr(self.active_label, 'file_path', None):
            base, _ = os.path.splitext(self.active_label.file_path)
            depth_path = f"{base}_DEPTH.tiff"
            depth_dir = f"{base}_DEPTH"
            if os.path.exists(depth_path) or os.path.isdir(depth_dir):
                has_depth_companion = True
                
        is_3d_enabled = is_single_channel or has_depth_companion
        
        if hasattr(self, 'threed_button'):
             self.threed_button.setEnabled(is_3d_enabled)
             # Update styling to look disabled/enabled
             if is_3d_enabled:
                 self.threed_button.setToolTip("Open 3D View" + (" (using Depth Companion)" if has_depth_companion else ""))
                 # Restore colors
                 self.label_3d_3.setStyleSheet("color: #00E676; font-weight: 900; font-size: 16px; font-family: 'Arial'; margin-right: -5px;")
                 self.label_3d_d.setStyleSheet("color: #FF1744; font-weight: 900; font-size: 16px; font-family: 'Arial'; margin-left: -5px;")
             else:
                 self.threed_button.setToolTip("3D View available only for single-channel or depth map companions")
                 # Set to gray
                 self.label_3d_3.setStyleSheet("color: #808080; font-weight: 900; font-size: 16px; font-family: 'Arial'; margin-right: -5px;")
                 self.label_3d_d.setStyleSheet("color: #808080; font-weight: 900; font-size: 16px; font-family: 'Arial'; margin-left: -5px;")
        
        # 3. Auto-select flow mode or Reset to Default
        # Check if we were in 'flow' mode but now data is not compatible (e.g. RGB)
        # OR if we switched from RG (2-channel) to something else
        if sliced_data.ndim == 3 and sliced_data.shape[2] == 2:
             if self.colormap_combo.currentText() != "flow":
                 self.colormap_combo.setCurrentText("flow")
        elif self.colormap_combo.currentText() == "flow":
             # We entered a non-flow compatible state but still have 'flow' selected.
             # Reset to default 'gray' as requested.
             self.colormap_combo.setCurrentText("gray")
        
        # Capture inspection data (raw values) before any visualization conversion
        # With new architecture, sliced_data IS the raw data passed to the label
        inspection_data = sliced_data
        
        # 4. [REMOVED] Flow Visualization is now handled dynamically in ZoomableDraggableLabel.apply_colormap
        # This allows histogram contrast limits to affect the flow rendering (normalization)
        
        # 5. Set Data to Label (as pristine base)
        # Pass both the display data (sliced_data) and the raw values (inspection_data)
        # For Montage, we do NOT want to overwrite 'pristine_data' because it's the only source of the full (multi-channel) image
        update_pristine = not is_montage
        
        # If active_label is different from self.image_label (ie. Montage), use it
        target_label = self.active_label if self.active_label else self.image_label
        target_label.set_data(sliced_data, reset_view=reset_view, is_pristine=update_pristine, inspection_data=inspection_data)
        
        # Safeguard: If we are displaying RGB data, ensure colormap is gray 
        # (otherwise it might extract channel 0 if colormap is stuck on something else and is_rgb logic fails)
        if sliced_data.ndim == 3 and sliced_data.shape[2] in [3, 4] and self.colormap_combo.currentText() != "gray" and self.colormap_combo.currentText() != "flow":
             self.colormap_combo.blockSignals(True)
             self.colormap_combo.setCurrentText("gray")
             self.colormap_combo.blockSignals(False)
             target_label.set_colormap("gray")
        
        # 6. Apply Math Transform if exists
        if self.current_math_expression:
            self.apply_math_transform(self.current_math_expression, from_update=True)
        
        # 7. Trigger repaint and histogram update
        target_label.repaint()
        self.update_histogram_data()
        
        # 8. Sync Montage labels for Video/Depth (Ensure all related labels update together)
        if is_montage and self.image_label.file_path and self.image_handler.is_video:
             video_file = self.image_label.file_path
             video_base, _ = os.path.splitext(video_file)
             depth_dir = f"{video_base}_DEPTH"
             
             frame_idx = self.image_handler.current_frame_index
             depth_frame_path = os.path.join(depth_dir, f"frame_{frame_idx+1}.tiff")
             
             for label in self.montage_labels:
                 if label == target_label:
                     continue
                     
                 # Case A: Sync the VIDEO label if it's not the target but we are playing video
                 if label.file_path == video_file:
                      label.set_data(self.image_handler.original_image_data, reset_view=False, is_pristine=False)
                      label.repaint()
                      
                 # Case B: Sync DEPTH labels
                 elif os.path.isdir(depth_dir) and label.file_path and label.file_path.startswith(depth_dir):
                      if os.path.exists(depth_frame_path) and label.file_path != depth_frame_path:
                           try:
                                from image_handler import ImageHandler
                                temp_h = ImageHandler()
                                temp_h.load_image(depth_frame_path)
                                if temp_h.original_image_data is not None:
                                     label.set_data(temp_h.original_image_data, reset_view=False, is_pristine=True)
                                     label.file_path = depth_frame_path
                                     label.repaint()
                           except Exception:
                                pass

        # 9. Update 3D View if open
        if hasattr(self, 'point_cloud_viewer') and self.point_cloud_viewer and self.point_cloud_viewer.isVisible():
             self.open_3d_view()
        


    def _create_video_toolbar(self):
        self.video_toolbar = QToolBar("Video Playback")
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, self.video_toolbar)
        self.video_toolbar.hide() 
        
        # Play/Pause
        self.play_action = QAction("Play", self) 
        self.play_action.setCheckable(True)
        self.play_action.triggered.connect(self._on_play_pause)
        self.video_toolbar.addAction(self.play_action)
        
        self.video_toolbar.addSeparator()

        # Prev
        prev_action = QAction("<", self)
        prev_action.triggered.connect(self._prev_frame)
        self.video_toolbar.addAction(prev_action)
        
        # Slider
        self.video_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_slider.setMinimumWidth(200)
        self.video_slider.valueChanged.connect(self._on_frame_slider_changed)
        self.video_slider.sliderPressed.connect(self._on_slider_pressed)
        self.video_slider.sliderReleased.connect(self._on_slider_released)
        self.video_toolbar.addWidget(self.video_slider)
        
        # Next
        next_action = QAction(">", self)
        next_action.triggered.connect(self._next_frame)
        self.video_toolbar.addAction(next_action)
        
        self.video_toolbar.addSeparator()

        # FPS
        self.video_toolbar.addWidget(QLabel(" FPS: "))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(30)
        self.fps_spin.valueChanged.connect(self._on_fps_changed)
        self.video_toolbar.addWidget(self.fps_spin)
        
        # Frame Counter
        self.frame_label = QLabel(" 0 / 0 ")
        self.video_toolbar.addWidget(self.frame_label)
        
    def _on_play_pause(self, checked):
        if checked:
            fps = self.fps_spin.value()
            interval = int(1000 / fps)
            self.playback_timer.start(interval)
            self.play_action.setText("Pause")
        else:
            self.playback_timer.stop()
            self.play_action.setText("Play")

    def _video_timer_timeout(self):
        if not self.image_handler.is_video:
            self.playback_timer.stop()
            return
            
        success = self.image_handler.get_next_frame()
        if success:
             self._update_video_view()
        else:
             # Loop or stop? Let's loop
             self.image_handler.seek_frame(0)
             self._update_video_view()
             
    def _update_video_view(self):
        # Block signals to prevent feedback loop from slider
        self.video_slider.blockSignals(True)
        self.video_slider.setValue(self.image_handler.current_frame_index)
        self.video_slider.blockSignals(False)
        
        self.frame_label.setText(f" {self.image_handler.current_frame_index + 1} / {self.image_handler.video_frame_count} ")
        
        # Update image without resetting view, but UPDATE pristine data for this new frame
        # This ensures subsequent math transforms use this frame as base
        self.update_image_display(reset_view=False)
        
    def _on_frame_slider_changed(self, value):
        if self.image_handler.is_video and not self.is_slider_pressed: # Only seek if not dragging
             self.image_handler.seek_frame(value)
             # Update image without resetting view
             self.update_image_display(reset_view=False)
             self.frame_label.setText(f" {self.image_handler.current_frame_index + 1} / {self.image_handler.video_frame_count} ")

    def _on_slider_pressed(self):
        self.is_slider_pressed = True
        self.was_playing_before_drag = self.playback_timer.isActive()
        self.playback_timer.stop()
        
    def _on_slider_released(self):
        self.is_slider_pressed = False
        if self.was_playing_before_drag:
             self.playback_timer.start()
        # Ensure frame is updated after release, in case valueChanged was blocked during drag
        self._on_frame_slider_changed(self.video_slider.value())

    def _on_fps_changed(self, value):
        if self.playback_timer.isActive():
             self.playback_timer.setInterval(int(1000/value))

    def _prev_frame(self):
        if self.image_handler.is_video:
             current = self.image_handler.current_frame_index
             self.image_handler.seek_frame(current - 1)
             self._update_video_view()

    def _next_frame(self):
        if self.image_handler.is_video:
             current = self.image_handler.current_frame_index
             self.image_handler.seek_frame(current + 1)
             self._update_video_view()

    def _create_info_pane(self):
        self.info_pane = InfoPane(self)
        self.info_pane.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.info_pane)
        self.info_pane.hide()
        self.info_pane.settings_changed.connect(self.reapply_raw_parameters)

    def _create_math_transform_pane(self):
        self.math_transform_pane = MathTransformPane(self)
        self.math_transform_pane.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.math_transform_pane.transform_requested.connect(self.apply_math_transform)
        self.math_transform_pane.restore_original_requested.connect(self.restore_original_image)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.math_transform_pane)
        self.math_transform_pane.hide()

    def _create_histogram_window(self):
        self.histogram_window = HistogramWidget()
        self.histogram_window.hide()
        self.histogram_window.use_visible_checkbox.toggled.connect(lambda: self.update_histogram_data(new_image=False))

    def _create_thumbnail_pane(self):
        self.thumbnail_pane = ThumbnailPane(self)
        self.thumbnail_pane.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.thumbnail_pane)
        self.thumbnail_pane.hide()
        self.thumbnail_pane.selection_changed.connect(self._on_thumbnail_selection_changed)
        self.thumbnail_pane.overlay_changed.connect(self._on_overlay_changed)

    def _create_file_explorer_pane(self):
        self.file_explorer_pane = FileExplorerPane(self)
        self.file_explorer_pane.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.file_explorer_pane.setMaximumWidth(600)  # Prevent excessive expansion
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.file_explorer_pane)
        self.file_explorer_pane.hide()
        self.file_explorer_pane.files_selected.connect(self._on_explorer_files_selected)
        
        # Configure supported extensions for filtering
        extensions = self.image_handler.raw_extensions + ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif', '.webp', '.heic', '.heif'] + self.image_handler.video_extensions
        # Create unique list

        extensions = list(set(extensions))
        
        # Create case-insensitive filters (add both lowercase and uppercase)
        ext_filters = []
        for ext in extensions:
            ext_filters.append('*' + ext.lower())
            ext_filters.append('*' + ext.upper())
            
        self.file_explorer_pane.set_supported_extensions(ext_filters)

        self.overlay_alphas = {} # (source_path, target_path) -> alpha
        self.overlay_cache = {}  # (source_path, target_path) -> QPixmap (resized to match target view)

    def display_montage(self, file_paths, is_manual=False):
        if is_manual and self.thumbnail_pane and file_paths:
            self.thumbnail_pane.add_to_manual_paths(file_paths)
            
        # Cache state of existing montage labels (for re-applying when refreshing montage)
        # This prevents losing colormap/contrast when adding 3rd image to an existing 2-image montage
        existing_states = {} # path -> state_dict
        for label in self.montage_labels:
            if hasattr(label, 'file_path') and label.file_path:
                existing_states[label.file_path] = label.get_view_state()

        # Reset active_label BEFORE clearing layout to prevent accessing deleted objects
        self.active_label = None
        
        while self.montage_layout.count():
            item = self.montage_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Force immediate processing of delete events to prevent ghosting
        QApplication.processEvents()
        
        self.montage_labels.clear()
        
        # Clear overlays for files not in the new selection
        if file_paths:
            file_paths_set = set(file_paths)
            overlays_to_remove = [(src, tgt) for src, tgt in self.overlay_alphas.keys() 
                                  if src not in file_paths_set or tgt not in file_paths_set]
            for pair in overlays_to_remove:
                del self.overlay_alphas[pair]
                if pair in self.overlay_cache:
                    del self.overlay_cache[pair]

        if not file_paths:
            # Thoroughly clear all state to prevent re-populating from ghost paths
            self.current_file_path = None
            self.image_label.current_pixmap = None
            self.image_label.original_data = None
            self.image_label.processed_data = None
            
            # Clear layout and show welcome/empty
            self.montage_labels.clear()
            self._update_active_view()
            self.stacked_widget.setCurrentWidget(self.welcome_widget)
            self.setWindowTitle("ImageViewer")
            self._refresh_thumbnail_pane() # Ensure thumbnails are cleared
            return

        if self.thumbnail_pane:
            self.thumbnail_pane.unremove_files(file_paths)

        self.montage_shared_state = SharedViewState()
        max_montage_zoom = 100.0

        row, col = 0, 0
        override_settings = getattr(self, '_temp_montage_override', None)
        
        for file_path in file_paths:
            # Handle NPZ key paths (format: "file.npz#key_name")
            npz_key_to_load = None
            actual_file_path = file_path
            if '#' in file_path:
                actual_file_path, npz_key_to_load = file_path.rsplit('#', 1)
            
            temp_handler = ImageHandler()
            basename = os.path.basename(actual_file_path)
            
            # [FIX] Clear override_settings for non-raw files to prevent metadata leakage
            current_override = override_settings
            
            _, ext = os.path.splitext(actual_file_path)
            ext_lower = ext.lower()
            
            if current_override and ext_lower not in temp_handler.raw_extensions:
                current_override = None
            elif current_override:
                # Check if target file has explicit resolution in name
                if re.search(r"[\-_](\d+)x(\d+)", basename):
                    current_override = None
                # [NEW] Check if file size matches the override settings
                elif current_override and os.path.exists(actual_file_path):
                         try:
                             fsize = os.path.getsize(actual_file_path)
                             ow = current_override.get('width', 0)
                             oh = current_override.get('height', 0)
                             odtype = current_override.get('dtype', np.uint8)
                             
                             if isinstance(odtype, str):
                                 # Try to parse string dtype
                                 try:
                                     cont, _, _, _ = temp_handler._parse_dtype_string(odtype)
                                     bpp = np.dtype(cont).itemsize
                                 except Exception:
                                     bpp = 1
                             else:
                                 bpp = np.dtype(odtype).itemsize
                                 
                             c_fmt = current_override.get('color_format', 'Grayscale')
                             ch = 4 if 'RGBA' in c_fmt else (3 if 'RGB' in c_fmt else 1)
                             
                             expected_size = ow * oh * bpp * ch
                             
                             if fsize != expected_size:
                                 current_override = None
                         except Exception:
                             pass

            try:
                temp_handler.load_image(actual_file_path, override_settings=current_override)
                # ... (rest of loading) ...


                
                # If this is an NPZ file with a specific key, switch to that key
                if npz_key_to_load and hasattr(temp_handler, 'npz_keys'):
                    if npz_key_to_load in temp_handler.npz_keys and temp_handler.npz_keys[npz_key_to_load]:
                        temp_handler.original_image_data = temp_handler.npz_data[npz_key_to_load]
                        temp_handler.current_npz_key = npz_key_to_load
                        # Update dimensions
                        if temp_handler.original_image_data.ndim == 2:
                            temp_handler.height, temp_handler.width = temp_handler.original_image_data.shape
                        elif temp_handler.original_image_data.ndim == 3:
                            temp_handler.height, temp_handler.width = temp_handler.original_image_data.shape[:2]
            except Exception:
                # Fallback if load fails (e.g. missing resolution and no override)
                continue
                
            data = temp_handler.original_image_data

            if data is not None and data.size > 0:
                # Update Max Zoom Limit
                dims = max(temp_handler.width, temp_handler.height)
                if dims > max_montage_zoom: max_montage_zoom = float(dims)

                image_label = ZoomableDraggableLabel(shared_state=self.montage_shared_state)
                image_label.set_data(data, is_pristine=True)
                image_label.file_path = file_path # Store for overlay restoration
                image_label.open_companion_depth.connect(self._on_open_companion_depth)
                
                # Store metadata for Info Pane
                image_label.metadata = {
                    'width': temp_handler.width,
                    'height': temp_handler.height,
                    'dtype': temp_handler.dtype,
                    'color_format': getattr(temp_handler, 'color_format', 'Grayscale'),
                    'file_size': os.path.getsize(actual_file_path) if os.path.isfile(actual_file_path) else 0,
                    'is_raw': temp_handler.is_raw
                }

                state_applied = False
                
                # 1. Try to inherit from PREVIOUS montage state
                if file_path in existing_states:
                    state = existing_states[file_path]
                    filtered_state = {
                        'colormap': state.get('colormap'),
                        'contrast_limits': state.get('contrast_limits')
                    }
                    image_label.set_view_state(filtered_state)
                    state_applied = True
                
                # 2. If not found, check other open windows
                if not state_applied:
                    for window in self.window_list:
                        # Retrieve the path safely, handling potential missing attribute or closed window
                        try:
                            # Skip windows in Montage Mode as their current_file_path might be stale
                            if window.stacked_widget.currentWidget() == window.montage_widget:
                                continue

                            win_path = getattr(window, 'current_file_path', None)
                            if win_path == file_path and window.active_label:
                                state = window.active_label.get_view_state()
                                # Only inherit Colormap and Contrast (preserve Montage's layout/zoom)
                                filtered_state = {
                                    'colormap': state.get('colormap'),
                                    'contrast_limits': state.get('contrast_limits')
                                }
                                image_label.set_view_state(filtered_state)
                                break
                        except Exception:
                            pass # Ignore errors accessing other windows

                # Use centralized method to set initial text/visibility based on pane state
                self._update_overlay_labels()
                image_label.clicked.connect(lambda label=image_label: self._set_active_montage_label(label))
                self.montage_labels.append(image_label)

                # Container for Image + Active Indicator Line
                container = QWidget()
                container_layout = QVBoxLayout(container)
                container_layout.setContentsMargins(0, 0, 0, 0)
                container_layout.setSpacing(0)
                
                container_layout.addWidget(image_label)
                
                # Active Indicator Line
                line = QFrame()
                line.setFixedHeight(4)
                line.setStyleSheet("background-color: transparent;")
                container_layout.addWidget(line)
                
                # Attach line to label for easy access
                image_label.indicator_line = line

                self.montage_layout.addWidget(container, row, col)
                # Removed hover_moved connection for activation (Click only)
                image_label.hover_moved.connect(self.update_status_bar) # Just update status bar, don't activate
                
                col += 1
                if col % 3 == 0:
                    row += 1
                    col = 0
        if self.montage_labels:
            self.montage_shared_state.max_zoom_limit = max_montage_zoom
            self._set_active_montage_label(self.montage_labels[0])

        self.stacked_widget.setCurrentWidget(self.montage_widget)
        # Synchronous update
        self.montage_widget.repaint()
        
        # Update thumbnail pane - ensure new images show up in gallery
        if not self._updating_from_thumbnail:
            self._refresh_thumbnail_pane()

    def _set_active_montage_label(self, label):
        if self.active_label:
            try:
                self.active_label.set_active(False)
                if hasattr(self.active_label, 'indicator_line'):
                     self.active_label.indicator_line.setStyleSheet("background-color: transparent;")
            except RuntimeError:
                # Widget was deleted, ignore
                pass

        self.active_label = label
        if self.active_label:
            try:
                self.active_label.set_active(True)
                if hasattr(self.active_label, 'indicator_line'):
                     self.active_label.indicator_line.setStyleSheet("background-color: #007AFF;") # Active Blue
            except RuntimeError:
                # Widget was deleted, ignore
                pass
        self._update_active_view()

    def _update_active_view(self, reset_histogram=True):
        if not self.active_label:
            return

        # Disconnect all signals first to prevent multiple connections
        try:

            self.image_label.zoom_factor_changed.disconnect()
            self.image_label.hover_moved.disconnect()
            self.image_label.view_changed.disconnect()
            for label in self.montage_labels:
                label.zoom_factor_changed.disconnect()
                label.hover_moved.disconnect()
                label.view_changed.disconnect()
            self.histogram_window.region_changed.disconnect()
        except TypeError:
            pass

        # Enable relevant tools/actions
        self.histogram_action.setEnabled(True)
        self.info_action.setEnabled(True)
        self.math_transform_action.setEnabled(True)
        self.colormap_combo.setEnabled(True)

        # Update Info Pane if metadata is available
        if hasattr(self.active_label, 'metadata'):
            m = self.active_label.metadata
            self.info_pane.blockSignals(True)
            self.info_pane.update_info(
                m['width'], m['height'], m['dtype'], 
                self.image_handler.dtype_map,
                file_size=m['file_size'],
                color_format=m['color_format']
            )
            self.info_pane.blockSignals(False)
            self.info_pane.set_raw_mode(m.get('is_raw', False))
        
        # Connect signals for the active label
        self.active_label.zoom_factor_changed.connect(self._on_image_label_zoom_changed)
        
        # Ensure overlay is correct (e.g. if switching views)
        self._update_overlay_labels()

        self.active_label.hover_moved.connect(self.update_status_bar)
        self.active_label.view_changed.connect(self.update_histogram_data)
        self.histogram_window.region_changed.connect(self.set_contrast_limits)

        self.colormap_combo.setEnabled(True)
        self.update_channel_options()
        
        # Sync the colormap combo box with the active label's current colormap
        self.colormap_combo.blockSignals(True)
        self.colormap_combo.setCurrentText(self.active_label.colormap)
        self.colormap_combo.blockSignals(False)

        self.update_histogram_data(new_image=reset_histogram)
        
        # Update 3D point cloud viewer if it's open
        if hasattr(self, 'point_cloud_viewer') and self.point_cloud_viewer and self.point_cloud_viewer.isVisible():
            if self.active_label and self.active_label.pristine_data is not None:
                self.point_cloud_viewer.set_data(self.active_label.pristine_data)

    def toggle_info_pane(self):
        visible = not self.info_pane.isVisible()
        self._set_dock_visibility_preserving_window(self.info_pane, visible)

    def toggle_math_transform_pane(self):
        visible = not self.math_transform_pane.isVisible()
        self._set_dock_visibility_preserving_window(self.math_transform_pane, visible)
        self._update_overlay_labels()

    def _update_overlay_labels(self):
        """Updates overlay labels based on current mode and Math Pane visibility."""
        show_math_vars = self.math_transform_pane.isVisible()
        is_montage = self.stacked_widget.currentWidget() == self.montage_widget
        
        # Helper to set text and visibility
        def update_label(label, math_text, file_path_attr):
            if show_math_vars:
                label.set_overlay_text(math_text)
                label.overlay_label.show()
            else:
                mode = getattr(self, 'overlay_mode', 0)
                if mode == 0:
                    label.overlay_label.hide()
                elif mode == 1:
                    # Basename
                    txt = os.path.basename(file_path_attr) if file_path_attr else ""
                    label.set_overlay_text(txt)
                    label.overlay_label.show()
                elif mode == 2:
                    # Full Path
                    txt = file_path_attr if file_path_attr else ""
                    label.set_overlay_text(txt)
                    label.overlay_label.show()

        if is_montage:
             for i, label in enumerate(self.montage_labels):
                 # Math Text: x (if single) or x1, x2...
                 if len(self.montage_labels) == 1:
                     math = "x"
                 else:
                     math = f"x{i+1}"
                 
                 # File Text
                 file_path = getattr(label, 'file_path', "")
                 update_label(label, math, file_path)
        else:
             if self.active_label:
                 file_path = getattr(self.active_label, 'file_path', "")
                     # If active_label is one of the montage labels (e.g. focused), use its path
                 update_label(self.active_label, "x", file_path)

    def generate_da3_depth_map(self):
        if not self.current_file_path:
            QMessageBox.warning(self, "Warning", "Please open an image first.")
            return

        dialog = DA3ModelDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            model_name = dialog.get_model()
            
            # Show progress UI
            self.progress_bar.setVisible(True)
            self.status_bar.showMessage(f"Generating DA3 Depth Map using {model_name}...")
            self.da3_action.setEnabled(False)
            
            # Start worker
            self.da3_worker = DA3Worker(self.current_file_path, model_name)
            self.da3_worker.finished.connect(self._on_da3_finished)
            self.da3_worker.failed.connect(self._on_da3_failed)
            self.da3_worker.progress.connect(self._on_da3_progress)
            self.da3_worker.download_progress.connect(self._on_da3_download_progress)
            self.da3_worker.start()

    def _on_da3_finished(self, output_path):
        self.progress_bar.setVisible(False)
        self.status_bar.clearMessage()
        self.da3_action.setEnabled(True)
        
        is_dir = os.path.isdir(output_path)
        msg = f"Depth {'folder' if is_dir else 'map'} generated: {output_path}"
        self.status_bar.showMessage(msg, 5000)
        
        if is_dir:
            # Refresh UI to show the 3D indication button/enable the 3D action
            self.update_image_display(reset_view=False)
            return
        
        # Determine existing files to append to
        current_files = []
        if self.stacked_widget.currentWidget() == self.montage_widget and self.montage_labels:
            current_files = [label.file_path for label in self.montage_labels if label.file_path]
        elif self.current_file_path:
            current_files = [self.current_file_path]
            
        combined_paths = current_files + [output_path]
        self.display_montage(combined_paths, is_manual=True)

    def _on_da3_failed(self, error_msg):
        self.progress_bar.setVisible(False)
        self.status_bar.clearMessage()
        self.da3_action.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Failed to generate depth map:\n{error_msg}")

    def _on_da3_progress(self, current, total):
        if total > 0:
            percent = int((current / total) * 100)
            self.progress_bar.setValue(percent)
            # Find the model name to include in status
            model_name = "Model"
            if hasattr(self, 'da3_worker') and self.da3_worker:
                model_name = self.da3_worker.model_name
            self.status_bar.showMessage(f"Processing {model_name}... {percent}% ({current}/{total})")

    def _on_da3_download_progress(self, filename, percent):
        self.progress_bar.setValue(int(percent))
        self.status_bar.showMessage(f"Downloading {filename}... {percent:.1f}%")

    def toggle_histogram_window(self):
        # Histogram is a tool window, but let's see if we want to preserve window size here too.
        # User said "file explorer... math... opened images", didn't explicitly mention histogram,
        # but it's cleaner to be consistent. 
        # Actually histogram window is a QDialog/Tool in current impl?
        # Let's check.
        # Assuming HistogramWidget is a QWidget with Qt.WindowType.Tool flag, not a QDockWidget.
        # The _set_dock_visibility_preserving_window is designed for QDockWidgets.
        # So, we keep the existing logic for the histogram window.
        if self.histogram_window.isVisible():
            self.histogram_window.hide()
        else:
            self.histogram_window.show()
            self.update_histogram_data(new_image=True)
            
            # Position to the right of the main window
            frame_geo = self.frameGeometry()
            screen_width = self.screen().geometry().width()
            
            # Default small size
            hist_width = 400
            hist_height = 300
            self.histogram_window.resize(hist_width, hist_height)
            
            x = frame_geo.right() + 10
            y = frame_geo.top()
            
            # If off-screen, clamp or move
            if x + hist_width > screen_width:
                x = screen_width - hist_width - 10
            
            self.histogram_window.move(x, y)

    def toggle_thumbnail_pane(self):
        visible = not self.thumbnail_pane.isVisible()
        if visible:
            self.thumbnail_pane.populate(self.window_list)
            
        self._set_dock_visibility_preserving_window(self.thumbnail_pane, visible)

        if visible:
            self.thumbnail_pane.setFocus()
            
    def _set_dock_visibility_preserving_window(self, dock, visible):
        """Toggle dock visibility while keeping image pane centered in same screen position."""
        if dock.isVisible() == visible:
            return

        dock_name = dock.objectName()
        
        # Save original position
        original_x = self.x()
        original_y = self.y()
        
        # Get all docks except the one being toggled
        all_docks = [self.file_explorer_pane, self.info_pane, 
                     self.math_transform_pane, self.thumbnail_pane]
        other_docks = [d for d in all_docks if d != dock and d.isVisible() and not d.isFloating()]
        
        # Determine the dock's width
        if visible:
            dock_width = self._dock_sizes.get(dock_name, 0)
            if dock_width <= 0:
                dock_width = dock.sizeHint().width()
        else:
            dock_width = dock.width()
            self._dock_sizes[dock_name] = dock_width
        
        # Save current sizes of other docks
        for d in other_docks:
            self._dock_sizes[d.objectName()] = d.width()
        
        # Determine if dock is on left or right
        is_left_dock = (dock == self.file_explorer_pane)
        
        # Calculate new window width (half dock width split)
        current_width = self.width()
        half_dock_width = dock_width // 2
        
        if visible:
            new_width = current_width + half_dock_width
        else:
            new_width = current_width - half_dock_width
        
        # Toggle visibility
        dock.setVisible(visible)
        
        # Calculate new window x position to keep image pane centered
        if is_left_dock:
            if visible:
                new_window_x = original_x - half_dock_width
            else:
                new_window_x = original_x + half_dock_width
        else:
            new_window_x = original_x
        
        # Use separate calls to avoid any coordinate coupling
        self.move(new_window_x, original_y)
        self.resize(new_width, self.height())
        
        # Restore other dock sizes
        QTimer.singleShot(0, lambda: self._restore_dock_sizes(other_docks))
    
    def _restore_dock_sizes(self, docks):
        """Restore saved sizes for the given docks."""
        for dock in docks:
            dock_name = dock.objectName()
            if dock_name in self._dock_sizes:
                saved_width = self._dock_sizes[dock_name]
                if saved_width > 0:
                    self.resizeDocks([dock], [saved_width], Qt.Orientation.Horizontal)





    def _on_image_label_zoom_changed(self, scale_factor):
        if self.sender() is not self.active_label: return

            
        self.zoom_status_label.setText(f"Zoom: {int(scale_factor * 100)}%")

    def reapply_raw_parameters(self, raw_settings):
        # Renamed argument to raw_settings to avoid shadowing global settings module if imported
        try:
            self.progress_bar.setVisible(True)
            QApplication.processEvents()
            self.image_handler.load_image(self.current_file_path, override_settings=raw_settings)
            if self.active_label:
                self.active_label.set_data(self.image_handler.original_image_data)
            
            # Update channel options (e.g. if format changed from Raw to Bayer RGB)
            self.update_channel_options()
            
            # Ensure colormap is 'gray' if image is RGB to allowing RGB display in Label
            # Otherwise it might treat it as single channel (Channel 0) if colormap is actively set to something else
            data = self.image_handler.original_image_data
            if data.ndim == 3 and data.shape[2] in [3, 4]:
                if self.colormap_combo.currentText() != "gray" and self.colormap_combo.currentText() != "flow":
                     self.colormap_combo.blockSignals(True)
                     self.colormap_combo.setCurrentText("gray")
                     self.colormap_combo.blockSignals(False)
                     # Force active label update
                     if self.active_label:
                         self.active_label.set_colormap("gray")

            self.update_histogram_data(new_image=True)
            
            # Save to history if this file lacks explicit resolution
            width, height, _ = self.image_handler.parse_resolution(self.current_file_path)
            if width == 0 or height == 0:
                settings.update_raw_history(self.current_file_path, raw_settings)
            
            if self.active_label:
                self.active_label.repaint()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying parameters:\n{e}")
        finally:
             self.progress_bar.setVisible(False)

    def apply_math_transform(self, expression, from_update=False):
        self.current_math_expression = expression
        try:
            context = {}
            if self.stacked_widget.currentWidget() == self.montage_widget:
                for i, label in enumerate(self.montage_labels):
                    if label.pristine_data is not None:
                        context[f"x{i+1}"] = label.pristine_data.astype(np.float64)
                
                # Alias 'x' to 'x1' if only one image in montage
                if len(self.montage_labels) == 1 and "x1" in context:
                    context["x"] = context["x1"]
                    
            elif self.active_label and self.active_label.pristine_data is not None:
                context["x"] = self.active_label.pristine_data.astype(np.float64)

            transformed_data = self.image_handler.apply_math_transform(expression, context_dict=context)
            if self.active_label:
                self.active_label.set_data(transformed_data, reset_view=False)
                self.active_label.repaint()
            
                # Update 3D View if open
                if hasattr(self, 'point_cloud_viewer') and self.point_cloud_viewer and self.point_cloud_viewer.isVisible():
                     self.point_cloud_viewer.set_data(transformed_data)
                     
            self.update_histogram_data(new_image=True)
            self.math_transform_pane.set_error_message("")
        except Exception as e:
            self.math_transform_pane.set_error_message(str(e))

    def restore_original_image(self):
        self.current_math_expression = None
        if self.active_label and self.active_label.pristine_data is not None:
            self.active_label.set_data(self.active_label.pristine_data, reset_view=False)
            
            # Update 3D View if open
            if hasattr(self, 'point_cloud_viewer') and self.point_cloud_viewer and self.point_cloud_viewer.isVisible():
                 self.point_cloud_viewer.set_data(self.active_label.pristine_data)

            self.update_histogram_data(new_image=True)
        # The instruction had a try-except block here, but it's not needed as the `if` condition handles the main case.
        # If `active_label` or `pristine_data` is None, it simply won't execute the `if` block.
        # If an error occurs within `set_data` or `update_histogram_data`, it should be handled there or propagate.
        # Adding a generic `except Exception` here without specific error handling might hide issues.
        # Keeping it as per the original structure of similar methods.

    def open_zoom_settings(self):
        dialog = ZoomSettingsDialog(self)
        dialog.set_settings(self.zoom_settings)
        if dialog.exec():
            self.zoom_settings = dialog.get_settings()
            self.apply_zoom_settings()

    def apply_zoom_settings(self):
        # Apply to main image label
        if hasattr(self, 'image_label'):
             self._apply_zoom_settings_to_label(self.image_label)
        
        # Apply to all montage labels
        for label in self.montage_labels:
             self._apply_zoom_settings_to_label(label)

    def _apply_zoom_settings_to_label(self, label):
        label.zoom_speed = self.zoom_settings["zoom_speed"]
        interp_map = {"Smooth": Qt.TransformationMode.SmoothTransformation,
                      "Nearest": Qt.TransformationMode.FastTransformation}
        label.zoom_in_interp = interp_map[self.zoom_settings["zoom_in_interp"]]
        label.zoom_out_interp = interp_map[self.zoom_settings["zoom_out_interp"]]

    def set_colormap(self, name):
        # "flow" is a special mode that requires re-generating the RGB buffer via update_image_display.
        # Standard colormaps are applied in-shader/texture on the ImageLabel.

        current_is_flow = (name == "flow")
        # We need a way to know if we are currently displaying a baked "flow" RGB image.
        # We can infer this: if we are in "flow" mode, we need to generate it.
        # If we are switching AWAY from "flow" (meaning name != flow), we need to revert to raw data.
        # Since we don't strictly track "previous mode", calling update_image_display is the safest way 
        # to ensure the data sent to ImageLabel matches the mode.
        
        # However, for pure 1ch -> 1ch colormap changes (gray -> hot), calling update_image_display 
        # is slightly heavier but correct. Given modern CPUs, slicing a 2K image is negligible.
        # Let's prioritize correctness:
        
        self.update_image_display(reset_view=False)
        
        if self.active_label:
             # If mode is "flow", the data is now RGB, so set_colormap("flow") might act as "none" 
             # or simply be ignored by ImageLabel if data is RGB.
             self.active_label.set_colormap(name)
            
             # Broadcast invalidation to all windows using this image as overlay
             path = getattr(self.active_label, 'file_path', self.current_file_path)
             if path:
                 for win in self.window_list:
                     if hasattr(win, '_invalidate_overlay_cache'):
                         win._invalidate_overlay_cache(path)

    def toggle_colorbar(self, checked):
        if self.stacked_widget.currentWidget() == self.montage_widget:
            for label in self.montage_labels:
                label.show_colorbar = checked
                label.update()
        else:
            if self.active_label:
                self.active_label.show_colorbar = checked
                self.active_label.update()

    def set_contrast_limits(self, min_val, max_val):
        # Apply only to the active label, ensuring independence in montage view
        if self.active_label:
            self.active_label.set_contrast_limits(min_val, max_val)

            # Broadcast invalidation to all windows using this image as overlay
            path = getattr(self.active_label, 'file_path', self.current_file_path)
            if path:
                for win in self.window_list:
                    if hasattr(win, '_invalidate_overlay_cache'):
                        win._invalidate_overlay_cache(path)

    def open_3d_view(self):
        if not self.active_label or self.active_label.original_data is None:
             return
             
        data = self.active_label.original_data
        rgb_data = None
        
        
        # Pair depth map and RGB image if in montage view
        if self.stacked_widget.currentWidget() == self.montage_widget and len(self.montage_labels) >= 2:
            is_active_rgb = (data.ndim == 3 and data.shape[2] >= 3)
            is_active_depth = (data.ndim == 2 or (data.ndim == 3 and data.shape[2] == 1))
            
            for label in self.montage_labels:
                if label == self.active_label or label.original_data is None:
                    continue
                    
                other_data = label.original_data
                is_other_rgb = (other_data.ndim == 3 and other_data.shape[2] >= 3)
                is_other_depth = (other_data.ndim == 2 or (other_data.ndim == 3 and other_data.shape[2] == 1))
                
                if is_active_rgb and is_other_depth:
                    # Active is RGB, other is depth. Geometry from depth, colors from RGB.
                    data = other_data
                    rgb_data = self.active_label.original_data
                    break
                elif is_active_depth and is_other_rgb:
                    # Active is depth, other is RGB.
                    rgb_data = other_data
                    break
                    
        # If in single view OR we couldn't find a pair in montage, check for companion _DEPTH automatically
        if rgb_data is None and self.active_label.file_path:
             is_active_rgb = (data.ndim == 3 and data.shape[2] >= 3)
             if is_active_rgb:
                 base, _ = os.path.splitext(self.active_label.file_path)
                 depth_path = f"{base}_DEPTH.tiff"
                 depth_dir = f"{base}_DEPTH"
                 
                 target_depth = None
                 if os.path.exists(depth_path):
                     target_depth = depth_path
                 elif os.path.isdir(depth_dir):
                     # Frame-accurate depth for video
                     frame_idx = getattr(self.image_handler, 'current_frame_index', 0)
                     # Files saved as frame_1.tiff (1-based)
                     potential_path = os.path.join(depth_dir, f"frame_{frame_idx+1}.tiff")
                     if os.path.exists(potential_path):
                         target_depth = potential_path
                 
                 if target_depth:
                     try:
                         from image_handler import ImageHandler
                         temp_handler = ImageHandler()
                         temp_handler.load_image(target_depth)
                         depth_img_data = temp_handler.original_image_data
                         if depth_img_data is not None:
                              rgb_data = data
                              data = depth_img_data
                     except Exception as e:
                         pass
        
        if not hasattr(self, 'point_cloud_viewer') or self.point_cloud_viewer is None:
             from widgets import PointCloudViewer
             self.point_cloud_viewer = PointCloudViewer(self)
             self.point_cloud_viewer.set_data(data, reset_view=True, rgb_data=rgb_data)
        else:
             self.point_cloud_viewer.set_data(data, reset_view=False, rgb_data=rgb_data)
             
        self.point_cloud_viewer.show()
        self.point_cloud_viewer.activateWindow()

    def _on_open_companion_depth(self, depth_path):
        """Callback to open associated depth map alongside current image in montage view"""
        if not self.active_label or not self.active_label.file_path:
            return
            
        current_path = self.active_label.file_path
        
        # If it's a directory (video depth), find the current frame
        if os.path.isdir(depth_path):
            frame_idx = getattr(self.image_handler, 'current_frame_index', 0)
            target_depth_path = os.path.join(depth_path, f"frame_{frame_idx+1}.tiff")
            if os.path.exists(target_depth_path):
                depth_path = target_depth_path

        # Force a montage display with the original image AND its depth map
        # display_montage can take duplicates but in this case we want [original, depth]
        self.display_montage([current_path, depth_path], is_manual=True)

    def toggle_file_explorer_pane(self):
        visible = not self.file_explorer_pane.isVisible()
        # If showing, set root path to current file's directory if available
        if visible:
            self.file_explorer_pane.set_root_path(self.current_file_path)
            
        self._set_dock_visibility_preserving_window(self.file_explorer_pane, visible)

    def _on_explorer_files_selected(self, file_paths):
        # Allow empty list to proceed to clear the view if needed
        if file_paths is None:
            return
            
        if not file_paths:
             # Selection cleared
             # If we are in Montage mode, clear it.
             # If in Single view, maybe stay? Or clear?
             # For now, let's clear the montage if we were using it, or do nothing.
             # But the user complains "unselected images still appear".
             # If they deselect everything, they probably expect an empty view or the last valid single file?
             # Let's try displaying an empty montage if we are already in montage mode, or just return if not?
             # Actually, if I select A,B (Montage), then deselect all. I expect empty.
             if self.stacked_widget.currentWidget() == self.montage_widget:
                 self.display_montage([], is_manual=False)
             return

        # Capture View State and Raw Settings from ACTIVE image
        view_state = None
        raw_settings = None
        
        if self.active_label and self.active_label.current_pixmap:
             data = self.active_label.original_data
             limits = self.active_label.contrast_limits
             percentiles = self._get_percentiles_from_limits(data, limits)
             
             view_state = {
                'zoom_scale': self.active_label.zoom_scale,
                'pan_pos': self.active_label.pan_pos, # QPointF
                'colormap': self.active_label.colormap, # Use active_label's colormap
                'contrast_percentiles': percentiles
             }
        
        # Capture Raw Settings if current file was loaded with them (explicit or implicit)
        # We can construct them from image_handler state
        if self.image_handler.is_raw:
             self.last_raw_settings = {
                'width': self.image_handler.width,
                'height': self.image_handler.height,
                'dtype': self.image_handler.dtype,
                'color_format': self.image_handler.color_format
             }
        
        raw_settings = self.last_raw_settings

        if len(file_paths) == 1:
            target_path = file_paths[0]
            _, ext = os.path.splitext(target_path)
            target_is_raw = ext.lower() in self.image_handler.raw_extensions
            
            # Only inherit raw settings (resolution/dtype) if target is also raw
            effective_raw = raw_settings if target_is_raw else None
            self.open_file(target_path, override_settings=effective_raw, maintain_view_state=view_state, is_manual=False)
        else:
            # For Montage, check if the first file is raw as a representative
            _, ext = os.path.splitext(file_paths[0])
            target_is_raw = ext.lower() in self.image_handler.raw_extensions
            
            self._temp_montage_override = raw_settings if target_is_raw else None
            self.display_montage(file_paths)
            self._temp_montage_override = None # Clear
            
            # Apply View State to montage (approximate)
            if view_state and self.montage_shared_state:
                # Apply Colormap
                self.set_colormap(view_state['colormap'])
                
                # Apply Contrast Percentiles to all labels in montage
                percentiles = view_state.get('contrast_percentiles')
                if percentiles:
                    for label in self.montage_labels:
                        if label.original_data is not None:
                             new_limits = self._get_limits_from_percentiles(label.original_data, percentiles)
                             label.set_contrast_limits(*new_limits)
                # Apply Zoom (if possible) - Montage controls its own zoom initially to fit. 
                # Maintaining zoom from single view to montage is ambiguous (zoom level relative to what?).
                # But requirement says "using its all current modifications".
                # We can try to set zoom on the shared state.
                pass 

    def open_image_dialog(self):
        extensions = self.image_handler.raw_extensions + ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif', '.webp'] + self.image_handler.video_extensions
        # Create unique list
        extensions = list(set(extensions))
        ext_str = " ".join(['*' + ext for ext in extensions])
        
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "",
                                                   f"Image Files ({ext_str})")
        if file_path:
            self.open_file(file_path, is_manual=True)

    def open_files(self, file_paths):
        """
        Refined multi-file opening logic:
        - <= 3 files: Open as montage.
        - > 3 files: Open first in main view, add all to sidebar, show sidebar.
        """
        if not file_paths:
            return

        # Filter out invalid paths
        valid_paths = [p for p in file_paths if os.path.exists(p)]
        if not valid_paths:
            return

        if len(valid_paths) <= 3:
            self.display_montage(valid_paths, is_manual=True)
        else:
            if self.thumbnail_pane:
                self.thumbnail_pane.unremove_files(valid_paths)
            
            # 1. Open the first one in main view
            self.open_file(valid_paths[0], is_manual=True)
            
            # 2. Add all to the thumbnail pane
            self.thumbnail_pane.add_files(valid_paths)
            
            # 3. Automatically show/expand the thumbnail pane
            if hasattr(self, 'thumbnail_pane'):
                self.thumbnail_pane.show()
                self.thumbnail_pane.raise_()

    def open_file(self, file_path=None, override_settings=None, maintain_view_state=None, is_manual=False):
        if file_path and self.thumbnail_pane:
            if is_manual:
                self.thumbnail_pane.add_to_manual_paths([file_path])
            self.thumbnail_pane.unremove_files([file_path])
        
        # Handle NPZ key paths (format: "file.npz#key_name")
        npz_key_to_switch = None
        actual_file_path = file_path  # The actual file on disk
        if '#' in file_path:
            actual_file, key_name = file_path.rsplit('#', 1)
            npz_key_to_switch = key_name
            actual_file_path = actual_file
        
        # Store the original path (with NPZ key if present) for overlay tracking
        self.current_file_path = file_path
        self.image_label.file_path = actual_file_path
        
        # Clear overlays for files other than the current one
        # For NPZ files, keep overlays that involve keys from the same base file
        overlays_to_remove = []
        for (src, tgt) in self.overlay_alphas.keys():
            # Extract base file paths for comparison
            src_base = src.rsplit('#', 1)[0] if '#' in src else src
            tgt_base = tgt.rsplit('#', 1)[0] if '#' in tgt else tgt
            current_base = file_path.rsplit('#', 1)[0] if '#' in file_path else file_path
            # Remove pairs where neither source nor target matches the current file
            if src_base != current_base and tgt_base != current_base:
                overlays_to_remove.append((src, tgt))
        
        for pair in overlays_to_remove:
            del self.overlay_alphas[pair]
            if pair in self.overlay_cache:
                del self.overlay_cache[pair]
        
        # Update File Explorer Path (use actual file on disk)
        if self.file_explorer_pane.isVisible():
             self.file_explorer_pane.set_root_path(actual_file_path)

        # Determine if it's a raw file
        _, ext = os.path.splitext(actual_file_path)
        is_raw = ext.lower() in self.image_handler.raw_extensions
        
        # Get File Size
        if os.path.exists(actual_file_path):
             file_size = os.path.getsize(actual_file_path)
        else:
             file_size = 0

        # [FIX] Clear override_settings for non-raw files to prevent metadata leakage.
        # This MUST happen before any logic that uses override_settings.
        if override_settings and not is_raw:
            override_settings = None

        # [REFINEMENT] Inherit settings only if target file has no explicit settings (filename or history)
        if is_raw and override_settings:
            basename = os.path.basename(actual_file_path)
            has_explicit = bool(re.search(r"[\-_](\d+)x(\d+)", basename))
            if not has_explicit:
                history = settings.load_raw_history()
                if actual_file_path in history:
                    has_explicit = True
            
            if has_explicit:
                # Target has its own explicit/historical settings, ignore inherited override
                override_settings = None

        # Check resolution for raw files BEFORE attempting load
        if is_raw and not override_settings: # Only guess/check history if no explicit override_settings provided
             # This uses the new method which returns (0,0) instead of raising
             width, height, dtype_raw = self.image_handler.parse_resolution(actual_file_path)
             
             if width == 0 or height == 0:
                 # Logic for missing resolution (formerly in except block)
                 self.info_action.setEnabled(True)
                 self.info_pane.set_raw_mode(True)
                 self.stacked_widget.setCurrentWidget(self.image_display_container)
                 
                 # Infer dtype from extension
                 dtype = self.image_handler.dtype_map.get(ext.lower(), np.uint8)
                 
                 # Determine BPP from dtype
                 try:
                     if isinstance(dtype, str):
                          container, _, _, _ = self.image_handler._parse_dtype_string(dtype)
                          guess_bpp = float(np.dtype(container).itemsize)
                     else:
                          guess_bpp = float(np.dtype(dtype).itemsize)
                 except Exception:
                     guess_bpp = 1.0

                 # Guess Parameters using 4:3 Aspect Ratio
                 guess_width = int(np.sqrt((file_size / guess_bpp) * (4.0/3.0)))
                 # Align to 16
                 guess_width = (guess_width // 16) * 16
                 if guess_width < 1: guess_width = 1280

                 width = guess_width
                 height = int(file_size / (width * guess_bpp))
                 if height < 1: height = 1
                 
                 self.info_pane.update_info(width, height, dtype, self.image_handler.dtype_map, file_size=file_size)
                 self.info_pane.show()
                 self.toggle_info_pane() # Ensure visible
                 
                 self.status_bar.showMessage("Resolution missing from filename. Using estimated parameters.", 10000)
                 
                 # Check history for explicit settings
                 history = settings.load_raw_history()
                 params = history.get(file_path)
                 
                 if params:
                      guess_settings = params
                      # Update Info Pane with historical params to prevent mismatch if user opens pane
                      self.info_pane.update_info(params['width'], params['height'], params['dtype'], 
                                                 self.image_handler.dtype_map, file_size=file_size)
                      # Note: We can't easily set color_format in info_pane here without exposing more API, 
                      # but it will sync when settings_changed is emitted or we could add a method.
                      # For now, update_info handles W/H/Dtype. Format might default to Grayscale.
                 else:
                      # Auto-load with guessed parameters
                      guess_settings = {
                             'width': width,
                             'height': height,
                             'dtype': dtype, 
                             'color_format': 'Grayscale'
                      }
                 
                 # Use the guessed settings as override
                 override_settings = guess_settings

                 try:
                     self.image_handler.load_image(file_path, override_settings=guess_settings)
                     
                     # Success - enable UI controls
                     QApplication.processEvents() # Force layout update
                     
                     # Success - store metadata for Info Pane
                     self.image_label.metadata = {
                         'width': self.image_handler.width,
                         'height': self.image_handler.height,
                         'dtype': self.image_handler.dtype,
                         'color_format': getattr(self.image_handler, 'color_format', 'Grayscale'),
                         'file_size': os.path.getsize(file_path) if os.path.isfile(file_path) else 0,
                         'is_raw': self.image_handler.is_raw
                     }
                     
                     self._set_active_montage_label(self.image_label)
                     self.update_image_display(reset_view=True)
                     self._apply_histogram_preset(0, 100)
                     self.overlay_cache.clear()
                     self._update_overlays() 
                     
                     self.recent_files = settings.add_to_recent_files(self.recent_files, file_path)
                     self._update_recent_files_menu()
                     self._update_overlay_labels()
                        
                     # Save successful load to history
                     settings.update_raw_history(file_path, guess_settings)
                     
                     # Synchronous update
                     self.image_label.repaint()
                     if self.histogram_action.isChecked():
                         self.histogram_window.repaint()
                     
                     # Update channel options based on loaded image
                     self.update_channel_options()
                     
                     # Update thumbnail pane - ensure new image is in gallery
                     if not self._updating_from_thumbnail:
                         self._refresh_thumbnail_pane()
                     
                 except Exception as e:
                     pass
                     
                 return # Stop here, as we've handled the load attempt manually

        # Proceed to load image (Standard or Raw with Resolution or Override)
        try:     
            # If override_settings is passed (from History, Guess, or Explorer Inheritance), use it.
            self.progress_bar.setVisible(True)
            QApplication.processEvents()
            self.image_handler.load_image(actual_file_path, override_settings=override_settings)

            # Success - store metadata for Info Pane
            self.image_label.metadata = {
                'width': self.image_handler.width,
                'height': self.image_handler.height,
                'dtype': self.image_handler.dtype,
                'color_format': getattr(self.image_handler, 'color_format', 'Grayscale'),
                'file_size': file_size,
                'is_raw': self.image_handler.is_raw
            }

            # Proceed to set active
            self.stacked_widget.setCurrentWidget(self.image_display_container)
            QApplication.processEvents() # Force layout update so widget has correct size for fit_to_view
            self._set_active_montage_label(self.image_label)
            self.update_image_display(reset_view=not maintain_view_state)
            
            # Apply State (View / Contrast)
            if maintain_view_state:
                 # Restore View State
                 self.image_label.zoom_scale = maintain_view_state.get('zoom_scale', 1.0)
                 self.image_label.pan_pos = maintain_view_state.get('pan_pos', QPointF(0,0))
                 self.image_label.update_transform()
                 
                 # Contrast
                 percentiles = maintain_view_state.get('contrast_percentiles')
                 if percentiles:
                      new_limits = self._get_limits_from_percentiles(self.image_handler.original_image_data, percentiles)
                      self.image_label.set_contrast_limits(*new_limits)
                      
                 # Colormap
                 cmap = maintain_view_state.get('colormap')
                 if cmap:
                      self.set_colormap(cmap)
                      self.colormap_combo.setCurrentText(cmap)

            else:
                # Default Apply Min-Max contrast stretch for new image
                self._apply_histogram_preset(0, 100)
            
            # Clear overlay cache
            self.overlay_cache.clear()
            self._update_overlays() 
    
            self.recent_files = settings.add_to_recent_files(self.recent_files, actual_file_path)
            self._update_recent_files_menu()
            self.image_label.file_path = actual_file_path # Essential for overlay system to identify target
            self._update_overlay_labels()
                
            # Save override to history if successful (and if it was a missing resolution file)
            # Logic: If we used override_settings, we should save it? 
            # If explicit resolution exists, self.image_handler.parse_resolution returns values.
            # If we passed override settings, we don't necessarily update history unless it was "missing".
            # But implementation plan says "update history on success".
            # Let's check:
            w, h, _ = self.image_handler.parse_resolution(actual_file_path)
            if (w == 0 or h == 0) and override_settings:
                  settings.update_raw_history(actual_file_path, override_settings)

            # Update sticky settings if it's a raw file
            if self.image_handler.is_raw:
                self.last_raw_settings = {
                    'width': self.image_handler.width,
                    'height': self.image_handler.height,
                    'dtype': self.image_handler.dtype,
                    'color_format': self.image_handler.color_format
                }
            
            # Synchronous update
            self.image_label.repaint()
            
            # Update thumbnail selection states to reflect current image
            self._update_thumbnail_selection_states()
            if self.histogram_action.isChecked():
                self.histogram_window.repaint()


            # Configure Video UI
            if self.image_handler.is_video:
                 self.video_toolbar.show()
                 self.video_slider.blockSignals(True)
                 self.video_slider.setRange(0, max(0, self.image_handler.video_frame_count - 1))
                 self.video_slider.setValue(self.image_handler.current_frame_index)
                 self.video_slider.blockSignals(False)
                 self.fps_spin.blockSignals(True)
                 self.fps_spin.setValue(int(self.image_handler.video_fps))
                 self.fps_spin.blockSignals(False)
                 self.frame_label.setText(f" {self.image_handler.current_frame_index + 1} / {self.image_handler.video_frame_count} ")
                 # Reset Play state
                 self.playback_timer.stop()
                 self.play_action.setChecked(False)
                 self.play_action.setText("Play")
            else:
                 self.video_toolbar.hide()
                 self.playback_timer.stop()
                 
            self.update_channel_options()
            
            # Switch to specific NPZ key if requested
            if npz_key_to_switch and hasattr(self.image_handler, 'npz_keys'):
                if npz_key_to_switch in self.image_handler.npz_keys:
                    # Update channel combo to trigger key switch
                    self.channel_combo.blockSignals(True)
                    try:
                        index = list(self.image_handler.npz_keys.keys()).index(npz_key_to_switch)
                        self.channel_combo.setCurrentIndex(index)
                    except ValueError:
                        pass
                    self.channel_combo.blockSignals(False)
                    # Manually trigger the update
                    self.update_image_display(reset_view=False)

            # Update thumbnail pane - ensure new image is in gallery
            if not self._updating_from_thumbnail:
                 self._refresh_thumbnail_pane()

        except Exception as e:
            if is_raw and override_settings:
                # Transform/Inheritance caused a mismatch? Fallback to guessing.
                self.open_file(actual_file_path, override_settings=None, maintain_view_state=maintain_view_state)
                return
            
            QMessageBox.critical(self, "Error", f"Error opening image:\n{e}")
            self.math_transform_action.setEnabled(False)
            self.info_action.setEnabled(False)
            self.info_pane.set_raw_mode(False)

            self.histogram_action.setEnabled(False)
        
        finally:
            self.progress_bar.setVisible(False)

    def paste_from_clipboard(self):
        clipboard = QApplication.clipboard()
        mime_data = clipboard.mimeData()
        
        # Determine focus target
        focus_widget = QApplication.focusWidget()
        is_thumbnail_focused = False
        if self.thumbnail_pane and focus_widget:
            # Check if focus is inside thumbnail pane
            if self.thumbnail_pane.isAncestorOf(focus_widget) or focus_widget == self.thumbnail_pane:
                is_thumbnail_focused = True

        paths_to_add = []

        if mime_data.hasUrls():
            urls = mime_data.urls()
            if urls:
                paths_to_add = [u.toLocalFile() for u in urls if u.toLocalFile() and os.path.isfile(u.toLocalFile())]

        elif mime_data.hasText() and mime_data.text().startswith("IV_STATE:"):
            # Handle View State Paste
            try:
                text = mime_data.text()
                json_str = text[9:] # Strip prefix
                state = json.loads(json_str)
                
                if self.active_label:
                    # Convert percentiles back to limits based on target image
                    if 'contrast_percentiles' in state and self.active_label.original_data is not None:
                        new_limits = self._get_limits_from_percentiles(
                            self.active_label.original_data, 
                            state['contrast_percentiles']
                        )
                        state['contrast_limits'] = new_limits
                    
                    self.active_label.set_view_state(state)
                    self._update_active_view(reset_histogram=False) # Update UI to reflect new state
                    self.status_bar.showMessage("View Settings Pasted", 2000)
                    return
            except Exception as e:
                print(f"Error pasting view state: {e}")
                self.status_bar.showMessage("Paste Settings Failed", 2000)
            return

        elif mime_data.hasImage():
            qimage = clipboard.image()
            if not qimage.isNull():
                try:
                    # Check if we should paste as new image or replace current
                    # For now, let's keep the existing behavior of creating a temp file
                    # BUT if we are pasting into an existing view, maybe we want to replace correctly?
                    # The user asked for "paste image" support.
                    # Existing logic saves to temp file and "opens" it.
                    
                    # Save to temp file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    temp_dir = tempfile.gettempdir()
                    temp_path = os.path.join(temp_dir, f"Clipboard_{timestamp}.png")
                    qimage.save(temp_path, "PNG")
                    paths_to_add = [temp_path]
                except Exception as e:
                    QMessageBox.warning(self, "Paste Error", f"Failed to save clipboard image: {e}")
                    return
        else:
             self.status_bar.showMessage("Clipboard does not contain an image or valid file.", 3000)
             return

        if not paths_to_add:
            return

        # Action Logic
        if is_thumbnail_focused:
            # Add to thumbnail pane logic
            self.thumbnail_pane.add_files(paths_to_add)
        else:
            # Main View Logic
            current_has_image = (self.image_handler.original_image_data is not None)
            
            if current_has_image and len(paths_to_add) > 0:
                # Add to Montage
                current_paths = []
                if self.stacked_widget.currentWidget() == self.montage_widget:
                     current_paths = [label.file_path for label in self.montage_labels if hasattr(label, 'file_path') and label.file_path]
                elif self.current_file_path:
                     current_paths = [self.current_file_path]
                
                # Combine unique paths
                new_montage_paths = []
                # Keep order
                seen = set()
                for p in current_paths + paths_to_add:
                    if p not in seen:
                        new_montage_paths.append(p)
                        seen.add(p)
                
                self.display_montage(new_montage_paths)
            else:
                # Open first file (standard behavior)
                if len(paths_to_add) == 1:
                    self.open_file(paths_to_add[0])
                else:
                    self.open_files(paths_to_add)

    def show_about_dialog(self):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("About ImageViewer")
        msg_box.setText(f"<h3>ImageViewer</h3>Version {__version__}")
        msg_box.setInformativeText("A modern, high-performance image viewer for scientific and standard formats.<br><br>Created by Yakirma.")
        
        # Add GitHub Link
        github_url = "https://github.com/yakirma/ImageViewer"
        msg_box.setDetailedText(f"Source Code: {github_url}")
        
        # Add a clickable button/link to GitHub? 
        # DetailedText provides a hidden text area, but for a link we might want a button.
        # Let's add a button to visit GitHub.
        
        visit_btn = msg_box.addButton("Visit GitHub", QMessageBox.ButtonRole.ActionRole)
        msg_box.addButton(QMessageBox.StandardButton.Ok)
        
        msg_box.exec()
        
        if msg_box.clickedButton() == visit_btn:
            QDesktopServices.openUrl(QUrl(github_url))


    def export_current_view(self):
        if self.active_label and self.active_label.current_pixmap:
            # Default to user home directory
            home_dir = QDir.homePath()
            
            # Default filename: use original filename if available
            default_name = "image_export"
            if hasattr(self.active_label, 'file_path') and self.active_label.file_path:
                 base_name = os.path.splitext(os.path.basename(self.active_label.file_path))[0]
                 default_name = f"{base_name}_export"
            
            initial_path = os.path.join(home_dir, default_name)

            # Filters
            filters = "Images (*.png *.tiff *.tif *.jpg *.jpeg *.bmp);;PNG (*.png);;TIFF (*.tiff *.tif);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)"
            
            file_name, selected_filter = QFileDialog.getSaveFileName(self, "Export Current View", initial_path, filters)
            
            if file_name:
                # auto-append extension if missing based on filter logic is usually handled by OS or we can enforce
                # QImage.save usually detects by extension.
                # If username types 'foo' and selected 'PNG', it might save as 'foo'.
                # Let's trust QImage.save or ensure extension.
                
                # Simple check: if no extension, append one based on filter? 
                # Or just let user responsible. User requested "suggest formats".
                
                # Check if file_name has extension
                _, ext = os.path.splitext(file_name)
                if not ext:
                    # Try to infer from filter
                    if "PNG" in selected_filter: ext = ".png"
                    elif "TIFF" in selected_filter: ext = ".tiff"
                    elif "JPEG" in selected_filter: ext = ".jpg"
                    elif "BMP" in selected_filter: ext = ".bmp"
                    else: ext = ".png" # Default
                    file_name += ext
                
                self.active_label.current_pixmap.save(file_name)

    def update_status_bar(self, x_coord, y_coord):
        if self.sender() is not self.active_label: return

        if self.stacked_widget.currentWidget() == self.montage_widget and self.montage_shared_state.crosshair_norm_pos:
            norm_pos = self.montage_shared_state.crosshair_norm_pos
            status_text = []
            for i, label in enumerate(self.montage_labels):
                data = label.original_data
                if data is not None:
                     img_h, img_w = data.shape[0], data.shape[1]
                     img_x = int(norm_pos.x() * img_w)
                     img_y = int(norm_pos.y() * img_h)
                     
                     if 0 <= img_x < img_w and 0 <= img_y < img_h:
                        value = data[img_y, img_x]
                        status_text.append(f"Img{i + 1}: ({img_x},{img_y}) {value}")
                     else:
                        status_text.append(f"Img{i + 1}: Out of bounds")

            self.status_bar.showMessage(" | ".join(status_text))
        elif self.active_label and self.active_label.original_data is not None:
            # Use inspection_data which contains the raw values (before visualization conversions)
            data = getattr(self.active_label, 'inspection_data', self.active_label.original_data)
            
            if data is None:
                data = self.active_label.original_data

            if y_coord < data.shape[0] and x_coord < data.shape[1]:
                value = data[y_coord, x_coord]
                self.status_bar.showMessage(f"({x_coord}, {y_coord}): {value}")

    def _update_recent_files_menu(self):
        self.recent_files_menu.clear()
        if not self.recent_files:
            self.recent_files_menu.setEnabled(False)
            return
        self.recent_files_menu.setEnabled(True)
        for file_path in self.recent_files:
            action = QAction(file_path, self)
            # Use *args to safely handle 'triggered' signal which may or may not send 'checked' boolean
            action.triggered.connect(lambda *args, path=file_path: self.open_file(path))
            self.recent_files_menu.addAction(action)

    def update_histogram_data(self, new_image=False):
        if not self.active_label or not self.histogram_window.isVisible():
            return
            
        use_visible_only = self.histogram_window.use_visible_checkbox.isChecked()
        
        # Optimization: If not using visible only (meaning full image), and we didn't just load a new image, 
        # then the histogram data (full image) hasn't changed, so skip update.
        if not use_visible_only and not new_image:
             # We might check if we actually have data in the histogram first?
             # Assuming if not new_image, we already populated it once.
             # However, let's be safe: if histogram is empty, we should run.
             if self.histogram_window.data is not None:
                 return

        visible_data = self._get_visible_image_data(use_visible_only=use_visible_only)
 
        # Only allow the histogram widget to reset the region if we don't have custom limits
        # AND we are being told this is a new image context.
        has_custom_limits = self.active_label.contrast_limits is not None
        should_reset_region = new_image and not has_custom_limits

        self.histogram_window.update_histogram(visible_data, should_reset_region)

        if has_custom_limits:
            min_v, max_v = self.active_label.contrast_limits
            self.histogram_window.min_val_input.setText(f"{min_v:.2f}")
            self.histogram_window.max_val_input.setText(f"{max_v:.2f}")
            
            # Block signals to prevent re-triggering set_contrast_limits
            self.histogram_window.region.blockSignals(True)
            self.histogram_window.region.setRegion([min_v, max_v])
            self.histogram_window.region.blockSignals(False)
        else:
            if visible_data is not None and visible_data.size > 0:
                if len(visible_data.shape) == 3:
                    if visible_data.shape[2] >= 3:
                        hist_data = np.dot(visible_data[..., :3], [0.2989, 0.5870, 0.1140])
                    elif visible_data.shape[2] == 2:
                        hist_data = np.linalg.norm(visible_data, axis=2)
                    else:
                        hist_data = visible_data[:, :, 0]
                else:
                    hist_data = visible_data
                with np.errstate(divide='ignore', invalid='ignore'):
                    if hist_data.size > 0 and np.any(np.isfinite(hist_data)):
                        min_v = np.nanmin(hist_data)
                        max_v = np.nanmax(hist_data)
                    else:
                        min_v = max_v = 0.0
                
                self.histogram_window.min_val_input.setText(f"{min_v:.2f}")
                self.histogram_window.max_val_input.setText(f"{max_v:.2f}")
                
                # Block signals here too just in case
                self.histogram_window.region.blockSignals(True)
                self.histogram_window.region.setRegion([min_v, max_v])
                self.histogram_window.region.blockSignals(False)

    def keyPressEvent(self, event):
        # Universal Copy/Paste: Accept either Control or Meta to handle various OS/Keyboard configs
        # (e.g., standard Mac Meta, or remapped/VNC Mac Control, or standard Windows Control)
        is_cmd_or_ctrl = event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)

        # Copy View State (Cmd+C) or Copy Image (Cmd+Shift+C)
        if event.key() == Qt.Key.Key_C and is_cmd_or_ctrl:
            # Check for Shift -> Copy Image
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                if self.active_label and self.active_label.current_pixmap:
                    clipboard = QApplication.clipboard()
                    clipboard.setImage(self.active_label.current_pixmap.toImage())
                    self.status_bar.showMessage("Image Copied to Clipboard", 2000)
                return

            # No Shift -> Copy View State
            if self.active_label:
                state = self.active_label.get_view_state()
                # Store percentiles for more robust pasting across different images
                if self.active_label.original_data is not None:
                    state['contrast_percentiles'] = self._get_percentiles_from_limits(
                        self.active_label.original_data, 
                        self.active_label.contrast_limits
                    )
                
                try:
                    # Serialize to JSON for System Clipboard
                    json_str = json.dumps(state, cls=NumpyEncoder)
                    
                    mime_data = QMimeData()
                    mime_data.setText(f"IV_STATE:{json_str}")
                    # Note: We purposely do NOT set image data here to keep clipboard clean for state-only transfer.
                    
                    clipboard = QApplication.clipboard()
                    clipboard.setMimeData(mime_data)
                    self.status_bar.showMessage("View State Copied (JSON)", 2000)
                except Exception as e:
                    print(f"Error copying view state: {e}")
                    self.status_bar.showMessage("Copy Failed", 2000)
            return



        # Video Navigation
        if self.image_handler.is_video:
            if event.key() == Qt.Key.Key_Left:
                 self._prev_frame()
                 return
            elif event.key() == Qt.Key.Key_Right:
                 self._next_frame()
                 return
        


        if event.key() == Qt.Key.Key_N and is_cmd_or_ctrl:
            current_pos = self.pos()
            new_pos = current_pos + QPoint(30, 30)
            new_window = ImageViewer(window_list=self.window_list)
            new_window.move(new_pos)
            new_window.show()
            return

        # Single Key Shortcuts - Strict NoModifier Check
        if event.key() == Qt.Key.Key_C and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            if self.stacked_widget.currentWidget() == self.montage_widget:
                is_enabled = not self.montage_labels[0].crosshair_enabled
                for label in self.montage_labels:
                    label.set_crosshair_enabled(is_enabled)
            elif self.active_label:
                self.active_label.set_crosshair_enabled(not self.active_label.crosshair_enabled)
            return

        if self.active_label and self.active_label.original_data is not None:
            if event.key() == Qt.Key.Key_M:
                modifiers = event.modifiers()
                use_visible_only = bool(modifiers & Qt.KeyboardModifier.AltModifier)
                
                if modifiers & Qt.KeyboardModifier.ShiftModifier:
                    self._apply_histogram_preset(5, 95, use_visible_only)
                else:
                    self._apply_histogram_preset(0, 100, use_visible_only)

            if event.key() == Qt.Key.Key_V and event.modifiers() == Qt.KeyboardModifier.NoModifier:
                # Cycle overlay mode: 0 (Hidden) -> 1 (Basename) -> 2 (Full Path) -> 0
                self.overlay_mode = (getattr(self, 'overlay_mode', 0) + 1) % 3
                self._update_overlay_labels()
                return

        super().keyPressEvent(event)


    
    def dragEnterEvent(self, event):
        """Allow the window to accept drops if not handled by children or filter."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        """Necessary for some platforms to maintain the drop action."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def resizeEvent(self, event):
        old_size = event.oldSize()
        new_size = event.size()
        center = self.centralWidget()

        if hasattr(self, '_last_center_width') and self._last_center_width > 0:
            if old_size.isValid() and new_size.width() < old_size.width():
                # Window is shrinking
                delta = old_size.width() - new_size.width()
                
                # Check how much the docks can shrink
                docks = [self.file_explorer_pane, self.info_pane, 
                         self.math_transform_pane, self.thumbnail_pane]
                visible_docks = [d for d in docks if d.isVisible() and not d.isFloating()]
                
                shrink_capacity = sum(max(0, d.width() - d.minimumWidth()) for d in visible_docks)
                
                if delta <= shrink_capacity:
                    # Docks can handle all the shrinkage
                    target_w = self._last_center_width
                else:
                    # Docks handle as much as they can, center takes the rest
                    target_w = self._last_center_width - (delta - shrink_capacity)
                
                # Apply lock to force Qt's hand
                target_w = max(100, int(target_w))
                center.setMinimumWidth(target_w)
                center.setMaximumWidth(target_w)
                QTimer.singleShot(0, self._reset_center_constraints)
        
        super().resizeEvent(event)
        
        # Capture truth for next frame
        self._last_center_width = center.width()

    def _update_thumbnail_pane_from_montage(self):
        """Populate the thumbnail pane with current montage labels"""
        if not self.thumbnail_pane or not self.montage_labels:
            return
        
        # Clear existing thumbnails
        for item in self.thumbnail_pane.thumbnail_items:
            self.thumbnail_pane.thumbnail_layout.removeWidget(item)
            item.deleteLater()
        self.thumbnail_pane.thumbnail_items.clear()
        
        # Add thumbnails for each montage label
        from widgets import ThumbnailItem
        for label in self.montage_labels:
            if label.current_pixmap and hasattr(label, 'file_path'):
                item = ThumbnailItem(label.file_path, label.current_pixmap)
                item.clicked.connect(lambda event, i=item: self.thumbnail_pane._on_thumbnail_clicked(i, event))
                item.overlay_changed.connect(lambda alpha, path=label.file_path: self.thumbnail_pane.overlay_changed.emit(path, alpha))
                self.thumbnail_pane.thumbnail_items.append(item)
                self.thumbnail_pane.thumbnail_layout.addWidget(item)
        
        # Set first item as focused and selected
        if self.thumbnail_pane.thumbnail_items:
            self.thumbnail_pane._set_focused_item(0)
            self.thumbnail_pane._select_single(0)

    def _on_thumbnail_selection_changed(self, selected_files):
        """Handle thumbnail selection by updating montage with selected images"""
        # Block populate calls while we're updating the view from selection
        if hasattr(self.thumbnail_pane, 'block_populate'):
            self.thumbnail_pane.block_populate = True
        
        # If no files selected, clear the montage view
        if not selected_files:
            self.display_montage([], is_manual=False)
            if hasattr(self.thumbnail_pane, 'block_populate'):
                self.thumbnail_pane.block_populate = False
            return
        
        # Handle selections: single file/key or multiple
        self._updating_from_thumbnail = True
        try:
            # Handle selections: single file/key or multiple
            if len(selected_files) == 1:
                # Single selection - use open_file which handles NPZ keys
                self.open_file(selected_files[0])
            else:
                # Multiple selection - pass to display_montage which also handles NPZ keys
                self.display_montage(selected_files)
        finally:
            self._updating_from_thumbnail = False
        
        # Unblock populate
        if hasattr(self.thumbnail_pane, 'block_populate'):
            self.thumbnail_pane.block_populate = False



    def _refresh_thumbnail_pane(self):
        """Refresh thumbnail pane to show ONLY images from the current window"""
        if not self.thumbnail_pane:
            return
        
        # Clear existing thumbnails - remove from layout AND delete
        while self.thumbnail_pane.thumbnail_layout.count():
            item = self.thumbnail_pane.thumbnail_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
                item.widget().deleteLater()
        self.thumbnail_pane.thumbnail_items.clear()
        
        if self.stacked_widget.currentWidget() == self.montage_widget and self.montage_labels:
             # If filter mode active (implicit), might need special handling, but populate() is safest for "Showing what is open"
             pass
        
        # Use the unified populate method which now correctly scans all windows
        self.thumbnail_pane.populate(self.window_list)
        
        # Afterwards, ensure current window's item is selected if single view
        if self.current_file_path:
             for item in self.thumbnail_pane.thumbnail_items:
                 if item.file_path == self.current_file_path:
                     item.set_selected(True)
        
    def _update_thumbnail_selection_states(self):
        """Update selection states of existing thumbnails based on current montage"""
        if not self.thumbnail_pane or not self.thumbnail_pane.thumbnail_items:
            return
        
        # Only force selection if we are in Single View mode (to highlight current)
        # In Montage view, we should respect the ThumbnailPane's own selection state 
        # because the user uses it to DRIVE the montage. "What is selected in thumbnail = What is shown in Montage".
        # We shouldn't invert it to "What is shown in Montage = Force Selected in Thumbnail" constantly, 
        # or it creates a feedback loop where you can't deselect.
        
        if self.stacked_widget.currentWidget() != self.montage_widget and self.current_file_path:
             # Single view: Sync selection to current file
             # But be careful not to annoyingly clear other selections if user is multi-selecting for a future operation?
             # For now, let's just ensure current is selected.
             for item in self.thumbnail_pane.thumbnail_items:
                 if item.file_path == self.current_file_path:
                     item.set_selected(True)
        
        # In Montage mode -> DO NOTHING. 
        # The ThumbnailPane selection drives the Montage (via signal), not vice versa.
        
        # Update Select All checkbox state manually check
        selected_count = sum(1 for item in self.thumbnail_pane.thumbnail_items if item.is_selected)
        total_count = len(self.thumbnail_pane.thumbnail_items)
        
        self.thumbnail_pane.select_all_cb.blockSignals(True)
        if selected_count == 0:
            self.thumbnail_pane.select_all_cb.setCheckState(Qt.CheckState.Unchecked)
        elif selected_count == total_count:
            self.thumbnail_pane.select_all_cb.setCheckState(Qt.CheckState.Checked)
        else:
            self.thumbnail_pane.select_all_cb.setCheckState(Qt.CheckState.PartiallyChecked)
        self.thumbnail_pane.select_all_cb.blockSignals(False)

    def _reset_center_constraints(self):
        center = self.centralWidget()
        if center:
            center.setMinimumWidth(100)
            center.setMaximumWidth(16777215)


    def _apply_histogram_preset(self, min_percent, max_percent, use_visible_only=False):
        visible_data = self._get_visible_image_data(use_visible_only)
        if visible_data is not None and visible_data.size > 0:
            if len(visible_data.shape) == 3:
                if visible_data.shape[2] >= 3:
                    # RGB: compute luminance
                    hist_data = np.dot(visible_data[..., :3], [0.2989, 0.5870, 0.1140])
                elif visible_data.shape[2] == 2:
                    # 2-channel: use magnitude
                    hist_data = np.linalg.norm(visible_data, axis=2)
                else:
                    # Single channel in 3D array
                    hist_data = visible_data[:, :, 0]
            else:
                hist_data = visible_data

            with np.errstate(divide='ignore', invalid='ignore'):
                if hist_data.size > 0 and np.any(np.isfinite(hist_data)):
                    min_val = np.nanpercentile(hist_data[np.isfinite(hist_data)], min_percent)
                    max_val = np.nanpercentile(hist_data[np.isfinite(hist_data)], max_percent)
                else:
                    min_val = max_val = 0.0
            
            self.set_contrast_limits(min_val, max_val)
            
            # Block signals to prevent redundant set_contrast_limits call via region_changed
            self.histogram_window.region.blockSignals(True)
            self.histogram_window.region.setRegion([min_val, max_val])
            self.histogram_window.region.blockSignals(False)

    def _get_visible_image_data(self, use_visible_only=False):
        if not self.active_label or self.active_label.original_data is None:
            return None
        if use_visible_only:
             return self.active_label.get_visible_sub_image()
        return self.active_label.original_data

    def restore_image_view(self):
        if self.active_label:
            # Do not reset data, just the view (zoom/pan)
            # self.active_label.set_data(self.active_label.original_data)
            self.active_label.restore_view()
            self._update_active_view(reset_histogram=False)

    def reset_image_full(self):
        """Completely reset the image to its original state (zoom, pan, contrast, colormap)."""
        if self.active_label and self.active_label.original_data is not None:
             # Explicitly reset state variables
             self.active_label.contrast_limits = None
             
             # Reset data which clears contrast limits
             self.active_label.set_data(self.active_label.original_data, reset_view=True)
             
             # Reset View
             self.active_label.restore_view()
             
             # Reset Colormap
             self.set_colormap("gray")
             self.colormap_combo.blockSignals(True)
             self.colormap_combo.setCurrentText("gray")
             self.colormap_combo.blockSignals(False)
             
             # Reset Histogram/UI
             self._update_active_view(reset_histogram=True)
             
             # Apply default Min-Max contrast again to ensure it's not raw/black
             # Use full range (0-100%) to reset any inherited contrast narrowing
             self._apply_histogram_preset(0, 100, use_visible_only=False)

    def eventFilter(self, source, event):
        # 1. Handle D&D Events (DragEnter, DragMove, Drop)
        if event.type() in (QEvent.Type.DragEnter, QEvent.Type.DragMove, QEvent.Type.Drop):
            # We only care about events within our own window hierarchy
            if not isinstance(source, QWidget) or source.window() != self:
                return False

            # EXPLICITLY handle sidebar drops centrally for higher reliability
            if self.thumbnail_pane and (source == self.thumbnail_pane or self.thumbnail_pane.isAncestorOf(source)):
                if event.mimeData().hasUrls():
                    event.acceptProposedAction()
                    event.accept()
                    if event.type() == QEvent.Type.Drop:
                        urls = event.mimeData().urls()
                        paths = [u.toLocalFile() for u in urls if u.toLocalFile()]
                        if paths:
                            self.thumbnail_pane.add_files(paths)
                    return True
                return False

            # EXPLICITLY ignore file explorer pane to let it handle its own drops
            if self.file_explorer_pane and (source == self.file_explorer_pane or self.file_explorer_pane.isAncestorOf(source)):
                return False

            # For Enter/Move: Force Accept to enable dropping on central widgets
            if event.type() != QEvent.Type.Drop:
                if event.mimeData().hasUrls():
                    event.acceptProposedAction()
                    return True
                return False
            
            # For Drop: Handle Centralized Append
            urls = event.mimeData().urls()
            if urls:
                new_file_paths = []
                supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp', '.heic', '.heif'] + \
                                     self.image_handler.raw_extensions + self.image_handler.video_extensions
                
                for url in urls:
                    path = url.toLocalFile()
                    if os.path.isfile(path):
                        _, ext = os.path.splitext(path)
                        if ext.lower() in supported_extensions:
                            new_file_paths.append(path)
                
                if not new_file_paths:
                    return False
                    
                # Determine existing files to append to
                current_files = []
                if self.stacked_widget.currentWidget() == self.montage_widget and self.montage_labels:
                    current_files = [label.file_path for label in self.montage_labels if label.file_path]
                elif self.current_file_path:
                    current_files = [self.current_file_path]
                
                # Merge new files (allow duplicates for side-by-side comparison)
                combined_paths = current_files + new_file_paths
                
                if len(combined_paths) > 1:
                    self.display_montage(combined_paths, is_manual=True)
                elif len(combined_paths) == 1:
                    self.open_file(combined_paths[0], is_manual=True)
                    
                return True
            return False
        
        # 2. Handle MacOS FileOpen Events
        elif event.type() == QEvent.Type.FileOpen:
            # MacOS specific event for opening files (e.g. from Finder)
            file_path = event.file()
            
            # Check if we can reuse an existing empty window
            target_window = None
            for win in self.window_list:
                if not win.current_file_path:
                    target_window = win
                    break
            
            if target_window:
                target_window.show()
                target_window.raise_()
                target_window.activateWindow()
                target_window.open_file(file_path)
            else:
                # All windows full, create new one
                new_window = self.__class__(window_list=self.window_list)
                new_window.show()
                new_window.open_file(file_path)
            
            return True

        return super().eventFilter(source, event)

    def _invalidate_overlay_cache(self, file_path):
        """Invalidate the cache for a specific file path and refresh overlays.
           Call this when the source image appearance (colormap/contrast) changes.
        """
        # Remove all cache entries where this file_path is the source
        pairs_to_remove = [(src, tgt) for (src, tgt) in self.overlay_cache.keys() if src == file_path]
        for pair in pairs_to_remove:
            del self.overlay_cache[pair]
        self._update_overlays()

    def _on_overlay_changed(self, file_path, alpha):
        # file_path is the SOURCE of the overlay
        # Store it as a pair with the currently active label (the TARGET)
        if self.active_label and hasattr(self.active_label, 'file_path'):
            target_path = self.active_label.file_path
            if target_path and target_path != file_path:  # Don't overlay onto self
                self.overlay_alphas[(file_path, target_path)] = alpha
                self._update_overlays()
            elif alpha == 0:
                # If setting to 0, remove any existing pair
                pairs_to_remove = [(src, tgt) for (src, tgt) in self.overlay_alphas.keys() 
                                   if src == file_path]
                for pair in pairs_to_remove:
                    if pair in self.overlay_alphas:
                        del self.overlay_alphas[pair]
                    if pair in self.overlay_cache:
                        del self.overlay_cache[pair]
                self._update_overlays()

    def _update_overlays(self):
        if not self.active_label or not self.active_label.current_pixmap:
            return

        overlays_to_draw = []
        # target_size should be the ORIGINAL resolution to avoid smearing when zoomed
        # even if the label is using a downsampled proxy for display.
        target_size = self.active_label.get_original_size()
        active_path = getattr(self.active_label, 'file_path', None)
        
        if not active_path or not target_size or target_size.isEmpty():
            self.active_label.set_overlays([])
            return

        # Clear cache if target image changed to prevent alignment/scaling artifacts
        last_target = getattr(self, '_last_overlay_target', None)
        if last_target != active_path:
            self.overlay_cache.clear()
            self._last_overlay_target = active_path

        # Only apply overlays where the active_path is the TARGET
        for (source_path, target_path), alpha in self.overlay_alphas.items():
            if alpha > 0 and target_path == active_path:
                pair = (source_path, target_path)
                if pair not in self.overlay_cache:
                    # Find source pixmap
                    source_pixmap = None
                    
                    # 1. Search Open Windows
                    for win in self.window_list:
                        try:
                            # Check main image label
                            if getattr(win, 'current_file_path', None) == source_path and win.image_label:
                                # Ensure we don't use a downsampled proxy
                                label = win.image_label
                                if label.current_pixmap and getattr(label, '_proxy_scale', 1.0) == 1.0:
                                    source_pixmap = label.current_pixmap
                                    break
                            
                            # Check montage labels
                            if hasattr(win, 'montage_labels'):
                                for label in win.montage_labels:
                                    if hasattr(label, 'file_path') and label.file_path == source_path and label.current_pixmap:
                                         if getattr(label, '_proxy_scale', 1.0) == 1.0:
                                            # Verification: Ensure it's not a tiny thumbnail from somewhere
                                                source_pixmap = label.current_pixmap
                                                break
                                if source_pixmap:
                                    break
                        except:
                            pass
                    
                    if not source_pixmap:

                        # The gallery cache might contain downsampled proxies, so avoiding it for overlays is safer.
                        try:
                             temp_handler = ImageHandler()
                             # Determine if raw to apply overrides? 
                             # For overlays, we usually want the "default" view or we'd need to persist settings perfectly.
                             # Let's try loading.
                             # Check if we have history for this file to respect resolution
                             override = None
                             _, ext = os.path.splitext(source_path)
                             if ext.lower() in temp_handler.raw_extensions:
                                  basename = os.path.basename(source_path)
                                  if not re.search(r"[\-_](\d+)x(\d+)", basename):
                                      history = settings.load_raw_history()
                                      if source_path in history:
                                          override = history[source_path]
                             
                             temp_handler.load_image(source_path, override_settings=override)
                             
                             if temp_handler.original_image_data is not None:
                                  data = temp_handler.original_image_data
                                  if data.ndim == 2:
                                      h, w = data.shape
                                      # Normalize to uint8
                                      img_8bit = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
                                      qimg = QImage(img_8bit.data, w, h, w, QImage.Format.Format_Grayscale8)
                                      source_pixmap = QPixmap.fromImage(qimg)
                                  elif data.ndim == 3:
                                      if data.shape[2] == 3:
                                          h, w, _ = data.shape
                                          if not data.flags['C_CONTIGUOUS']:
                                               data = np.ascontiguousarray(data)
                                          qimg = QImage(data.data, w, h, 3*w, QImage.Format.Format_RGB888)
                                          source_pixmap = QPixmap.fromImage(qimg)
                                      elif data.shape[2] == 4:
                                          h, w, _ = data.shape
                                          if not data.flags['C_CONTIGUOUS']:
                                               data = np.ascontiguousarray(data)
                                          qimg = QImage(data.data, w, h, 4*w, QImage.Format.Format_RGBA8888)
                                          source_pixmap = QPixmap.fromImage(qimg)
                                      
                        except Exception:
                             pass
                    
                    if source_pixmap:
                        self.overlay_cache[pair] = source_pixmap
               
                if pair in self.overlay_cache:
                    overlays_to_draw.append((self.overlay_cache[pair], alpha))

        # Use set_overlays() method which expects a list of (pixmap, alpha) tuples
        self.active_label.set_overlays(overlays_to_draw)

    def closeEvent(self, event):
        """Remove window from the list when closed."""
        try:
            QApplication.instance().removeEventFilter(self)
        except Exception:
            pass
            
        # Call super first to let C++ handle its closure
        super().closeEvent(event)
        
        # Defer removal from global list to ensure Python object stays alive 
        # until the C++ event handling is completely finished.
        QTimer.singleShot(0, self._cleanup_window_list)

    def _cleanup_window_list(self):
        if hasattr(self, 'window_list') and self in self.window_list:
            self.window_list.remove(self)
            
            # Notify all other windows to refresh their thumbnail pane
            for win in self.window_list:
                try:
                    if hasattr(win, '_refresh_thumbnail_pane'):
                        win._refresh_thumbnail_pane()
                except RuntimeError:
                    pass