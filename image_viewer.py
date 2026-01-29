import os
import time
import re

import numpy as np
from PyQt6.QtCore import Qt, QEvent, QPoint, QPointF, QTimer
from PyQt6.QtGui import QAction, QPixmap, QImage, QIcon
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
)
import matplotlib.cm as cm

from widgets import ZoomableDraggableLabel, InfoPane, MathTransformPane, ZoomSettingsDialog, HistogramWidget, \
    ThumbnailPane, SharedViewState, FileExplorerPane, PointCloudViewer
from image_handler import ImageHandler
import settings
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)



class ImageViewer(QMainWindow):
    view_clipboard = None  # Class-level clipboard for cross-window sharing

    def __init__(self, window_list):
        super().__init__()
        self.setWindowTitle("Image Viewer")
        QApplication.instance().installEventFilter(self)

        self.window_list = window_list
        self.window_list.append(self)
        
        # Enable Drag and Drop
        self.setAcceptDrops(True)

        self.image_handler = ImageHandler()
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self._video_timer_timeout)
        self.is_slider_pressed = False
        
        self.shared_state = SharedViewState()
        self.shared_state.zoom_changed.connect(self._on_shared_zoom_changed)
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
        self.zoom_settings = {"zoom_speed": 1.1, "zoom_in_interp": "Nearest", "zoom_out_interp": "Smooth"}

        self._create_welcome_screen()
        self._create_image_display()
        self._create_montage_view()



        self._create_menus_and_toolbar()
        self._create_file_explorer_pane()
        self._create_info_pane()
        self._create_math_transform_pane()
        self._create_histogram_window()
        self._create_thumbnail_pane()

        # Force initial docking ratio if possible
        self.resizeDocks([self.file_explorer_pane], [225], Qt.Orientation.Horizontal)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.zoom_status_label = QLabel("Zoom: 100%")
        self.status_bar.addPermanentWidget(self.zoom_status_label)


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
        except:
            return (0.0, 1.0)

    def _create_welcome_screen(self):
        welcome_widget = QWidget()
        welcome_layout = QVBoxLayout(welcome_widget)
        welcome_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label = QLabel("Image Viewer")
        font = title_label.font()
        font.setPointSize(24)
        title_label.setFont(font)
        open_button = QPushButton("Open Image")
        open_button.clicked.connect(self.open_image_dialog)

        drag_drop_label = QLabel("(or drag and drop an image file here)")
        drag_drop_label.setStyleSheet("color: gray; font-style: italic;")

        welcome_layout.addWidget(title_label)
        welcome_layout.addWidget(open_button)
        welcome_layout.addWidget(drag_drop_label)
        self.stacked_widget.addWidget(welcome_widget)

    def _create_image_display(self):
        self.image_label = ZoomableDraggableLabel()
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(0, 5000)
        self.zoom_slider.setValue(2000)
        self.zoom_slider.setTickInterval(1000)
        self.zoom_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.image_display_container = QWidget()
        policy = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Expanding)
        policy.setHorizontalStretch(100)
        self.image_display_container.setSizePolicy(policy)
        image_display_layout = QVBoxLayout(self.image_display_container)
        image_display_layout.addWidget(self.image_label)
        image_display_layout.addWidget(self.zoom_slider)
        self.image_label.setSizePolicy(policy) # Also on the label itself


        self.stacked_widget.addWidget(self.image_display_container)
        self.apply_zoom_settings()  # Apply default settings immediately

    def _create_montage_view(self):
        self.montage_widget = QWidget()
        policy = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Expanding)
        policy.setHorizontalStretch(100)
        self.montage_widget.setSizePolicy(policy)
        self.montage_layout = QGridLayout(self.montage_widget)
        self.stacked_widget.addWidget(self.montage_widget)

    def _create_menus_and_toolbar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        open_action = QAction(QIcon.fromTheme("document-open"), "&Open", self)
        open_action.triggered.connect(self.open_image_dialog)
        file_menu.addAction(open_action)
        save_action = QAction("&Save View", self)
        save_action.triggered.connect(self.save_view)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        self.recent_files_menu = file_menu.addMenu("Recent Files")
        self._update_recent_files_menu()

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
        
        # Channel Selector
        self.channel_combo = QComboBox(self)
        self.channel_combo.addItem("Default")
        self.channel_combo.currentTextChanged.connect(self.update_image_display)
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

        # Custom "3D" Button with separate colors for '3' (Green) and 'D' (Red)
        self.threed_button = QWidget()
        self.threed_button.setFixedSize(40, 32)
        self.threed_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.threed_button.setToolTip("Open 3D View")
        
        container_layout = QHBoxLayout(self.threed_button)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        
        self.label_3d_3 = QLabel("3")
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
        
        image = self.image_handler.original_image_data
        if image is None:
            self.channel_combo.addItem("Default")
            self.channel_combo.blockSignals(False)
            return

        # Check if this is an NPZ file with multiple keys
        if hasattr(self.image_handler, 'npz_keys') and len(self.image_handler.npz_keys) > 1:
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
        if not self.active_label or self.image_handler.original_image_data is None:
            return
        
        # 1. Get the base data (original or current video frame)
        data = self.image_handler.original_image_data
        
        if data is None:
            return
        
        # 2. Apply Channel Selection
        sliced_data = self.apply_channel_selection(data)
        
        # Enable 3D View only for single channel images
        is_single_channel = (sliced_data.ndim == 2) or (sliced_data.ndim == 3 and sliced_data.shape[2] == 1)
        if hasattr(self, 'threed_button'):
             self.threed_button.setEnabled(is_single_channel)
             # Update styling to look disabled/enabled
             if is_single_channel:
                 self.threed_button.setToolTip("Open 3D View")
                 # Restore colors
                 self.label_3d_3.setStyleSheet("color: #00E676; font-weight: 900; font-size: 16px; font-family: 'Arial'; margin-right: -5px;")
                 self.label_3d_d.setStyleSheet("color: #FF1744; font-weight: 900; font-size: 16px; font-family: 'Arial'; margin-left: -5px;")
             else:
                 self.threed_button.setToolTip("3D View available only for single-channel images")
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
        self.image_label.set_data(sliced_data, reset_view=reset_view, is_pristine=True, inspection_data=inspection_data)
        
        # 6. Apply Math Transform if exists
        if self.current_math_expression:
            self.apply_math_transform(self.current_math_expression, from_update=True)
        
        # 7. Trigger repaint and histogram update
        self.image_label.repaint()
        self.update_histogram_data()
        
        # 8. Update 3D View if open
        if hasattr(self, 'point_cloud_viewer') and self.point_cloud_viewer and self.point_cloud_viewer.isVisible():
             self.point_cloud_viewer.set_data(sliced_data)
        


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
        extensions = self.image_handler.raw_extensions + ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif', '.webp'] + self.image_handler.video_extensions
        # Create unique list
        # Create unique list
        extensions = list(set(extensions))
        
        # Create case-insensitive filters (add both lowercase and uppercase)
        ext_filters = []
        for ext in extensions:
            ext_filters.append('*' + ext.lower())
            ext_filters.append('*' + ext.upper())
            
        self.file_explorer_pane.set_supported_extensions(ext_filters)

        self.overlay_alphas = {} # path -> alpha
        self.overlay_cache = {}  # path -> QPixmap (resized to match current active view)

    def display_montage(self, file_paths):
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
            overlays_to_remove = [path for path in self.overlay_alphas.keys() if path not in file_paths_set]
            for path in overlays_to_remove:
                del self.overlay_alphas[path]
                if path in self.overlay_cache:
                    del self.overlay_cache[path]

        if not file_paths:
            if self.current_file_path:
                # If we have a current file but nothing is selected, clearing the view is safer
                # than re-opening the unselected file.
                pass
            
            # Clear layout and show welcome/empty
            self.montage_labels.clear()
            self._update_active_view()
            self.stacked_widget.setCurrentWidget(self.welcome_widget)
            self._refresh_thumbnail_pane() # Ensure thumbnails are cleared
            return


        self.montage_shared_state = SharedViewState()
        max_montage_zoom = 100.0

        row, col = 0, 0
        override_settings = getattr(self, '_temp_montage_override', None)
        
        for file_path in file_paths:
            temp_handler = ImageHandler()
            
            # Smart Override: Only apply raw settings if the file itself is raw
            # This prevents trying to read a PNG using raw dimensions/settings
            current_override = override_settings
            if current_override:
                _, ext = os.path.splitext(file_path)
                if ext.lower() not in temp_handler.raw_extensions:
                    current_override = None

            try:
                temp_handler.load_image(file_path, override_settings=current_override)
            except:
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

                image_label.set_overlay_text(file_path)
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
        
        # Update thumbnail pane strictly
        # self._refresh_thumbnail_pane() # REMOVED: This causes a feedback loop where unselecting an item (which updates montage) rebuilding the thumbnail pane and potentially resetting selection.
        # The thumbnail selection drives the montage, not vice versa.
        pass

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
            self.zoom_slider.valueChanged.disconnect()
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

        # Connect signals for the active label
        self.active_label.zoom_factor_changed.connect(self._on_image_label_zoom_changed)
        self.zoom_slider.valueChanged.connect(self._on_zoom_slider_changed)
        self.active_label.hover_moved.connect(self.update_status_bar)
        self.active_label.view_changed.connect(self.update_histogram_data)
        self.histogram_window.region_changed.connect(self.set_contrast_limits)

        self.colormap_combo.setEnabled(True)
        
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
        
        # Update Overlays for ID identification
        is_montage = self.stacked_widget.currentWidget() == self.montage_widget
        
        if is_montage:
             for i, label in enumerate(self.montage_labels):
                 if visible:
                     label.set_overlay_text(f"x{i+1}")
                     label.overlay_label.show()
                 elif hasattr(label, 'file_path'):
                     label.set_overlay_text(label.file_path)
                     # Keep visible or revert to default? Assuming filenames usually hidden or explicitly shown.
                     # If we forced show, maybe we should respect prior state?
                     # Or just hide if it wasn't typical. 
                     # Code check: display_montage sets text but doesn't show. So default is Hidden.
                     label.overlay_label.hide()
        else:
             if self.active_label:
                 if visible:
                     self.active_label.set_overlay_text("x")
                     self.active_label.overlay_label.show()
                 else:
                     self.active_label.overlay_label.hide()

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
            self.update_histogram_data(new_image=True)
            self.histogram_window.show()
            
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

    def _on_zoom_slider_changed(self, value):
        if self.sender() is not self.zoom_slider: return
        if self.active_label:
            zoom_factor = 10 ** ((value / 1000.0) - 2.0)
            if abs(self.active_label._get_effective_scale_factor() - zoom_factor) > 1e-5:
                # Use setter which updates shared state if active
                self.active_label.zoom_scale = zoom_factor

    def _on_shared_zoom_changed(self, multiplier):
        # Triggered when shared state changes (e.g. via mouse wheel on image)
        if self.active_label and not self.zoom_slider.isSliderDown():
             zoom = self.active_label._get_effective_scale_factor()
             if zoom > 0:
                 val = (np.log10(zoom) + 2.0) * 1000.0
                 self.zoom_slider.blockSignals(True)
                 self.zoom_slider.setValue(int(val))
                 self.zoom_slider.blockSignals(False)

    def _on_image_label_zoom_changed(self, scale_factor):
        if self.sender() is not self.active_label: return
        if scale_factor > 0:
            slider_value = int((np.log10(scale_factor) + 2.0) * 1000.0)
            self.zoom_slider.blockSignals(True)
            self.zoom_slider.setValue(slider_value)
            self.zoom_slider.blockSignals(False)
            
            self.zoom_status_label.setText(f"Zoom: {int(scale_factor * 100)}%")

    def reapply_raw_parameters(self, raw_settings):
        # Renamed argument to raw_settings to avoid shadowing global settings module if imported
        try:
            self.image_handler.load_image(self.current_file_path, override_settings=raw_settings)
            if self.active_label:
                self.active_label.set_data(self.image_handler.original_image_data)
            self.update_histogram_data(new_image=True)
            
            # Save to history if this file lacks explicit resolution
            width, height, _ = self.image_handler.parse_resolution(self.current_file_path)
            if width == 0 or height == 0:
                settings.update_raw_history(self.current_file_path, raw_settings)
            
            if self.active_label:
                self.active_label.repaint()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying parameters:\n{e}")

    def apply_math_transform(self, expression, from_update=False):
        self.current_math_expression = expression
        try:
            context = {}
            if self.stacked_widget.currentWidget() == self.montage_widget:
                for i, label in enumerate(self.montage_labels):
                    if label.pristine_data is not None:
                        context[f"x{i+1}"] = label.pristine_data.astype(np.float64)
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
             
        # Check dimensionality and warn?
        data = self.active_label.original_data
        
        # Instantiate if needed
        # We don't cache it as a permanent member to avoid holding OpenGL contexts if closed?
        # Actually reusing window is better UI.
        if not hasattr(self, 'point_cloud_viewer') or self.point_cloud_viewer is None:
             from widgets import PointCloudViewer # Lazy or ensure it's there
             self.point_cloud_viewer = PointCloudViewer(self)
             
        self.point_cloud_viewer.set_data(data)
        self.point_cloud_viewer.show()
        self.point_cloud_viewer.activateWindow()

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
                 self.display_montage([])
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
            self.open_file(target_path, override_settings=effective_raw, maintain_view_state=view_state)
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
            self.open_file(file_path)

    def open_file(self, file_path=None, override_settings=None, maintain_view_state=None):
        """Load an image or video file."""
        if file_path is None:
            return
        
        # Handle NPZ key paths (format: "file.npz#key_name")
        npz_key_to_switch = None
        if '#' in file_path:
            actual_file, key_name = file_path.rsplit('#', 1)
            npz_key_to_switch = key_name
            file_path = actual_file
        
        self.current_file_path = file_path
        
        # Clear overlays for files other than the current one
        overlays_to_remove = [path for path in self.overlay_alphas.keys() if path != file_path]
        for path in overlays_to_remove:
            del self.overlay_alphas[path]
            if path in self.overlay_cache:
                del self.overlay_cache[path]
        
        # Update File Explorer Path
        if self.file_explorer_pane.isVisible():
             self.file_explorer_pane.set_root_path(file_path)

        # Determine if it's a raw file
        _, ext = os.path.splitext(file_path)
        is_raw = ext.lower() in self.image_handler.raw_extensions
        
        # Get File Size
        if os.path.exists(file_path):
             file_size = os.path.getsize(file_path)
        else:
             file_size = 0

        # [REFINEMENT] Inherit settings only if target file has no explicit settings (filename or history)
        if is_raw and override_settings:
            basename = os.path.basename(file_path)
            has_explicit = bool(re.search(r"_(\d+)x(\d+)", basename))
            if not has_explicit:
                history = settings.load_raw_history()
                if file_path in history:
                    has_explicit = True
            
            if has_explicit:
                # Target has its own explicit/historical settings, ignore inherited override
                override_settings = None

        # Check resolution for raw files BEFORE attempting load
        if is_raw and not override_settings: # Only guess/check history if no explicit override_settings provided
             # This uses the new method which returns (0,0) instead of raising
             width, height, dtype_raw = self.image_handler.parse_resolution(file_path)
             
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
                 except:
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
                     self.info_action.setEnabled(True)
                     self.info_pane.set_raw_mode(True)
                     self.math_transform_action.setEnabled(True)
                     self.zoom_slider.setEnabled(True)
                     self.histogram_action.setEnabled(True)
                     
                     self.stacked_widget.setCurrentWidget(self.image_display_container)
                     QApplication.processEvents() # Force layout update
                     self._set_active_montage_label(self.image_label)
                     self.update_image_display(reset_view=True)
                     self._apply_histogram_preset(0, 100)
                     self.overlay_cache.clear()
                     self._update_overlays() 
                     
                     self._update_overlays() 
                     
                     self.recent_files = settings.add_to_recent_files(self.recent_files, file_path)
                     self._update_recent_files_menu()
                     self.image_label.set_overlay_text(file_path)
                        
                     # Save successful load to history
                     settings.update_raw_history(file_path, guess_settings)
                     
                     # Synchronous update
                     self.image_label.repaint()
                     if self.histogram_action.isChecked():
                         self.histogram_window.repaint()
                     
                     # Update channel options based on loaded image
                     self.update_channel_options()
                     
                     # Don't refresh thumbnail pane - gallery persists and this causes selection loss
                     # self._refresh_thumbnail_pane()
                     
                 except Exception as e:
                     pass
                     
                 return # Stop here, as we've handled the load attempt manually

        # Proceed to load image (Standard or Raw with Resolution or Override)
        try:     
            # If override_settings is passed (from History, Guess, or Explorer Inheritance), use it.
            self.image_handler.load_image(file_path, override_settings=override_settings)

            self.info_action.setEnabled(self.image_handler.is_raw)
            self.info_pane.set_raw_mode(self.image_handler.is_raw)
            self.math_transform_action.setEnabled(True)
            self.zoom_slider.setEnabled(True)
            self.histogram_action.setEnabled(True)

            if self.image_handler.is_raw:
                self.info_pane.update_info(self.image_handler.width, self.image_handler.height,
                                           self.image_handler.dtype, self.image_handler.dtype_map,
                                           file_size=file_size)
            else:
                self.info_pane.hide()
            
            # Load Data (triggers render with set attributes)
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
    
            self.recent_files = settings.add_to_recent_files(self.recent_files, file_path)
            self._update_recent_files_menu()
            self.image_label.set_overlay_text(file_path)
                
            # Save override to history if successful (and if it was a missing resolution file)
            # Logic: If we used override_settings, we should save it? 
            # If explicit resolution exists, self.image_handler.parse_resolution returns values.
            # If we passed override settings, we don't necessarily update history unless it was "missing".
            # But implementation plan says "update history on success".
            # Let's check:
            w, h, _ = self.image_handler.parse_resolution(file_path)
            if (w == 0 or h == 0) and override_settings:
                  settings.update_raw_history(file_path, override_settings)

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

            # Don't refresh thumbnail pane - gallery persists and this causes selection loss
            # self._refresh_thumbnail_pane()

        except Exception as e:
            if is_raw and override_settings:
                # Transform/Inheritance caused a mismatch? Fallback to guessing.
                self.open_file(file_path, override_settings=None, maintain_view_state=maintain_view_state)
                return
            
            QMessageBox.critical(self, "Error", f"Error opening image:\n{e}")
            self.math_transform_action.setEnabled(False)
            self.info_action.setEnabled(False)
            self.info_pane.set_raw_mode(False)
            self.zoom_slider.setEnabled(False)
            self.histogram_action.setEnabled(False)

    def save_view(self):
        if self.active_label and self.active_label.current_pixmap:
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Image (*.png)")
            if file_name:
                self.active_label.current_pixmap.save(file_name, "png")

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
                min_v, max_v = np.min(hist_data), np.max(hist_data)
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

        # Copy View State
        if event.key() == Qt.Key.Key_C and is_cmd_or_ctrl:
            if self.active_label:
                ImageViewer.view_clipboard = self.active_label.get_view_state()
                self.status_bar.showMessage("View Copied", 2000)
            return

        # Paste View State
        if event.key() == Qt.Key.Key_V and is_cmd_or_ctrl:
            if self.active_label and ImageViewer.view_clipboard:
                self.active_label.set_view_state(ImageViewer.view_clipboard)
                self._update_active_view(reset_histogram=False) # Update UI to reflect new state
                self.status_bar.showMessage("View Pasted", 2000)
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
                if self.stacked_widget.currentWidget() == self.montage_widget:
                    for label in self.montage_labels:
                        label.toggle_overlay()
                elif self.active_label:
                    self.active_label.toggle_overlay()
                return

        super().keyPressEvent(event)


    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if os.path.exists(file_path):
                self.load_image(file_path)
    
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
            # Clear montage
            for i in reversed(range(self.montage_layout.count())):
                widget = self.montage_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            self.montage_labels.clear()
            self.active_label = None
            if hasattr(self.thumbnail_pane, 'block_populate'):
                self.thumbnail_pane.block_populate = False
            return
        
        # Rebuild montage with selected files
        if len(selected_files) == 1:
            self.open_file(selected_files[0])
        else:
            self.display_montage(selected_files)
        
        # Unblock populate
        if hasattr(self.thumbnail_pane, 'block_populate'):
            self.thumbnail_pane.block_populate = False

    def _update_thumbnail_pane_for_single_image(self):
        """Deprecated: Use _refresh_thumbnail_pane instead to show all windows."""
        self._refresh_thumbnail_pane()

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

            min_val = np.percentile(hist_data, min_percent)
            max_val = np.percentile(hist_data, max_percent)
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
        if event.type() == QEvent.Type.DragEnter:
            # Check if the event target belongs to this window
            if isinstance(source, QWidget) and source.window() == self:
                if event.mimeData().hasUrls():
                    urls = event.mimeData().urls()
                    if urls:
                        file_path = urls[0].toLocalFile()
                        _, ext = os.path.splitext(file_path)
                        supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp',
                                                '.tiff', '.tif'] + self.image_handler.raw_extensions + self.image_handler.video_extensions
                        if ext.lower() in supported_extensions:
                            event.acceptProposedAction()
        elif event.type() == QEvent.Type.Drop:
            # Check if the event target belongs to this window
            if isinstance(source, QWidget) and source.window() == self:
                urls = event.mimeData().urls()
                if urls:
                    file_path = urls[0].toLocalFile()
                    self.open_file(file_path)
                    return True
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
        if file_path in self.overlay_cache:
            del self.overlay_cache[file_path]
            self._update_overlays()

    def _on_overlay_changed(self, file_path, alpha):
        self.overlay_alphas[file_path] = alpha
        self._update_overlays()

    def _update_overlays(self):
        if not self.active_label or not self.active_label.current_pixmap:
            return

        overlays_to_draw = []
        target_size = self.active_label.current_pixmap.size()

        for path, alpha in self.overlay_alphas.items():
            if alpha > 0:
                if path not in self.overlay_cache:
                    # Find the source window with this image and use its current_pixmap (includes all modifications)
                    source_pixmap = None
                    for win in self.window_list:
                        try:
                            # Check if this window has the overlay image loaded in single view
                            if getattr(win, 'current_file_path', None) == path and win.image_label and win.image_label.current_pixmap:
                                source_pixmap = win.image_label.current_pixmap
                                break
                            # Also check montage labels
                            if hasattr(win, 'montage_labels'):
                                for label in win.montage_labels:
                                    if hasattr(label, 'file_path') and label.file_path == path and label.current_pixmap:
                                        source_pixmap = label.current_pixmap
                                        break
                                if source_pixmap:
                                    break
                        except:
                            pass
                    
                    if source_pixmap:
                        # Use the already-processed pixmap from the source window
                        scaled_pixmap = source_pixmap.scaled(target_size, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
                        self.overlay_cache[path] = scaled_pixmap
                    else:
                        # Fallback: load raw image if no source window found
                        try:
                            handler = ImageHandler()
                            handler.load_image(path)
                            if handler.original_image_data is not None:
                                height, width = handler.original_image_data.shape[:2]
                                data = handler.original_image_data
                            
                                # Retrieve state from source window
                                target_cmap = 'gray'
                                target_limits = None
                            for win in self.window_list:
                                try:
                                    # Skip windows in Montage Mode as their current_file_path might be stale/ambiguous
                                    if win.stacked_widget.currentWidget() == win.montage_widget:
                                        continue
                                        
                                    if getattr(win, 'current_file_path', None) == path and win.active_label:
                                        target_cmap = win.active_label.colormap
                                        target_limits = win.active_label.contrast_limits
                                        break
                                except: pass

                            # Normalization
                            if data.dtype != np.uint8:
                                data = data.astype(np.float32)

                            if len(data.shape) == 2: # Grayscale processing
                                if target_limits:
                                    min_v, max_v = target_limits
                                    if max_v > min_v:
                                        norm_data = (data - min_v) / (max_v - min_v)
                                        norm_data = np.clip(norm_data, 0, 1)
                                    else:
                                        norm_data = np.zeros_like(data, dtype=np.float32)
                                else:
                                    # Auto-Contrast
                                    dmin, dmax = np.min(data), np.max(data)
                                    if dmax > dmin:
                                        norm_data = (data - dmin) / (dmax - dmin)
                                    else:
                                        norm_data = np.zeros_like(data, dtype=np.float32)
                                
                                # Apply Colormap
                                if target_cmap != 'gray':
                                    try:
                                        colored_data = cm.get_cmap(target_cmap)(norm_data)
                                        # colored_data is (H, W, 4) float 0..1
                                        # Convert to RGB (H, W, 3)
                                        data = (colored_data[:, :, :3] * 255).astype(np.uint8)
                                    except Exception as e:
                                        print(f"Colormap error: {e}")
                                        data = (norm_data * 255).astype(np.uint8)
                                else:
                                    data = (norm_data * 255).astype(np.uint8)

                            else: # RGB Image
                                # Apply Contrast Limits or Auto-Contrast
                                if target_limits:
                                     min_v, max_v = target_limits
                                     if max_v > min_v:
                                          norm_data = (data - min_v) / (max_v - min_v)
                                          norm_data = np.clip(norm_data, 0, 1)
                                          data = (norm_data * 255).astype(np.uint8)
                                     else:
                                          data = np.zeros_like(data, dtype=np.uint8)
                                else:
                                     # Auto-Contrast (Global Min/Max)
                                     dmin, dmax = np.min(data), np.max(data)
                                     if dmax > dmin:
                                          if data.dtype == np.uint8 and dmin == 0 and dmax == 255:
                                               pass # Already full range
                                          else:
                                               norm_data = (data - dmin) / (dmax - dmin)
                                               data = (norm_data * 255).astype(np.uint8)
                                     else:
                                          if data.max() <= 1.0: 
                                              data = (data * 255).astype(np.uint8)
                                          else:
                                              # Fallback cast
                                              data = data.astype(np.uint8)

                                q_img = QImage(data.data, width, height, 3 * width, QImage.Format.Format_RGB888)
                            
                            # Scaling
                            # We must match the target_size exactly
                            # Ensure data is contiguous
                            if not data.flags['C_CONTIGUOUS']:
                                data = np.ascontiguousarray(data)

                            # QImage reference safety: Keep 'data' alive during QImage lifespan
                            # Using partial function or just ensuring scope is sufficient here as QPixmap.fromImage copies
                            q_img = None 
                            if len(data.shape) == 2:
                                q_img = QImage(data.data, width, height, width, QImage.Format.Format_Grayscale8)
                            else:
                                q_img = QImage(data.data, width, height, 3 * width, QImage.Format.Format_RGB888)
                            
                            # Explicit copy to QPixmap to detach from numpy buffer
                            scaled_pixmap = QPixmap.fromImage(q_img).scaled(target_size, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
                            self.overlay_cache[path] = scaled_pixmap
                        except Exception as e:
                            print(f"Error loading overlay {path}: {e}")
                            continue

                if path in self.overlay_cache:
                    overlays_to_draw.append((self.overlay_cache[path], alpha))

        if self.active_label:
            self.active_label.set_overlays(overlays_to_draw)
        
        # Also update all montage labels if in montage mode
        if self.montage_labels:
            for label in self.montage_labels:
                if label != self.active_label: # Active already updated
                    label.set_overlays(overlays_to_draw)

    def closeEvent(self, event):
        """Remove window from the list when closed."""
        try:
            QApplication.instance().removeEventFilter(self)
        except:
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