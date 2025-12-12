import numpy as np
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QPointF, QEvent, QObject, QTimer, QRectF, QSettings
from PyQt6.QtGui import QPixmap, QPainter, QNativeGestureEvent, QDoubleValidator, QKeyEvent, QImage, QMouseEvent, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QDialog,
    QSlider,
    QComboBox,
    QDialogButtonBox,
    QFormLayout,
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QHBoxLayout,
    QMessageBox,
    QGridLayout,
    QScrollArea,
    QSizePolicy,
    QCheckBox
)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
import pyqtgraph as pg
import os
import matplotlib.cm as cm


class SharedViewState(QObject):
    """An object to hold and synchronize view parameters for multiple labels."""
    view_changed = pyqtSignal()
    crosshair_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._zoom_multiplier = 1.0
        self._offset = QPointF()
        self._zoom_multiplier = 1.0
        self._offset = QPointF()
        self._crosshair_pos = None
        self.max_zoom_limit = 1000.0
        
        # Interaction State for Performance Optimization
        self._is_interacting = False
        self._interaction_timer = QTimer()
        self._interaction_timer.setInterval(200) # 200ms debounce
        self._interaction_timer.setSingleShot(True)
        self._interaction_timer.timeout.connect(self._end_interaction)

    def begin_interaction(self):
        if not self._is_interacting:
            self._is_interacting = True
            # We don't need to force update here, the action itself (zoom/pan) will trigger it
        
        self._interaction_timer.start()

    def _end_interaction(self):
        self._is_interacting = False
        self.view_changed.emit() # Trigger final update for high quality

    @property
    def zoom_multiplier(self):
        return self._zoom_multiplier

    @zoom_multiplier.setter
    def zoom_multiplier(self, value):
        if self._zoom_multiplier != value:
            self._zoom_multiplier = value
            self.view_changed.emit()

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value):
        if self._offset != value:
            self._offset = value
            self.view_changed.emit()

    @property
    def crosshair_pos(self):
        return self._crosshair_pos

    @crosshair_pos.setter
    def crosshair_pos(self, value):
        if self._crosshair_pos != value:
            self._crosshair_pos = value
            self.state_changed.emit()

    def reset(self):
        self._zoom_multiplier = 1.0
        self._offset = QPointF()
        self.view_changed.emit()

    @property
    def crosshair_norm_pos(self):
        return self._crosshair_pos # Now storing normalized pos (QPointF 0-1)

    @crosshair_norm_pos.setter
    def crosshair_norm_pos(self, value):
        if self._crosshair_pos != value:
            self._crosshair_pos = value
            self.crosshair_changed.emit()


class HistogramWidget(QWidget):
    region_changed = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Histogram")
        self.setWindowFlags(Qt.WindowType.Tool)
        self.setWindowOpacity(0.85)

        self.layout = QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget)

        self.region = pg.LinearRegionItem(values=[0, 255], orientation='vertical')
        self.region.sigRegionChanged.connect(self._on_region_changed)
        self.plot_widget.addItem(self.region)

        self.histograms = []
        self.data = None

        # Use visible area option
        self.use_visible_checkbox = QCheckBox("Use Visible Area")
        self.use_visible_checkbox.setChecked(True)
        self.layout.addWidget(self.use_visible_checkbox)

        # Preset buttons
        presets_layout = QHBoxLayout()
        self.min_max_btn = QPushButton("Min-Max")
        self.p5_95_btn = QPushButton("5%-95%")
        self.p1_99_btn = QPushButton("1%-99%")
        presets_layout.addWidget(self.min_max_btn)
        presets_layout.addWidget(self.p5_95_btn)
        presets_layout.addWidget(self.p1_99_btn)
        self.layout.addLayout(presets_layout)

        self.min_max_btn.clicked.connect(lambda: self.set_region_to_percentile(0, 100))
        self.p5_95_btn.clicked.connect(lambda: self.set_region_to_percentile(5, 95))
        self.p1_99_btn.clicked.connect(lambda: self.set_region_to_percentile(1, 99))

        # Custom value input
        value_input_layout = QHBoxLayout()
        self.min_val_label = QLabel("Min Value:")
        self.min_val_input = QLineEdit("0")
        self.max_val_label = QLabel("Max Value:")
        self.max_val_input = QLineEdit("255")
        self.apply_value_btn = QPushButton("Apply Values")
        self.apply_value_btn.clicked.connect(self._apply_custom_values)

        value_input_layout.addWidget(self.min_val_label)
        value_input_layout.addWidget(self.min_val_input)
        value_input_layout.addWidget(self.max_val_label)
        value_input_layout.addWidget(self.max_val_input)
        value_input_layout.addWidget(self.apply_value_btn)
        self.layout.addLayout(value_input_layout)

    def update_histogram(self, data, new_image=False):
        for item in self.histograms:
            self.plot_widget.removeItem(item)
        self.histograms.clear()

        if data is None or data.size == 0:
            self.data = None
            return

        self.data = data.astype(np.float64)

        if len(self.data.shape) == 3 and self.data.shape[2] == 3:  # RGB
            colors = [(255, 0, 0, 150), (0, 255, 0, 150), (0, 0, 255, 150)]
            for i in range(3):
                channel_data = self.data[..., i].flatten()
                y, x = np.histogram(channel_data, bins=256, range=(0, 256))
                hist = self.plot_widget.plot(x, y, stepMode="center", fillLevel=0, brush=colors[i])
                self.histograms.append(hist)
            # Use luminance for percentile calculations
            self.data = np.dot(self.data[..., :3], [0.2989, 0.5870, 0.1140])
        else:  # Grayscale
            self.data = self.data.flatten()
            y, x = np.histogram(self.data, bins=256)
            hist = self.plot_widget.plot(x, y, stepMode="center", fillLevel=0, brush=(200, 200, 200, 150))
            self.histograms.append(hist)

        min_val, max_val = np.min(self.data), np.max(self.data)
        if min_val < max_val:
            self.plot_widget.setXRange(min_val, max_val, padding=0.05)
            # Do NOT set bounds here. If we are looking at visible area, global contrast might be outside.
            # self.region.setBounds([min_val, max_val]) 
            if new_image:
                self.region.setRegion([min_val, max_val])

    def _on_region_changed(self, region_item):
        min_val, max_val = region_item.getRegion()
        self.min_val_input.setText(f"{min_val:.2f}")
        self.max_val_input.setText(f"{max_val:.2f}")
        self.region_changed.emit(min_val, max_val)

    def set_region_to_percentile(self, min_percent, max_percent):
        if self.data is not None and self.data.size > 0:
            min_val = np.percentile(self.data, min_percent)
            max_val = np.percentile(self.data, max_percent)
            self.region.setRegion([min_val, max_val])

    def _apply_custom_values(self):
        try:
            min_v = float(self.min_val_input.text())
            max_v = float(self.max_val_input.text())

            if min_v > max_v:
                QMessageBox.warning(self, "Invalid Input", "Min Value must be less than or equal to Max Value.")
                return

            self.region.setRegion([min_v, max_v])
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numbers for values.")


class MathTransformPane(QDockWidget):
    transform_requested = pyqtSignal(str)
    restore_original_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Math Transform", parent)
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        self.content_widget = QWidget()
        self.layout = QVBoxLayout(self.content_widget)

        self.expression_input = QComboBox()
        self.expression_input.setEditable(True)
        self.expression_input.lineEdit().setPlaceholderText("Enter math expression (e.g., x+1, np.log(x))")
        self.expression_input.lineEdit().returnPressed.connect(self._on_apply)
        self.expression_input.lineEdit().setClearButtonEnabled(True)
        # Deferred connection of editTextChanged is handled at end of __init__ or after error_label creation
        
        # Load History
        settings = QSettings()
        history = settings.value("math_transform_history", [])
        # Ensure it's a list (QSettings might return None or type wrapper)
        if not history:
             history = ["x * 2", "x / 2", "255 - x", "np.log(x + 1)", "np.sqrt(x)"]
        else:
             # Ensure types are strings
             history = [str(x) for x in history]

        self.expression_input.addItems(history)
        self.expression_input.setCurrentIndex(-1) # Start blank
        
        self.layout.addWidget(self.expression_input)

        self.apply_button = QPushButton("Apply Transform")
        self.apply_button.clicked.connect(self._on_apply)
        self.layout.addWidget(self.apply_button)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: red;")
        self.layout.addWidget(self.error_label)
        
        # Connect signal after all widgets are initialized
        self.expression_input.editTextChanged.connect(self._on_text_changed)

        self.setWidget(self.content_widget)

    def _on_apply(self):
        expression = self.expression_input.currentText()
        if expression:
            self.error_label.clear()
            
            # Manage History: Move to top or Add
            index = self.expression_input.findText(expression)
            if index != -1:
                self.expression_input.removeItem(index)
            self.expression_input.insertItem(0, expression)
            self.expression_input.setCurrentIndex(0)
            
            # Limit Widget Count to 30 (FIFO eviction from bottom)
            while self.expression_input.count() > 30:
                self.expression_input.removeItem(30)
            
            # Save History
            settings = QSettings()
            history = [self.expression_input.itemText(i) for i in range(self.expression_input.count())]
            settings.setValue("math_transform_history", history)
            
            self.transform_requested.emit(expression)
        else:
            self.error_label.setText("Expression cannot be empty.")

    def _on_text_changed(self, text):
        if not text.strip():
            self.error_label.clear()
            self.restore_original_requested.emit()

    def set_error_message(self, message):
        self.error_label.setText(message)


class InfoPane(QDockWidget):
    apply_clicked = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__("Image Info", parent)
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        self.content_widget = QWidget()
        self.layout = QFormLayout(self.content_widget)

        self.dtype_combo = QComboBox()
        self.width_spinbox = QSpinBox()
        self.width_spinbox.setRange(1, 16384)
        self.height_spinbox = QSpinBox()
        self.height_spinbox.setRange(1, 16384)
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.on_apply)

        self.layout.addRow("Data Type:", self.dtype_combo)
        self.layout.addRow("Width:", self.width_spinbox)
        self.layout.addRow("Height:", self.height_spinbox)
        self.layout.addWidget(self.apply_button)

        self.setWidget(self.content_widget)

    def on_apply(self):
        settings = {
            'width': self.width_spinbox.value(),
            'height': self.height_spinbox.value(),
            'dtype': self.dtype_combo.currentData(),
        }
        self.apply_clicked.emit(settings)

    def update_info(self, width, height, dtype, dtype_map):
        if self.dtype_combo.count() == 0:
            for ext, dt in dtype_map.items():
                self.dtype_combo.addItem(f"{ext} ({np.dtype(dt).name})", dt)

        self.width_spinbox.setValue(width)
        self.height_spinbox.setValue(height)

        index = self.dtype_combo.findData(dtype)
        if index != -1:
            self.dtype_combo.setCurrentIndex(index)

    def set_raw_mode(self, is_raw):
        self.setEnabled(is_raw)


class ZoomSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Zoom Settings")
        self.layout = QFormLayout(self)
        self.zoom_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_speed_slider.setRange(1, 100)
        self.zoom_speed_slider.setValue(10)
        self.layout.addRow("Zoom Speed:", self.zoom_speed_slider)
        self.zoom_in_interp_combo = QComboBox()
        self.zoom_in_interp_combo.addItems(["Smooth", "Nearest"])
        self.layout.addRow("Zoom In Interpolation:", self.zoom_in_interp_combo)
        self.zoom_out_interp_combo = QComboBox()
        self.zoom_out_interp_combo.addItems(["Smooth", "Nearest"])
        self.layout.addRow("Zoom Out Interpolation:", self.zoom_out_interp_combo)
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

    def get_settings(self):
        return {
            "zoom_speed": self.zoom_speed_slider.value() / 100.0 + 1.0,
            "zoom_in_interp": self.zoom_in_interp_combo.currentText(),
            "zoom_out_interp": self.zoom_out_interp_combo.currentText(),
        }

    def set_settings(self, settings):
        self.zoom_speed_slider.setValue(int((settings["zoom_speed"] - 1.0) * 100))
        self.zoom_in_interp_combo.setCurrentText(settings["zoom_in_interp"])
        self.zoom_out_interp_combo.setCurrentText(settings["zoom_out_interp"])


class ZoomableDraggableLabel(QOpenGLWidget): # Inherits QOpenGLWidget for GPU acceleration
    hover_moved = pyqtSignal(int, int) # Signals mouse position in image coordinates
    hover_left = pyqtSignal()
    zoom_factor_changed = pyqtSignal(float)
    view_changed = pyqtSignal()
    clicked = pyqtSignal()

    def __init__(self, parent=None, shared_state=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.grabGesture(Qt.GestureType.PinchGesture)

        self.shared_state = shared_state
        if self.shared_state:
            self.shared_state.view_changed.connect(self._on_shared_view_changed)
            self.shared_state.crosshair_changed.connect(self._on_shared_crosshair_changed)

        self._scale_factor = 1.0
        self._pixmap_offset = QPointF()
        self._crosshair_pos = None
        self._fit_scale = 1.0
        self._proxy_scale = 1.0 # Scale of the display pixmap relative to original data

        self.is_active = False
        self.crosshair_enabled = False
        self.drag_start_position = QPoint()
        self.current_pixmap = None
        self.drag_start_position = QPoint()
        self.processed_data = None # Store processed numpy array instead of QPixmap

        self.overlays = [] # List of (QPixmap, opacity)
        self.pristine_data = None # Original loaded data (preserved across transforms)
        self.thumbnail_pixmap = None # Cached thumbnail for optimized rendering
        self.contrast_limits = None
        self.colormap = 'gray'

        self.zoom_speed = 1.1
        self.zoom_in_interp = Qt.TransformationMode.FastTransformation
        self.zoom_out_interp = Qt.TransformationMode.SmoothTransformation
        self._pinch_start_scale_factor = None

        # Overlay Label
        self.overlay_label = QLabel(self)
        self.overlay_label.setStyleSheet("background-color: rgba(0, 0, 0, 150); color: white; padding: 5px; border-radius: 5px; font-size: 14px;")
        self.overlay_label.hide()
        self.overlay_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
        self.overlays = [] # List of tuples: (QPixmap, opacity)

    def _get_effective_scale_factor(self):
        if self.shared_state:
            return self._fit_scale * self.shared_state.zoom_multiplier
        return self._scale_factor

    def _get_effective_offset(self):
        if self.shared_state:
            return self.shared_state.offset
        return self._pixmap_offset

    def _get_crosshair_norm_pos(self):
        return self.shared_state.crosshair_norm_pos if self.shared_state else self._crosshair_pos

    def _set_crosshair_norm_pos(self, value):
        if self.shared_state:
            self.shared_state.crosshair_norm_pos = value
        else:
            self._crosshair_pos = value
            self.update()  # Force repaint for single view

    def _on_shared_view_changed(self):
        self.update()

    def _on_shared_crosshair_changed(self):
        self.update()

    def set_active(self, active):
        if self.is_active != active:
            self.is_active = active
            self.update()

    def set_crosshair_enabled(self, enabled):
        self.crosshair_enabled = enabled
        self.setMouseTracking(enabled)
        if enabled:
            # Try to set initial position from current mouse location using QCursor
            # This ensures crosshair appears immediately without needing mouse move
            from PyQt6.QtGui import QCursor
            local_pos = self.mapFromGlobal(QCursor.pos())
            if self.rect().contains(local_pos):
                # We need to simulate the calculation done in mouseMoveEvent
                # Reusing the logic would be cleaner, but for now let's duplicate the essential mapping part
                # or trigger a fake mouse event? No, calculation is safer.
                
                if self.original_data is not None and self.current_pixmap is not None:
                     self._update_crosshair_from_pos(local_pos)

        if not enabled:
            self._set_crosshair_norm_pos(None)
            self.unsetCursor()
        
        self.update()

    def _update_crosshair_from_pos(self, local_pos):
        if self.original_data is None or self.current_pixmap is None:
            return

        scale = self._get_effective_scale_factor()
        p_w = self.current_pixmap.width()
        p_h = self.current_pixmap.height()
        
        if self._proxy_scale > 0:
            w = (p_w / self._proxy_scale) * scale
            h = (p_h / self._proxy_scale) * scale
        else:
            w = p_w * scale
            h = p_h * scale

        target_rect = QRectF(0, 0, w, h)
        target_rect.moveCenter(QPointF(self.rect().center()) + self._get_effective_offset())

        if target_rect.contains(QPointF(local_pos)):
            # Calculate coordinate relative to the Display Rect (0 to 1)
            rel_x = (local_pos.x() - target_rect.left()) / target_rect.width()
            rel_y = (local_pos.y() - target_rect.top()) / target_rect.height()
            
            # Map to Original Image Coordinates
            if self.original_data is not None:
                 img_h, img_w = self.original_data.shape[:2]
                 x = int(rel_x * img_w)
                 y = int(rel_y * img_h)
            else:
                x = 0; y = 0 # Fallback

            if 0 <= x < img_w and 0 <= y < img_h:
                if self.crosshair_enabled:
                    # Calculate normalized position
                    norm_x = x / img_w
                    norm_y = y / img_h
                    self._set_crosshair_norm_pos(QPointF(norm_x, norm_y))
                    # Set cursor to crosshair
                    self.setCursor(Qt.CursorShape.CrossCursor)
                else:
                    self.unsetCursor()
                # Emit actual data coordinates for status bar
                self.hover_moved.emit(x, y)
            else:
                self.hover_left.emit()
                self.unsetCursor()
        else:
            self.hover_left.emit()
            self.unsetCursor()

    def set_data(self, data, reset_view=True, is_pristine=False):
        self.original_data = data
        if is_pristine:
            self.pristine_data = data
        self.contrast_limits = None
        self.apply_colormap(is_new_image=reset_view)

    def is_single_channel(self):
        return self.original_data is not None and self.original_data.ndim == 2

    def get_visible_sub_image(self):
        """Returns the sub-image data corresponding to the currently visible area."""
        if self.original_data is None:
            return None

        scale = self._get_effective_scale_factor()
        offset = self._get_effective_offset()
        
        # Visible rect in widget coordinates
        visible_rect = self.rect()
        
        # Map widget coords to image coords
        # Center of the widget + offset is the center of the image
        
        h, w = self.original_data.shape[:2]
        
        center_x = self.width() / 2 + offset.x()
        center_y = self.height() / 2 + offset.y()
        
        # Calculate top-left of the image in widget coords
        img_x = center_x - (w * scale) / 2
        img_y = center_y - (h * scale) / 2
        
        # Calculate visible region relative to image top-left (in screen pixels)
        rel_x1 = visible_rect.left() - img_x
        rel_y1 = visible_rect.top() - img_y
        rel_x2 = visible_rect.right() - img_x
        rel_y2 = visible_rect.bottom() - img_y
        
        # Convert to original image coordinates
        orig_x1 = int(rel_x1 / scale)
        orig_y1 = int(rel_y1 / scale)
        orig_x2 = int(rel_x2 / scale)
        orig_y2 = int(rel_y2 / scale)
        
        h, w = self.original_data.shape[:2]
        
        # Clamp to image bounds
        orig_x1 = max(0, min(orig_x1, w))
        orig_y1 = max(0, min(orig_y1, h))
        orig_x2 = max(0, min(orig_x2, w))
        orig_y2 = max(0, min(orig_y2, h))
        
        if orig_x1 >= orig_x2 or orig_y1 >= orig_y2:
            return None
            
        return self.original_data[orig_y1:orig_y2, orig_x1:orig_x2]

    def set_contrast_limits(self, min_val, max_val):
        self.contrast_limits = (min_val, max_val)
        self.apply_colormap()

    def set_colormap(self, name):
        self.colormap = name
        self.apply_colormap()

    def apply_colormap(self, is_new_image=False):
        if self.original_data is None:
            return

        data = self.original_data

        if len(data.shape) == 3 and data.shape[2] == 3:  # Color Image
            if self.contrast_limits:
                min_val, max_val = self.contrast_limits
                if max_val > min_val:
                    stretched_channels = []
                    for i in range(3):
                        channel = data[..., i].astype(np.float32)
                        stretched = 255 * (channel - min_val) / (max_val - min_val)
                        stretched_channels.append(np.clip(stretched, 0, 255))
                    processed_data = np.stack(stretched_channels, axis=-1).astype(np.uint8)
                else:
                    processed_data = np.zeros_like(data, dtype=np.uint8)
            else:
                processed_data = data.astype(np.uint8)

            h, w, _ = processed_data.shape
            q_image = QImage(processed_data.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)

        else:  # Grayscale Image
            processed_data = data.copy()
            if self.contrast_limits:
                min_val, max_val = self.contrast_limits
                processed_data = np.clip(processed_data, min_val, max_val)

            min_val, max_val = np.min(processed_data), np.max(processed_data)
            if min_val == max_val:
                norm_data = np.zeros_like(processed_data, dtype=float)
            else:
                norm_data = (processed_data - min_val) / (max_val - min_val)

            colored_data = cm.get_cmap(self.colormap)(norm_data)
            image_data_8bit = (colored_data[:, :, :3] * 255).astype(np.uint8)
            processed_data = image_data_8bit # Fix: Update processed_data to hold the 3D RGB array
            h, w, _ = image_data_8bit.shape
            q_image = QImage(image_data_8bit.data, w, h, 3 * w, QImage.Format.Format_RGB888)

        # Proxy Rendering Logic
        # If image is too large, downsample it for display
        MAX_DIM = 2048
        h, w = processed_data.shape[:2]
        
        self._proxy_scale = 1.0
        if max(h, w) > MAX_DIM:
            # simple integer downsampling for speed and sharpness
            import math
            step = max(1, int(math.ceil(max(h, w) / MAX_DIM)))
            if step > 1:
                self._proxy_scale = 1.0 / step
                # Downsample
                if processed_data.ndim == 3:
                     processed_data = processed_data[::step, ::step, :]
                else:
                     processed_data = processed_data[::step, ::step]

        h, w = processed_data.shape[:2]
        # Ensure contiguous
        if not processed_data.flags['C_CONTIGUOUS']:
            processed_data = np.ascontiguousarray(processed_data)
            
        q_image = QImage(processed_data.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(q_image)
        
        if is_new_image:
            self.processed_data = None # We don't need to store full res processed data in RAM for display anymore
            self.current_pixmap = pixmap
            QTimer.singleShot(0, self.fit_to_view)
        else:
            self.update_pixmap_content(pixmap)

    def update_pixmap_content(self, pixmap):
        self.current_pixmap = pixmap
        # Generate Thumbnail for Montage Optimization (if large)
        if pixmap and max(pixmap.width(), pixmap.height()) > 800:
             self.thumbnail_pixmap = pixmap.scaled(800, 800, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        else:
             self.thumbnail_pixmap = None
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_fit_scale()
        self._update_overlay_position()

    def set_overlay_text(self, text):
        self.overlay_label.setText(text)
        self.overlay_label.adjustSize()
        self._update_overlay_position()

    def toggle_overlay(self):
        self.overlay_label.setVisible(not self.overlay_label.isVisible())
        self._update_overlay_position()

    def _update_overlay_position(self):
        if self.overlay_label.isVisible() and self.overlay_label.text():
            self.overlay_label.adjustSize()
            # Position at top-center
            x = (self.width() - self.overlay_label.width()) // 2
            y = 20
            self.overlay_label.move(x, y)
            self.overlay_label.raise_()

    def update_fit_scale(self):
        if self.current_pixmap and self.size().width() > 0 and self.size().height() > 0:
            label_size = self.size()
            
            # Fit calculation needs to account for proxy scaling
            # effective_image_w = pixmap_w / proxy_scale
            # But simpler: fit scale is how much we scale the ORIGINAL to fit window.
            # pixmap is (Original * proxy_scale).
            
            # Let's rely on original dimensions if available, or infer from pixmap/proxy
            if self.original_data is not None:
                h, w = self.original_data.shape[:2]
            else:
                # Infer
                pix_size = self.current_pixmap.size()
                w = pix_size.width() / self._proxy_scale
                h = pix_size.height() / self._proxy_scale
            
            if w == 0 or h == 0: return

            scale_w = label_size.width() / w
            scale_h = label_size.height() / h

            self._fit_scale = min(scale_w, scale_h)

            if self.shared_state:
                if self.shared_state.zoom_multiplier == 1.0: # Only reset if we are not zoomed
                     pass # We let the shared state keep its multiplier relative to fit scale?
                     # Actually, if we resize, the fit scale changes, so the effective scale factor changes.
                     # If we are in "fit to view" mode (z=1), we probably want to stay fit to view.
            else:
                 if self._scale_factor == self._fit_scale:
                     pass # Already at fit scale

            self.update()

    def fit_to_view(self):
        if self.current_pixmap and self.size().width() > 0 and self.size().height() > 0:
            self.update_fit_scale()
            self._pixmap_offset = QPointF()

            if self.shared_state:
                self.shared_state.reset()
            else:
                self._scale_factor = self._fit_scale

            self.update()

        self.zoom_factor_changed.emit(self._get_effective_scale_factor())
        self.view_changed.emit()

    def restore_view(self):
        self.fit_to_view()

    def get_view_state(self):
        """Returns a dictionary containing the current view state (zoom, pan, contrast, colormap)."""
    def get_view_state(self):
        """Returns a dictionary containing the current view state (zoom, pan, contrast, colormap)."""
        # effective_scale is relative to the PIXMAP (which might be a proxy).
        # We need to Normalize it to be relative to the ORIGINAL IMAGE.
        # ScaleProxy = ScaleOriginal * ProxyFactor
        # ScaleOriginal = ScaleProxy / ProxyFactor
        scale_vs_original = self._get_effective_scale_factor()
        if self._proxy_scale > 0:
            scale_vs_original /= self._proxy_scale

        return {
            'colormap': self.colormap,
            'contrast_limits': self.contrast_limits,
            'scale_factor': scale_vs_original,
            'offset': self._get_effective_offset()
        }

    def set_view_state(self, state):
        """Applies a view state dictionary to this label."""
        if not state:
            return

        # Apply Colormap (only if target is single channel/grayscale)
        if 'colormap' in state and self.is_single_channel():
            self.set_colormap(state['colormap'])

        # Apply Contrast Limits
        if 'contrast_limits' in state:
            limits = state['contrast_limits']
            if limits:
                self.set_contrast_limits(*limits)
            else:
                self.contrast_limits = None
                self.apply_colormap()

        # Apply Zoom
        if 'scale_factor' in state:
            desired_scale_vs_original = state['scale_factor']
            
            # Convert back to Scale vs Proxy (which is what effective_scale expects)
            # ScaleProxy = ScaleOriginal * ProxyFactor
            desired_effective_scale = desired_scale_vs_original
            if self._proxy_scale > 0:
                 desired_effective_scale *= self._proxy_scale

            if self.shared_state:
                 # Shared state uses a multiplier relative to fit_scale
                 if self._fit_scale > 0:
                     internal_factor = desired_effective_scale / self._fit_scale
                     # Clamp multiplier
                     internal_factor = max(0.01, min(100.0, internal_factor))
                     self.shared_state.zoom_multiplier = internal_factor
            else:
                 # Single view uses absolute scale factor
                 self._scale_factor = desired_effective_scale
                 self.zoom_factor_changed.emit(self._get_effective_scale_factor())

        # Apply Pan (Offset)
        if 'offset' in state:
            new_offset = state['offset']
            if self.shared_state:
                self.shared_state.offset = new_offset
            else:
                self._pixmap_offset = new_offset

        self.view_changed.emit()
        self.update()



    def _apply_zoom(self, new_effective_scale, mouse_pos=None):
        if self.current_pixmap is None:
            return

        if self.shared_state:
            old_zoom_multiplier = self.shared_state.zoom_multiplier
            new_zoom_multiplier = new_effective_scale / self._fit_scale
            
            # Dynamic Limits for Shared State
            # Max Limit: Defined globally by the image with the largest Max Zoom Requirement
            max_mult = self.shared_state.max_zoom_limit
            # Min Limit: 10% effective scale for the current image
            min_mult = 0.1 / self._fit_scale
            
            new_zoom_multiplier = max(min_mult, min(max_mult, new_zoom_multiplier))

            if abs(new_zoom_multiplier - old_zoom_multiplier) < 1e-9:
                return

            zoom_ratio = new_zoom_multiplier / old_zoom_multiplier
            if mouse_pos:
                mouse_rel_center = mouse_pos - QPointF(self.rect().center())
                new_offset = mouse_rel_center - (mouse_rel_center - self.shared_state.offset) * zoom_ratio
                self.shared_state.offset = new_offset
            else:
                self.shared_state.offset *= zoom_ratio
            self.shared_state.zoom_multiplier = new_zoom_multiplier
        else:
            old_scale_factor = self._scale_factor
            
            # Dynamic Limit for Single View: 1 pixel takes the whole view
            max_limit = max(self.width(), self.height())
            new_scale_factor = max(0.1, min(float(max_limit), new_effective_scale))
            
            if abs(new_scale_factor - old_scale_factor) < 1e-9:
                return
            zoom_ratio = new_scale_factor / old_scale_factor
            if mouse_pos:
                mouse_rel_center = mouse_pos - QPointF(self.rect().center())
                new_offset = mouse_rel_center - (mouse_rel_center - self._pixmap_offset) * zoom_ratio
                self._pixmap_offset = new_offset
            else:
                self._pixmap_offset *= zoom_ratio
            self._scale_factor = new_scale_factor
            self.update()

        self.zoom_factor_changed.emit(self._get_effective_scale_factor())
        self.view_changed.emit()

    def wheelEvent(self, event):
        current_effective_scale = self._get_effective_scale_factor()
        if self.shared_state:
             self.shared_state.begin_interaction()
        
        if event.angleDelta().y() > 0:
            new_effective_scale = current_effective_scale * self.zoom_speed
        else:
            new_effective_scale = current_effective_scale / self.zoom_speed
        self._apply_zoom(new_effective_scale, event.position())

    def event(self, event):
        if event.type() == QEvent.Type.Gesture:
            pinch = event.gesture(Qt.GestureType.PinchGesture)
            if pinch:
                if pinch.state() == Qt.GestureState.GestureStarted:
                    self._pinch_start_scale_factor = self._get_effective_scale_factor()
                elif pinch.state() == Qt.GestureState.GestureUpdated:
                    new_scale_factor = self._pinch_start_scale_factor * pinch.totalScaleFactor()
                    self._apply_zoom(new_scale_factor, pinch.centerPoint())
                elif pinch.state() == Qt.GestureState.GestureFinished:
                    self._pinch_start_scale_factor = None
                return True
        return super().event(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.current_pixmap is not None:
            self.drag_start_position = event.pos()
            self.clicked.emit()
            self.setFocus()
        return super().mousePressEvent(event)

    def enterEvent(self, event):
        if self.shared_state and self.crosshair_enabled:
            # When we enter a widget in shared mode, we want to ensure we track mouse
            # but we don't necessarily need to "click" to activate if we just want to see values.
            # For now, let's just ensure we update the shared crosshair position.
            pass
        super().enterEvent(event)

    def mouseMoveEvent(self, event):
        if not self.drag_start_position.isNull() and self.current_pixmap is not None:
            delta = event.pos() - self.drag_start_position
            if self.shared_state:
                # Use explicit addition/assignment to ensure property setter is triggered correctly
                # avoiding potential in-place modification issues with += on QPointF properties
                self.shared_state.offset = self.shared_state.offset + QPointF(delta)
            else:
                self._pixmap_offset += QPointF(delta)
                self.update()
            self.drag_start_position = event.pos()
            self.view_changed.emit()

        if self.original_data is not None and self.current_pixmap is not None:
             # Even if not active/focused, if we are in shared state we should update.
             # We want "seamless" movement.
             self._update_crosshair_from_pos(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start_position = QPoint()

    def set_overlays(self, overlays):
        """Sets the list of overlays to be drawn on top of the main image.
           overlays: List of (QPixmap, opacity) tuples. 
           Assumes pixmaps are already resized to match the current_pixmap size.
        """
        self.overlays = overlays
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        
        if self.current_pixmap is None:
            return

        painter = QPainter(self)
        
        # Calculate Effective Transform
        offset = self._get_effective_offset()
        scale = self._get_effective_scale_factor()

        painter.save()
        
        # Apply Zoom and Pan (Center-based)
        painter.translate(self.width() / 2 + offset.x(), self.height() / 2 + offset.y())
        
        # Calculate draw_scale accounting for Proxy
        draw_scale = scale
        if self._proxy_scale > 0:
            draw_scale = scale / self._proxy_scale
            
        painter.scale(draw_scale, draw_scale)
        
        # Move back by half the PIXMAP size (center it)
        painter.translate(-self.current_pixmap.width() / 2, -self.current_pixmap.height() / 2)
        
        # Determine if we should use Smooth Transformation
        # Use Fast if interacting (scrolling/dragging) to improve performance, especially in Montage
        use_smooth = True
        if self.shared_state and self.shared_state._is_interacting:
            use_smooth = False
        
        scale_factor_for_hint = self._get_effective_scale_factor()
        
        if scale_factor_for_hint < 1.0 and use_smooth:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        elif scale_factor_for_hint > 1.0:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.zoom_in_interp == Qt.TransformationMode.SmoothTransformation)
        else: # scale == 1.0 or not use_smooth for zoom out
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.zoom_out_interp == Qt.TransformationMode.SmoothTransformation and use_smooth)

        # Draw Image (Thumbnail Optimized)
        drawn = False
        if self.thumbnail_pixmap and scale < 0.5: # Only use thumb if significantly zoomed out
             target_width = self.current_pixmap.width() * scale
             # If target on screen is smaller than thumbnail, use thumbnail
             if target_width < self.thumbnail_pixmap.width():
                 painter.save()
                 # Adjust scale: We are in "Original Image Space".
                 # Thumbnail is smaller. We need to scale UP the drawing context to make the thumbnail fill the Original space.
                 # Ratio > 1.0
                 width_ratio = self.current_pixmap.width() / self.thumbnail_pixmap.width()
                 painter.scale(width_ratio, width_ratio)
                 painter.drawPixmap(0, 0, self.thumbnail_pixmap)
                 painter.restore()
                 drawn = True
        
        if not drawn:
            painter.drawPixmap(0, 0, self.current_pixmap)

        # Draw Overlays (in Original Image Space)
        if self.overlays:
            for overlay_pixmap, opacity in self.overlays:
                if overlay_pixmap and not overlay_pixmap.isNull():
                    painter.setOpacity(opacity)
                    painter.drawPixmap(0, 0, overlay_pixmap)
            painter.setOpacity(1.0)

        painter.restore()
        
        # Crosshair drawing continues below...

        # Calculate target rect for Crosshair (Widget Coordinates)
        w = self.current_pixmap.width() * draw_scale
        h = self.current_pixmap.height() * draw_scale
        target_rect = QRectF(0, 0, w, h)
        target_rect.moveCenter(QPointF(self.rect().center()) + offset)

        if self.crosshair_enabled and self._get_crosshair_norm_pos():
            norm_pos = self._get_crosshair_norm_pos()
            
            img_height, img_width = self.original_data.shape[0], self.original_data.shape[1]
            # Map normalized pos back to this specific image's coordinates
            img_x = int(norm_pos.x() * img_width)
            img_y = int(norm_pos.y() * img_height)

            scale = self._get_effective_scale_factor()
            
            # Calculate view position
            # Note: img_x * scale might be slightly off if we don't consider exact pixel centers, but close enough for cursor
            view_x = (img_x * scale) + target_rect.left()
            view_y = (img_y * scale) + target_rect.top()
            
            if 0 <= img_x < img_width and 0 <= img_y < img_height:

                pen = painter.pen()
                pen.setColor(QColor(255, 255, 255))
                pen.setWidth(2)
                painter.setPen(pen)

                # Draw Crosshair Cursor (Small cross)
                cross_size = 10
                painter.drawLine(int(view_x) - cross_size, int(view_y), int(view_x) + cross_size, int(view_y))
                painter.drawLine(int(view_x), int(view_y) - cross_size, int(view_x), int(view_y) + cross_size)
                
                # Draw outline for better visibility
                pen.setColor(QColor(0, 0, 0))
                pen.setWidth(1)
                painter.setPen(pen)
                painter.drawLine(int(view_x) - cross_size, int(view_y) - 1, int(view_x) + cross_size, int(view_y) - 1) # Top shadow
                painter.drawLine(int(view_x) - cross_size, int(view_y), int(view_x) + cross_size, int(view_y))
                painter.drawLine(int(view_x), int(view_y) - cross_size, int(view_x), int(view_y) + cross_size)

                # Fetch value
                value = self.original_data[img_y, img_x]
                text = f"({img_x}, {img_y}): {value}"

                # Draw Tooltip
                painter.setBrush(QColor(0, 0, 0, 180)) # Semi-transparent black background
                painter.setPen(Qt.GlobalColor.transparent)
                
                text_rect = painter.fontMetrics().boundingRect(text)
                text_padding = 5
                text_w = text_rect.width() + 2 * text_padding
                text_h = text_rect.height() + 2 * text_padding
                
                tooltip_x = int(view_x) + 15
                tooltip_y = int(view_y) + 15
                
                # Adjust if out of bounds (simple check)
                if tooltip_x + text_w > self.width():
                    tooltip_x = int(view_x) - text_w - 5
                if tooltip_y + text_h > self.height():
                    tooltip_y = int(view_y) - text_h - 5
                
                painter.drawRoundedRect(tooltip_x, tooltip_y, text_w, text_h, 5, 5)
                
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(tooltip_x + text_padding, tooltip_y + text_padding + text_rect.height() - 2, text)


class ThumbnailItem(QWidget):
    clicked = pyqtSignal(QEvent)
    overlay_changed = pyqtSignal(float)

    def __init__(self, file_path, pixmap, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.is_selected = False
        self.is_focused = False

        self.setFixedSize(160, 180) # Increased height for slider
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5) # Reduce margins
        self.image_label = QLabel()
        self.image_label.setPixmap(
            pixmap.scaled(128, 128, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.filename_label = QLabel(os.path.basename(file_path))
        self.filename_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.filename_label.setWordWrap(True)

        layout.addWidget(self.image_label)
        layout.addWidget(self.filename_label)

        # Overlay Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(0)
        self.slider.setToolTip("Overlay Opacity")
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider)

        self.setLayout(layout)

    def _on_slider_changed(self, value):
        self.overlay_changed.emit(value / 100.0)

    def mousePressEvent(self, event: QMouseEvent):
        self.clicked.emit(event)
        super().mousePressEvent(event)

    def set_selected(self, selected):
        if self.is_selected != selected:
            self.is_selected = selected
            self.update()

    def set_focused(self, focused):
        if self.is_focused != focused:
            self.is_focused = focused
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        if self.is_selected:
            pen_color = Qt.GlobalColor.blue
            pen_width = 3
        elif self.is_focused:
            pen_color = Qt.GlobalColor.darkGray
            pen_width = 2
        else:
            return

        pen = painter.pen()
        pen.setColor(pen_color)
        pen.setWidth(pen_width)
        painter.setPen(pen)
        painter.drawRect(self.rect().adjusted(1, 1, -1, -1))


class ThumbnailPane(QDockWidget):
    selection_changed = pyqtSignal(list)
    overlay_changed = pyqtSignal(str, float)

    def __init__(self, parent=None):
        super().__init__("Opened Images", parent)
        self.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.setFloating(False)
        self.setMinimumWidth(240) # Ensure thumbnails (160px) + sliders fit with scrollbar

        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.thumbnail_layout = QVBoxLayout(self.scroll_content)
        self.thumbnail_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setWidget(self.scroll_content)

        self.scroll_area.installEventFilter(self)

        self.main_layout.addWidget(self.scroll_area)
        self.setWidget(self.main_widget)

        self.thumbnail_items = []
        self.focused_index = -1
        self.last_selection_anchor = -1

    def eventFilter(self, source, event):
        if source == self.scroll_area and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key in [Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Space]:
                self.keyPressEvent(event)
                return True
        return super().eventFilter(source, event)

    def keyPressEvent(self, event: QKeyEvent):
        if not self.thumbnail_items:
            return

        key = event.key()
        modifiers = event.modifiers()

        current_index = self.focused_index if self.focused_index != -1 else 0

        new_index = current_index
        if key == Qt.Key.Key_Down:
            new_index = min(current_index + 1, len(self.thumbnail_items) - 1)
        elif key == Qt.Key.Key_Up:
            new_index = max(current_index - 1, 0)

        if new_index != current_index:
            self._handle_selection(new_index, modifiers)
        elif key == Qt.Key.Key_Space:
            self._handle_selection(current_index, modifiers | Qt.KeyboardModifier.ControlModifier)

    def populate(self, windows):
        for item in self.thumbnail_items:
            self.thumbnail_layout.removeWidget(item)
            item.deleteLater()
        self.thumbnail_items.clear()

        for window in windows:
            if window.image_label.current_pixmap and window.current_file_path:
                pixmap = window.image_label.current_pixmap
                file_path = window.current_file_path
                item = ThumbnailItem(file_path, pixmap)
                item.clicked.connect(lambda event, i=item: self._on_thumbnail_clicked(i, event))
                item.overlay_changed.connect(lambda alpha, path=file_path: self.overlay_changed.emit(path, alpha))
                self.thumbnail_items.append(item)
                self.thumbnail_layout.addWidget(item)

        if self.thumbnail_items:
            self._set_focused_item(0)

    def _on_thumbnail_clicked(self, item, event):
        index = self.thumbnail_items.index(item)
        self._handle_selection(index, event.modifiers())

    def _handle_selection(self, index, modifiers):
        self._set_focused_item(index)

        is_shift = modifiers & Qt.KeyboardModifier.ShiftModifier
        is_ctrl = modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)

        if is_shift:
            anchor = self.last_selection_anchor if self.last_selection_anchor != -1 else index
            self._select_range(anchor, index)
        elif is_ctrl:
            self._toggle_selection(index)
            self.last_selection_anchor = index
        else:
            self._select_single(index)
            self.last_selection_anchor = index

        self._emit_selection_change()

    def _toggle_selection(self, index):
        self.thumbnail_items[index].set_selected(not self.thumbnail_items[index].is_selected)

    def _select_single(self, index):
        for i, item in enumerate(self.thumbnail_items):
            item.set_selected(i == index)

    def _select_range(self, start, end):
        start_idx, end_idx = min(start, end), max(start, end)
        for i, item in enumerate(self.thumbnail_items):
            item.set_selected(start_idx <= i <= end_idx)

    def _emit_selection_change(self):
        selected_files = [item.file_path for item in self.thumbnail_items if item.is_selected]
        self.selection_changed.emit(selected_files)

    def _set_focused_item(self, index):
        if self.focused_index != -1:
            self.thumbnail_items[self.focused_index].set_focused(False)

        self.focused_index = index

        if self.focused_index != -1:
            self.thumbnail_items[self.focused_index].set_focused(True)
            self.scroll_area.ensureWidgetVisible(self.thumbnail_items[self.focused_index])