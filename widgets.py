import time

import numpy as np
import re
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QPointF, QEvent, QObject, QTimer, QRectF, QSettings, QSortFilterProxyModel, QDir, QRegularExpression, QItemSelectionModel, QSize
from PyQt6.QtGui import QPixmap, QPainter, QNativeGestureEvent, QDoubleValidator, QKeyEvent, QImage, QMouseEvent, QColor, QIcon, QFileSystemModel, QRadialGradient, QLinearGradient, QPen, QBrush
from PyQt6.QtGui import QMatrix4x4
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
    QCheckBox,
    QStyle,
    QTreeView,
    QHeaderView,
    QAbstractItemView,
    QToolButton,
    QStackedLayout,
    QFrame,
    QFileDialog,
    QInputDialog,
    QProgressDialog
)
import sys
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
import pyqtgraph as pg
try:
    import pyqtgraph.opengl as gl
except ImportError:
    gl = None
import os
import matplotlib.cm as cm
from utils import flow_to_color
from settings import load_folder_history, save_folder_history, load_filter_history, save_filter_history


class SharedViewState(QObject):
    """An object to hold and synchronize view parameters for multiple labels."""
    view_changed = pyqtSignal()
    zoom_changed = pyqtSignal(float)
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
            self.zoom_changed.emit(self._zoom_multiplier)

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
        
        # Debounce timer for histogram region updates
        self._region_update_timer = QTimer()
        self._region_update_timer.setSingleShot(True)
        self._region_update_timer.setInterval(200) # 200 ms
        self._region_update_timer.timeout.connect(self._emit_delayed_region_changed)
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

        # Ensure we work with float64 for precision/finite checks
        working_data = data.astype(np.float64)
        
        # We want self.data to be a 1D array of finite values for percentile calculations
        # But we also need to draw histograms for channels if RGB.
        
        if working_data.ndim == 3 and working_data.shape[2] == 3:  # RGB
            colors = [(255, 0, 0, 150), (0, 255, 0, 150), (0, 0, 255, 150)]
            # Calculate luminance for the overall "self.data" used in contrast/percentiles
            lum = np.dot(working_data[..., :3], [0.2989, 0.5870, 0.1140])
            self.data = lum[np.isfinite(lum)] # Only keep finite for percentiles
            
            for i in range(3):
                channel_data = working_data[..., i].flatten()
                channel_finite = channel_data[np.isfinite(channel_data)]
                if channel_finite.size > 0:
                    y, x = np.histogram(channel_finite, bins=256)
                    hist = self.plot_widget.plot(x, y, stepMode="center", fillLevel=0, brush=colors[i])
                    self.histograms.append(hist)
        else:  # Grayscale
            flat_data = working_data.flatten()
            self.data = flat_data[np.isfinite(flat_data)]
            if self.data.size > 0:
                y, x = np.histogram(self.data, bins=256)
                hist = self.plot_widget.plot(x, y, stepMode="center", fillLevel=0, brush=(200, 200, 200, 150))
                self.histograms.append(hist)

        if self.data is None or self.data.size == 0:
            return

        # Double check self.data is finite (should be from logic above, but be defensive)
        finite_data = self.data[np.isfinite(self.data)]
        if finite_data.size == 0:
            return

        with np.errstate(divide='ignore', invalid='ignore'):
            min_val = np.nanmin(finite_data)
            max_val = np.nanmax(finite_data)
            
        if np.isfinite(min_val) and np.isfinite(max_val) and min_val < max_val:
            self.plot_widget.setXRange(min_val, max_val, padding=0.05)
            if new_image:
                self.region.setRegion([min_val, max_val])

    def _on_region_changed(self, region_item):
        min_val, max_val = region_item.getRegion()
        self.min_val_input.setText(f"{min_val:.2f}")
        self.max_val_input.setText(f"{max_val:.2f}")
        # Debounce the expensive image update
        self._region_update_timer.start()

    def _emit_delayed_region_changed(self):
        min_val, max_val = self.region.getRegion()
        self.region_changed.emit(min_val, max_val)

    def set_region_to_percentile(self, min_percent, max_percent):
        if self.data is not None and self.data.size > 0:
            # self.data is filtered for finite values in update_histogram,
            # but we filter again to be 100% sure we don't trigger internal warnings.
            finite_data = self.data[np.isfinite(self.data)]
            if finite_data.size == 0:
                return

            with np.errstate(divide='ignore', invalid='ignore'):
                min_val = np.nanpercentile(finite_data, min_percent)
                max_val = np.nanpercentile(finite_data, max_percent)
                
            if np.isfinite(min_val) and np.isfinite(max_val):
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
        super().__init__("Math", parent)
        self.setMinimumWidth(200)
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        self.content_widget = QWidget()
        self.content_widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.layout = QVBoxLayout(self.content_widget)

        self.expression_input = QComboBox()
        self.expression_input.setEditable(True)
        self.expression_input.lineEdit().setPlaceholderText("Enter math expression (e.g., x+1, np.log(x))")
        self.expression_input.setMinimumWidth(50)
        self.expression_input.lineEdit().returnPressed.connect(self._on_apply)
        self.expression_input.textActivated.connect(self._on_apply)
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


        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: red;")
        self.layout.addWidget(self.error_label)
        
        # Connect signal after all widgets are initialized
        self.expression_input.editTextChanged.connect(self._on_text_changed)

        self.setWidget(self.content_widget)

    def _on_apply(self, _=None):
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
    # Signal now emits on change
    settings_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__("Info", parent)
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.setMinimumWidth(200)

        self.content_widget = QWidget()
        self.content_widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.layout = QFormLayout(self.content_widget)

        self.file_size = 0 # Bytes

        self.dtype_combo = QComboBox()
        self.dtype_combo.setEditable(True)  
        self.dtype_combo.setMinimumWidth(50)
        self.dtype_combo.currentTextChanged.connect(self._on_parameter_change)

        self.width_spinbox = QSpinBox()
        self.width_spinbox.setRange(1, 100000)
        self.width_spinbox.setValue(1280) # Default to reasonable value
        self.width_spinbox.valueChanged.connect(self._on_width_changed)
        
        self.height_spinbox = QSpinBox()
        self.height_spinbox.setRange(1, 100000)
        self.height_spinbox.setValue(720) # Default to reasonable value
        self.height_spinbox.valueChanged.connect(self._on_parameter_change)

        self.layout.addRow("Data Type:", self.dtype_combo)
        self.layout.addRow("Width:", self.width_spinbox)
        self.layout.addRow("Height:", self.height_spinbox)

        self.color_format_combo = QComboBox()
        self.color_format_combo.addItems(["Grayscale", "RGB", "RGBA", "Bayer GRBG", "Bayer RGGB", "Bayer BGGR", "Bayer GBRG", 
                                          "YUV NV12", "YUV NV21", "YUV YUYV", "YUV UYVY", "YUV I420"])
        self.color_format_combo.currentTextChanged.connect(self._on_parameter_change)
        self.layout.addRow("Format:", self.color_format_combo)

        self.reset_button = QPushButton()
        self.reset_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.reset_button.setToolTip("Reset Parameters")
        self.reset_button.setFixedWidth(30) # Small square-ish button
        self.reset_button.clicked.connect(self._on_reset)
        
        # Add to a row with spacers or specific alignment? 
        # Actually, maybe just a row with label "Reset:" or empty label?
        # Or put it next to something?
        # Existing code: self.layout.addWidget(self.reset_button) which puts it in a new row spanning width.
        # Let's align it right or make it less intrusive.
        
        reset_layout = QHBoxLayout()
        reset_layout.addStretch()
        reset_layout.addWidget(self.reset_button)
        self.layout.addRow("", reset_layout) # improved layout
        
        self.defaults = {}

        self.setWidget(self.content_widget)

    def _on_width_changed(self):
        self._recalculate_height()
        self._on_parameter_change()

    def _on_parameter_change(self):
        # Trigger recalculation if needed (e.g. if type/format changes, height might need update?)
        # User request: "when swutching the image type/format, try finding the closet appropriate resolution."
        # This implies changing type/format ALSO affects resolution if we want to respect file size.
        sender = self.sender()
        if sender == self.dtype_combo or sender == self.color_format_combo:
             self._recalculate_height()
        
        self._emit_settings()

    def _recalculate_height(self):
        if self.file_size <= 0: return

        width = self.width_spinbox.value()
        if width <= 0: return
        
        # Determine BPP
        dtype_str = self.dtype_combo.currentText()
        fmt = self.color_format_combo.currentText()
        
        # Parse bits from dtype string
        itemsize = 1
        if "float64" in dtype_str: itemsize = 8
        elif "float32" in dtype_str: itemsize = 4
        elif "float16" in dtype_str: itemsize = 2
        else:
             match = re.search(r'\d+', dtype_str)
             bits = int(match.group(0)) if match else 8
             # Container size logic (should match image_handler)
             if bits <= 8: itemsize = 1
             elif bits <= 16: itemsize = 2
             elif bits <= 32: itemsize = 4
             else: itemsize = 8
             
        bpp = float(itemsize)
        
        # Format Factors
        if "NV12" in fmt or "NV21" in fmt or "I420" in fmt:
            bpp *= 1.5
        elif "YUYV" in fmt or "UYVY" in fmt:
            bpp *= 2.0
        
        # Calculate Height
        # Size = W * H * BPP
        # H = Size / (W * BPP)
        
        if bpp > 0:
            new_height = int(self.file_size / (width * bpp))
            if new_height < 1: new_height = 1
            
            self.height_spinbox.blockSignals(True)
            self.height_spinbox.setValue(new_height)
            self.height_spinbox.blockSignals(False)

    def _emit_settings(self):
        settings = {
            'width': self.width_spinbox.value(),
            'height': self.height_spinbox.value(),
            'dtype': self.dtype_combo.currentText(),
            'color_format': self.color_format_combo.currentText(),
        }
        self.settings_changed.emit(settings) # Renamed signal

    def _on_reset(self):
        if not self.defaults: return
        
        self.blockSignals(True)
        self.width_spinbox.setValue(self.defaults.get('width', 1920))
        self.height_spinbox.setValue(self.defaults.get('height', 1080))
        self.dtype_combo.setCurrentText(self.defaults.get('dtype', 'uint8'))
        self.color_format_combo.setCurrentText(self.defaults.get('color_format', 'Grayscale'))
        self.blockSignals(False)
        self._emit_settings()

    def update_info(self, width, height, dtype, dtype_map, file_size=0, color_format="Grayscale"):
        self.file_size = file_size
        
        # Store defaults
        self.defaults = {
            'width': width,
            'height': height,
            'dtype': str(dtype) if not isinstance(dtype, type) else np.dtype(dtype).name,
            'color_format': color_format
        }

        self.width_spinbox.blockSignals(True)
        self.height_spinbox.blockSignals(True)
        self.dtype_combo.blockSignals(True)
        self.color_format_combo.blockSignals(True)
        self.width_spinbox.setValue(width)
        self.height_spinbox.setValue(height)
        # Populate Standard Types + float64
        if self.dtype_combo.count() == 0: # Only populate once or check better
             # Actually we might want to ensure items exist
             pass

        if self.dtype_combo.findText("float64") == -1:
             self.dtype_combo.addItem("float64")
        
        # Add basic uint/int types if missing
        # Standard + Common Raw Bit Depths
        common_bits = [8, 10, 12, 14, 16, 32, 64]
        for i in common_bits:
             if self.dtype_combo.findText(f"uint{i}") == -1: self.dtype_combo.addItem(f"uint{i}")
             if self.dtype_combo.findText(f"int{i}") == -1: self.dtype_combo.addItem(f"int{i}")

        index = self.dtype_combo.findText(str(dtype) if not isinstance(dtype, type) else np.dtype(dtype).name)
        # If not found, set text
        if index != -1:
            self.dtype_combo.setCurrentIndex(index)
        else:
             self.dtype_combo.setCurrentText(str(dtype) if not isinstance(dtype, type) else np.dtype(dtype).name)
             
        # Update Color Format Combo
        format_idx = self.color_format_combo.findText(color_format)
        if format_idx != -1:
            self.color_format_combo.setCurrentIndex(format_idx)
        else:
            self.color_format_combo.addItem(color_format)
            self.color_format_combo.setCurrentText(color_format)
             
        self.width_spinbox.blockSignals(False)
        self.height_spinbox.blockSignals(False)
        self.dtype_combo.blockSignals(False)
        self.color_format_combo.blockSignals(False)


    def set_raw_mode(self, is_raw):
        # Allow viewing info even if not raw (read-only mode for parameters)
        self.setEnabled(True) 
        
        # Enable/Disable editing widgets based on is_raw
        self.dtype_combo.setEnabled(is_raw)
        self.width_spinbox.setEnabled(is_raw)
        self.height_spinbox.setEnabled(is_raw)
        self.color_format_combo.setEnabled(is_raw)
        self.reset_button.setEnabled(is_raw)


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
    open_companion_depth = pyqtSignal(str) # Signals to open the associated DEPTAGH map

    def __init__(self, parent=None, shared_state=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents)
        self.grabGesture(Qt.GestureType.PinchGesture) # Re-enable for standard handling
        self.setAcceptDrops(True) # Enable drag and drop

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
        self.processed_data = None # Store processed numpy array instead of QPixmap
        self.overlays = [] # List of (QPixmap, opacity)
        self.original_data = None # Raw data before processing
        self.pristine_data = None # Original loaded data (preserved across transforms)
        self.thumbnail_pixmap = None # Cached thumbnail for optimized rendering
        self.contrast_limits = None
        self.colormap = 'gray'
        self.file_path = None

        self.zoom_speed = 1.1
        self.zoom_in_interp = Qt.TransformationMode.FastTransformation
        self.zoom_out_interp = Qt.TransformationMode.SmoothTransformation
        self._pinch_start_scale_factor = None

        # Active Indicator Line
        self.indicator_line = QFrame(self)
        self.indicator_line.setFixedHeight(3)
        self.indicator_line.setStyleSheet("background-color: transparent;")

        # Depth Map Indicator Button
        self.depth_btn = QPushButton("🗺️ 3D", self)
        self.depth_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(50, 50, 50, 200);
                color: white;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: rgba(80, 80, 80, 255);
            }
        """)
        self.depth_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.depth_btn.setToolTip("Click to open associated Depth Map")
        self.depth_btn.hide()

        # Overlay Label
        self.overlay_label = QLabel(self)
        self.overlay_label.setStyleSheet("background-color: rgba(0, 0, 0, 150); color: white; padding: 5px; border-radius: 5px; font-size: 14px;")
        self.overlay_label.hide()
        self.overlay_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
        
        self.show_colorbar = False # Toggle via external button

        # Debounce Timer for View Changed Signal (Histogram updates)
        self._view_update_timer = QTimer()
        self._view_update_timer.setSingleShot(True)
        self._view_update_timer.setInterval(200) # 200ms debounce
        self._view_update_timer.timeout.connect(self.view_changed.emit)

    @property
    def zoom_scale(self):
        return self._get_effective_scale_factor()

    @zoom_scale.setter
    def zoom_scale(self, value):
        if self.shared_state:
             # Prevent division by zero
             if self._fit_scale > 1e-6:
                 internal_factor = value / self._fit_scale
                 # Clamp multiplier
                 internal_factor = max(0.01, min(100.0, internal_factor))
                 self.shared_state.zoom_multiplier = internal_factor
        else:
             self._scale_factor = value
             self.zoom_factor_changed.emit(value)
        self.view_changed.emit()

    @property
    def pan_pos(self):
        return self._get_effective_offset()
        
    @pan_pos.setter
    def pan_pos(self, value):
        if self.shared_state:
            self.shared_state.offset = value
        else:
            self._pixmap_offset = value
        self.view_changed.emit()

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
        self._update_overlay_position()

    def _on_shared_crosshair_changed(self):
        self.update()

    def set_active(self, active):
        if self.is_active != active:
            self.is_active = active
            self.update()
            
    def update_transform(self):
        """Force a repaint/update of the transformation."""
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

    def set_data(self, data, reset_view=True, is_pristine=False, inspection_data=None):
        self.original_data = data
        # inspection_data is the raw data used for value introspection (e.g. status bar values)
        # If not provided, it defaults to the display data (data)
        self.inspection_data = inspection_data if inspection_data is not None else data

        if is_pristine:
            self.pristine_data = data
            self.base_inspection_data = self.inspection_data # Store base inspection data too
            
        if reset_view:
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
        data = self.original_data
        if data is None: return
        processed_data = None
        q_image = None
        
        if self.colormap == "flow":
            # If data is 2-channel, convert to RGB using flow_to_color
            if data.ndim == 3 and data.shape[2] == 2:
                # Use contrast max as max_flow for normalization (stretching)
                max_flow = None
                if self.contrast_limits:
                     try:
                         # Ensure valid max_flow
                         val = float(self.contrast_limits[1])
                         if val > 1e-6:
                             max_flow = val
                     except:
                         pass
                
                try:
                    # Make sure data is contiguous to avoid potential C-API issues with views
                    if not data.flags['C_CONTIGUOUS']:
                        data = np.ascontiguousarray(data)
                        
                    processed_data = flow_to_color(data, max_flow=max_flow)
                    h, w, _ = processed_data.shape
                    q_image = QImage(processed_data.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
                except Exception as e:
                    print(f"Error in flow_to_color: {e}")
                    return
            
        else:
            # Original logic for other colormaps
            # Determine mode: standard RGB or Colormapped (single channel)
            # If RGB and map is 'gray', show as RGB.
            # If RGB and map is NOT 'gray', extract Ch0 and map it.
            is_rgb = (data.ndim == 3 and data.shape[2] in [3, 4])
            treat_as_rgb = is_rgb and (self.colormap == 'gray')

            if treat_as_rgb:  # Color Image
                channels = data.shape[2]
                if self.contrast_limits:
                    # Sanitize contrast limits: if not finite, fallback to defaults
                    raw_min, raw_max = self.contrast_limits
                    min_val = raw_min if np.isfinite(raw_min) else 0.0
                    max_val = raw_max if np.isfinite(raw_max) else 255.0

                    if max_val > min_val:
                        stretched_channels = []
                        # Process RGB channels with contrast
                        with np.errstate(divide='ignore', invalid='ignore'):
                            for i in range(3):
                                channel = data[..., i].astype(np.float32)
                                # Handle non-finite values (inf/nan) by clipping them to the limits before math
                                safe_channel = np.clip(channel, min_val, max_val)
                                # Also handle any remaining NaNs (if min/max were bad)
                                safe_channel = np.nan_to_num(safe_channel, nan=min_val)
                                
                                stretched = 255 * (safe_channel - min_val) / (max_val - min_val)
                                stretched_channels.append(np.clip(stretched, 0, 255))
                        
                        # Process Alpha channel (if present)
                        if channels == 4:
                            alpha = data[..., 3]
                            if alpha.dtype.kind == 'f':
                                if alpha.max() <= 1.0:
                                     alpha_u8 = (alpha * 255).astype(np.uint8)
                                else:
                                     alpha_u8 = np.clip(alpha, 0, 255).astype(np.uint8)
                            else:
                                alpha_u8 = np.clip(alpha, 0, 255).astype(np.uint8)
                            
                            # Manual Blending (Simple Arithmetic)
                            # Flatten alpha to 0.0-1.0
                            alpha_f = alpha_u8.astype(np.float32) / 255.0
                            
                            # Background color (e.g. dark gray for visibility)
                            bg = 40.0 
                            
                            # Blend RGB channels
                            # Result = RGB * Alpha + BG * (1 - Alpha)
                            blended_channels = []
                            for i in range(3):
                                comp = stretched_channels[i].astype(np.float32)
                                blended = (comp * alpha_f) + (bg * (1.0 - alpha_f))
                                blended_channels.append(np.clip(blended, 0, 255).astype(np.uint8))
                                
                            processed_data = np.stack(blended_channels, axis=-1)
                        else:
                            processed_data = np.stack(stretched_channels, axis=-1).astype(np.uint8)

                    else:
                        processed_data = np.zeros_like(data, dtype=np.uint8)
                else:
                    processed_data = data.astype(np.uint8)
                    # If float 0-1, this becomes 0 or 1.
                    # We need checks for float -> uint8 conversion if limits not set?
                    # Original code: `processed_data = data.astype(np.uint8)`
                    # If data is float32 0-1, this truncates to 0/1. BAD.
                    # BUT self.contrast_limits usually defaults to min/max of data?
                    # If not, existing code was broken for floats without limits.
                    # I will assume limits are set or user accepts it.
                    # BUT specifically for floating point 0-1...
                    if data.dtype.kind == 'f' and data.max() <= 1.0:
                         processed_data = (data * 255).astype(np.uint8)

                h, w, c_out = processed_data.shape
                
                # Ensure contiguous
                if not processed_data.flags['C_CONTIGUOUS']:
                    processed_data = np.ascontiguousarray(processed_data)

                # Use RGBA8888 as native mapping for R,G,B,A data
                fmt = QImage.Format.Format_RGB888 if c_out == 3 else QImage.Format.Format_RGBA8888

                stride = processed_data.strides[0]
                
                # Keep reference to data to prevent garbage collection!
                self._qimage_bytes = processed_data.tobytes()
                
                q_image = QImage(self._qimage_bytes, w, h, stride, fmt)

            else:  # Grayscale / Colormapped
                if is_rgb:
                     data = data[:, :, 0] # Use Channel 0 for colormapping
                elif data.ndim == 3 and data.shape[2] == 2:
                     # 2-Channel non-flow -> Use Magnitude
                     data = np.linalg.norm(data, axis=2)

                processed_data = data.copy()
                
                # Check for unexpected multi-channel data (e.g. 16 channels from raw misread)
                if processed_data.ndim == 3 and processed_data.shape[2] not in [3, 4]:
                     # Collapse to mean to allow visualization instead of crashing
                     processed_data = np.mean(processed_data, axis=2)



                with np.errstate(divide='ignore', invalid='ignore'):
                    if self.contrast_limits:
                        # Sanitize contrast limits: if not finite, fallback to defaults
                        raw_min, raw_max = self.contrast_limits
                        min_val = raw_min if np.isfinite(raw_min) else 0.0
                        max_val = raw_max if np.isfinite(raw_max) else 255.0

                        # Use a copy to avoid modifying original_data
                        processed_data = np.clip(processed_data, min_val, max_val)
                        # Replace NaNs with min_val
                        processed_data = np.nan_to_num(processed_data, nan=min_val)

                    # Use nanmin/nanmax to be ultra-safe with any remaining non-finite values
                    with np.errstate(divide='ignore', invalid='ignore'):
                        # COMPLETELY sanitize processed_data for normalization
                        # Replace non-finite with 0 to prevent internal subtract warnings
                        safe_proc = np.nan_to_num(processed_data, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        if safe_proc.size > 0:
                            curr_min = np.nanmin(safe_proc)
                            curr_max = np.nanmax(safe_proc)
                        else:
                            curr_min = curr_max = 0.0
                    
                    if curr_min == curr_max or not np.isfinite(curr_min) or not np.isfinite(curr_max):
                        norm_data = np.zeros_like(processed_data, dtype=float)
                    else:
                        with np.errstate(divide='ignore', invalid='ignore'):
                            norm_data = (safe_proc - curr_min) / (curr_max - curr_min)
                        # Final safety clamp
                        norm_data = np.nan_to_num(norm_data, nan=0.0, posinf=1.0, neginf=0.0)

                    colored_data = cm.get_cmap(self.colormap)(norm_data)
                image_data_8bit = (colored_data[:, :, :3] * 255).astype(np.uint8)
                processed_data = image_data_8bit # Fix: Update processed_data to hold the 3D RGB array
                h, w, _ = image_data_8bit.shape
                q_image = QImage(image_data_8bit.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            # End of else block for non-flow colormaps

        # Proxy Rendering Logic
        # If image is too large, downsample it for display
        MAX_DIM = 2048
        if processed_data is None:
            return

        h, w = processed_data.shape[:2]
        c_out = processed_data.shape[2] if processed_data.ndim == 3 else 1
        
        self._proxy_scale = 1.0
        if max(h, w) > MAX_DIM:
            import math
            step = max(1, int(math.ceil(max(h, w) / MAX_DIM)))
            if step > 1:
                self._proxy_scale = 1.0 / step
                if processed_data.ndim == 3:
                     processed_data = processed_data[::step, ::step, :]
                else:
                     processed_data = processed_data[::step, ::step]

        h, w = processed_data.shape[:2]
        if not processed_data.flags['C_CONTIGUOUS']:
            processed_data = np.ascontiguousarray(processed_data)
            
        # Select correct QImage Format and Stride
        if processed_data.ndim == 3:
            if c_out == 4:
                fmt = QImage.Format.Format_RGBA8888
                stride = w * 4
            else:
                fmt = QImage.Format.Format_RGB888
                stride = w * 3
        else:
            fmt = QImage.Format.Format_Grayscale8
            stride = w

        # Keep reference to bytes to prevent crash
        self._last_processed_bytes = processed_data.tobytes()
        q_image = QImage(self._last_processed_bytes, w, h, stride, fmt).copy()
        pixmap = QPixmap.fromImage(q_image)
        
        if is_new_image:
            self.processed_data = None 
            self.update_pixmap_content(pixmap)
            self.fit_to_view()
        else:
            self.update_pixmap_content(pixmap)

    def update_pixmap_content(self, pixmap):
        self.current_pixmap = pixmap
        # Generate Thumbnail for Montage Optimization (if large)
        if pixmap and max(pixmap.width(), pixmap.height()) > 800:
             # Use FastTransformation for small images to keep them sharp, Smooth for large ones
            transform_mode = Qt.TransformationMode.SmoothTransformation
            if pixmap.width() < 800 or pixmap.height() < 800:
                transform_mode = Qt.TransformationMode.FastTransformation
            self.thumbnail_pixmap = pixmap.scaled(800, 800, Qt.AspectRatioMode.KeepAspectRatio, transform_mode)
        else:
             self.thumbnail_pixmap = None
             
        # Check for Depth Companion Map
        if hasattr(self, 'file_path') and self.file_path:
             base, _ = os.path.splitext(self.file_path)
             depth_path = f"{base}_DEPTH.tiff"
             depth_dir = f"{base}_DEPTH"
             
             has_depth = os.path.exists(depth_path) or os.path.isdir(depth_dir)
             
             if has_depth:
                 # Disconnect first to avoid multiple connections if updated multiple times
                 try:
                     self.depth_btn.clicked.disconnect()
                 except Exception:
                     pass
                     
                 # If it's a directory, we pass the directory path
                 target_path = depth_dir if os.path.isdir(depth_dir) else depth_path
                 self.depth_btn.clicked.connect(lambda checked, p=target_path: self.open_companion_depth.emit(p))
                 self.depth_btn.show()
             else:
                 self.depth_btn.hide()
                 
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_fit_scale()
        self._update_overlay_position()
        if hasattr(self, 'indicator_line'):
             self.indicator_line.setGeometry(0, self.height() - 3, self.width(), 3)
        if hasattr(self, 'depth_btn'):
             self.depth_btn.setGeometry(10, 10, self.depth_btn.sizeHint().width(), self.depth_btn.sizeHint().height())

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
            
            w_label = self.overlay_label.width()
            h_label = self.overlay_label.height()
            w_widget = self.width()
            h_widget = self.height()
            
            # Default to center of widget if no image
            target_x = (w_widget - w_label) // 2
            target_y = 20
            
            if self.current_pixmap:
                offset = self._get_effective_offset()
                scale = self._get_effective_scale_factor()
                
                # Image Center in Widget Coordinates
                img_cx = w_widget / 2 + offset.x()
                img_cy = h_widget / 2 + offset.y()
                
                # Visual dimensions
                draw_scale = scale
                if self._proxy_scale > 0:
                     draw_scale = scale / self._proxy_scale
                
                vis_h = self.current_pixmap.height() * draw_scale
                
                # Image Top Y
                img_top_y = img_cy - vis_h / 2
                
                # Target X: Center on Image
                target_x = int(img_cx - w_label / 2)
                
                # Target Y: Top of Image + padding
                target_y = int(img_top_y + 10)

            self.overlay_label.move(target_x, target_y)
            self.overlay_label.raise_()

    def get_original_size(self):
        if self.current_pixmap:
            if self.original_data is not None:
                h, w = self.original_data.shape[:2]
            else:
                # Infer from pixmap and proxy
                pix_size = self.current_pixmap.size()
                w = pix_size.width() / self._proxy_scale
                h = pix_size.height() / self._proxy_scale
            return QSize(int(w), int(h))
        return None

    def update_fit_scale(self):
        if self.current_pixmap and self.size().width() > 0 and self.size().height() > 0:
            label_size = self.size()
            
            orig_size = self.get_original_size()
            if not orig_size: return
            w, h = orig_size.width(), orig_size.height()
            
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
        # Relative zoom is scale factor / fit_scale
        relative_zoom = 1.0
        if self._fit_scale > 0:
            if self.shared_state:
                relative_zoom = self.shared_state.zoom_multiplier
            else:
                relative_zoom = self._scale_factor / self._fit_scale

        return {
            'colormap': self.colormap,
            'contrast_limits': self.contrast_limits,
            'relative_zoom': relative_zoom,
            'relative_pan': self._get_effective_offset() # QPointF
        }

    def set_view_state(self, state):
        """Applies a view state dictionary to this label."""
        if not state:
            return

        # Apply Colormap
        if 'colormap' in state:
            self.set_colormap(state['colormap'])

        # Apply Contrast Limits
        if 'contrast_limits' in state:
            limits = state['contrast_limits']
            if limits:
                self.set_contrast_limits(*limits)
            else:
                self.contrast_limits = None
                self.apply_colormap()

        # Apply Zoom (Relative)
        if 'relative_zoom' in state:
            relative_zoom = state['relative_zoom']
            if self.shared_state:
                self.shared_state.zoom_multiplier = relative_zoom
            else:
                self.update_fit_scale()
                self._scale_factor = self._fit_scale * relative_zoom
        
        # Apply Pan (Relative to Fit which is offset 0,0)
        if 'relative_pan' in state:
            pan = state['relative_pan']
            if isinstance(pan, (list, tuple)) and len(pan) == 2:
                self.pan_pos = QPointF(pan[0], pan[1])
            else:
                self.pan_pos = pan

        self.zoom_factor_changed.emit(self._get_effective_scale_factor())
        self.view_changed.emit()

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
        
        # Debounce the expensive view changed signal (histogram update)
        self._view_update_timer.start()
        
        # Update overlay position (keep it floating on image)
        self._update_overlay_position()

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
        if event.type() == QEvent.Type.NativeGesture:
            if event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
                value = event.value()
                current_scale = self._get_effective_scale_factor()
                new_scale_factor = current_scale * (1.0 + value)
                # Cast to QPointF explicitly to avoid TypeError
                local_pos = QPointF(event.position())
                self._apply_zoom(new_scale_factor, local_pos)
                return True
        elif event.type() == QEvent.Type.Gesture:
            pinch = event.gesture(Qt.GestureType.PinchGesture)
            if pinch:
                if pinch.state() == Qt.GestureState.GestureStarted:
                    self._pinch_start_scale_factor = self._get_effective_scale_factor()
                elif pinch.state() == Qt.GestureState.GestureUpdated:
                    new_scale_factor = self._pinch_start_scale_factor * pinch.totalScaleFactor()
                    center_point = pinch.hotSpot()
                    # Cast to QPointF explicitly
                    local_pos = QPointF(self.mapFromGlobal(center_point.toPoint()))
                    self._apply_zoom(new_scale_factor, local_pos)
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

    def keyPressEvent(self, event):
        if event.key() in [Qt.Key.Key_Left, Qt.Key.Key_Right]:
            # Propagate to parent to handle video navigation
            event.ignore()
        else:
            super().keyPressEvent(event)

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
            
            # Update overlay position (keep it floating on image)
            self._update_overlay_position()
            
            # self.view_changed.emit() # Removed for performance, moved to release

        if self.original_data is not None and self.current_pixmap is not None:
             # Even if not active/focused, if we are in shared state we should update.
             # We want "seamless" movement.
             self._update_crosshair_from_pos(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start_position = QPoint()
            self.view_changed.emit() # Update histogram on release

    def set_overlays(self, overlays):
        """Sets the list of overlays to be drawn on top of the main image.
           overlays: List of (QPixmap, opacity) tuples. 
           Assumes pixmaps are already resized to match the current_pixmap size.
        """
        self.overlays = overlays
        self.update()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if os.path.exists(file_path):
                # Try to find load_image in the window hierarchy
                window = self.window()
                if hasattr(window, 'open_file'):
                    window.open_file(file_path)

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

        # Ensure Antialiasing is OFF for pixel-sharp rendering
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

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
            painter.save()
            # If the label is using a proxy, the painter coordinate system 
            # is currently aligned to PIXMAP pixels (scaled by draw_scale).
            # Overlays are now scaled to match the ORIGINAL resolution for sharpness.
            # So we must scale the painter by _proxy_scale to draw at original res.
            if self._proxy_scale > 0:
                painter.scale(self._proxy_scale, self._proxy_scale)
            
            # Get target dimensions (Original image space)
            orig_size = self.get_original_size()
            
            for overlay_pixmap, opacity in self.overlays:
                if overlay_pixmap and not overlay_pixmap.isNull():
                    painter.save()
                    painter.setOpacity(opacity)
                    
                    # Calculate effective scale of this overlay relative to screen pixels
                    # draw_scale is ScreenPixels / OriginalImagePixels
                    # overlay_rel_scale is OriginalImagePixels / OverlayPixels
                    overlay_rel_scale_w = orig_size.width() / overlay_pixmap.width() if orig_size and overlay_pixmap.width() > 0 else 1.0
                    overlay_rel_scale_h = orig_size.height() / overlay_pixmap.height() if orig_size and overlay_pixmap.height() > 0 else 1.0
                    
                    # Use average or aspect-ratio aware? Usually they match aspect ratio.
                    overlay_rel_scale = overlay_rel_scale_w 
                    effective_overlay_scale = draw_scale * overlay_rel_scale
                    
                    # Ensure Sharp Rendering for Each Overlay
                    # If effective scale > 1.05 (upscaling on screen), use FastTransformation (Nearest Neighbor)
                    if effective_overlay_scale > 1.05:
                        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.zoom_in_interp == Qt.TransformationMode.SmoothTransformation)
                    elif effective_overlay_scale < 0.95:
                        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.zoom_out_interp == Qt.TransformationMode.SmoothTransformation)
                    
                    # Instead of drawPixmap(target_rect), we scale the painter.
                    # This ensures Qt samples from the full-res source directly into the screen transform.
                    painter.scale(overlay_rel_scale_w, overlay_rel_scale_h)
                    painter.drawPixmap(0, 0, overlay_pixmap)
                    painter.restore()
            painter.setOpacity(1.0)
            painter.restore()

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

            # Calculate view position using target_rect (already accounts for proxy scale)
            pixel_scale_x = target_rect.width() / img_width
            pixel_scale_y = target_rect.height() / img_height
            view_x = (img_x * pixel_scale_x) + target_rect.left()
            view_y = (img_y * pixel_scale_y) + target_rect.top()
            
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

        if self.show_colorbar and self.current_pixmap:
            self.draw_colorbar(painter)

    def draw_colorbar(self, painter):
        if not self.original_data is not None:
             return

        # Dimensions regarding widget
        bar_w = 20
        bar_h = min(300, self.height() - 60)
        padding = 20
        
        # Position: Right side, vertically centered
        x = self.width() - bar_w - padding
        y = (self.height() - bar_h) // 2
        
        # Draw Background (Semi-transparent)
        bg_rect = QRectF(x - 5, y - 5, bar_w + 35, bar_h + 10) # Wider for text
        painter.fillRect(bg_rect, QColor(0, 0, 0, 150))
        
        # Draw Gradient Bar
        gradient = pyqtgraph_gradient = QRadialGradient() # Dummy, replaced below
        
        # Get Colormap colors
        try:
            # Create QLinearGradient
            qn_gradient = QLinearGradient(x, y + bar_h, x, y) # Bottom to Top
            
            # Matplotlib Colormap
            cmap_name = getattr(self, 'colormap', 'gray')
            if cmap_name == 'gray':
                 qn_gradient.setColorAt(0, QColor(0, 0, 0))
                 qn_gradient.setColorAt(1, QColor(255, 255, 255))
            else:
                 # Sample colormap at 0, 0.25, 0.5, 0.75, 1
                 cmap = cm.get_cmap(cmap_name)
                 for i in range(11):
                     pos = i / 10.0
                     rgba = cmap(pos)
                     color = QColor(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
                     qn_gradient.setColorAt(pos, color)
            
            painter.fillRect(QRectF(x, y, bar_w, bar_h), QBrush(qn_gradient))
            
            # Draw Border
            painter.setPen(QPen(QColor(200, 200, 200), 1))
            painter.drawRect(QRectF(x, y, bar_w, bar_h))
            
            # Draw Text Limits (Min/Max)
            min_val, max_val = 0.0, 1.0
            if self.contrast_limits:
                min_val, max_val = self.contrast_limits
            elif self.original_data is not None:
                min_val = np.min(self.original_data)
                max_val = np.max(self.original_data)
                
            font = painter.font()
            font.setPointSize(10)
            painter.setFont(font)
            painter.setPen(QColor(255, 255, 255))
            
            # Format numbers (scientific if too large/small)
            def fmt(v):
                if abs(v) > 10000 or (abs(v) < 0.01 and v != 0):
                    return f"{v:.1e}"
                return f"{v:.2f}"
            
            # Max (Top)
            painter.drawText(QRectF(x - 45, y - 10, 40, 20), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, fmt(max_val))
            
            # Min (Bottom)
            painter.drawText(QRectF(x - 45, y + bar_h - 10, 40, 20), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, fmt(min_val))

        except Exception as e:
            print(f"Error drawing colorbar: {e}")



class ThumbnailItem(QWidget):
    clicked = pyqtSignal(QMouseEvent)
    overlay_changed = pyqtSignal(float)
    remove_requested = pyqtSignal()

    def __init__(self, file_path, pixmap, removable=True, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.is_selected = False
        self.is_focused = False

        self.setFixedSize(160, 180) # Increased height for slider
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5) # Reduce margins
        self.image_label = QLabel()
        # Use FastTransformation (Nearest Neighbor) for upscaling small images to keep them sharp,
        # and SmoothTransformation for downscaling large images to avoid aliasing.
        transform_mode = Qt.TransformationMode.SmoothTransformation
        if pixmap.width() < 128 or pixmap.height() < 128:
            transform_mode = Qt.TransformationMode.FastTransformation
            
        self.image_label.setPixmap(
            pixmap.scaled(128, 128, Qt.AspectRatioMode.KeepAspectRatio, transform_mode))
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
        self.slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.slider.setToolTip("Overlay Opacity")
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider)

        self.setLayout(layout)

        # Remove Button (Top Right)
        self.removable = removable
        self.remove_btn = QPushButton("×", self)
        self.remove_btn.setFixedSize(20, 20)
        self.remove_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(128, 128, 128, 150);
                color: white;
                border-radius: 10px;
                font-weight: bold;
                font-size: 14px;
                border: none;
            }
            QPushButton:hover {
                background-color: rgba(255, 0, 0, 200);
            }
        """)
        self.remove_btn.move(135, 5) # Top right corner
        self.remove_btn.clicked.connect(self.remove_requested.emit)
        self.remove_btn.setVisible(self.removable)

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
        
        # Determine border color and width based on state
        if self.is_selected and self.is_focused:
            # Both selected and focused - use orange/gold to show both states
            pen_color = QColor(255, 140, 0)  # Orange
            pen_width = 4
            adjustment = 0  # Extend further out
        elif self.is_selected:
            pen_color = Qt.GlobalColor.blue
            pen_width = 3
            adjustment = 1
        elif self.is_focused:
            pen_color = Qt.GlobalColor.yellow
            pen_width = 4
            adjustment = -1  # Negative adjustment makes rectangle larger (extends beyond edges)
        else:
            return

        pen = painter.pen()
        pen.setColor(pen_color)
        pen.setWidth(pen_width)
        painter.setPen(pen)
        painter.drawRect(self.rect().adjusted(adjustment, adjustment, -adjustment, -adjustment))


class ThumbnailPane(QDockWidget):
    selection_changed = pyqtSignal(list)
    overlay_changed = pyqtSignal(str, float)

    def __init__(self, parent=None):
        super().__init__("Images", parent)
        self.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.setFloating(False)
        self.setMinimumWidth(200)  # Increased to show full thumbnail width

        self.main_widget = QWidget()
        self.main_widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.main_widget.setAcceptDrops(True) # Ensure main widget accepts drops
        self.main_layout = QVBoxLayout(self.main_widget)
        
        # Header layout (Select All + Refresh)
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 5, 0)
        
        self.select_all_cb = QCheckBox("Select All")
        self.select_all_cb.setTristate(True)  # Allow indeterminate state
        self.select_all_cb.stateChanged.connect(self._on_select_all_changed)
        header_layout.addWidget(self.select_all_cb)
        
        header_layout.addStretch()
        
        self.refresh_btn = QPushButton("🗘")
        self.refresh_btn.setFixedSize(24, 24)
        self.refresh_btn.setToolTip("Refresh Gallery (Prune closed images)")
        self.refresh_btn.setFlat(True)
        self.refresh_btn.setStyleSheet("""
            QPushButton { 
                border: none; 
                font-size: 16px; 
                color: #555;
            }
            QPushButton:hover { 
                background-color: rgba(0,0,0,0.1); 
                border-radius: 4px;
                color: #000;
            }
        """)
        self.refresh_btn.clicked.connect(self.populate)
        header_layout.addWidget(self.refresh_btn)
        
        self.main_layout.addLayout(header_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAcceptDrops(True)
        self.scroll_area.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self.scroll_area.viewport().setAcceptDrops(True) # Explicitly accept on viewport
        self.scroll_area.viewport().setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self.scroll_content = QWidget()
        self.scroll_content.setAcceptDrops(True)
        self.thumbnail_layout = QVBoxLayout(self.scroll_content)
        self.thumbnail_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setWidget(self.scroll_content)

        self.scroll_area.installEventFilter(self)
        self.scroll_area.viewport().installEventFilter(self)
        self.scroll_content.installEventFilter(self)

        self.main_layout.addWidget(self.scroll_area)
        self.setWidget(self.main_widget)

        self.thumbnail_items = []
        self.focused_index = -1
        self.last_selection_anchor = -1
        
        # Persistent gallery: {file_path: pixmap}
        self.gallery_images = {}
        # Metadata for gallery images: {file_path: {"is_manual": bool}}
        self.gallery_meta = {} # Deprecated, using manual_paths
        self.manual_paths = set()
        self.removed_paths = set()
        
        # Enable Drag and Drop
        self.setAcceptDrops(True)

    def showEvent(self, event):
        """Refresh thumbnail pane contents when shown"""
        super().showEvent(event)
        # Trigger a refresh from the parent ImageViewer
        if self.parent() and hasattr(self.parent(), '_refresh_thumbnail_pane'):
            self.parent()._refresh_thumbnail_pane()

    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.KeyPress:
            if source in (self.scroll_area, self.scroll_area.viewport(), self.scroll_content) or isinstance(source, ThumbnailItem):
                key = event.key()
                if key in [Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Left, Qt.Key.Key_Right, 
                           Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter,
                           Qt.Key.Key_Delete, Qt.Key.Key_Backspace]:
                    self.keyPressEvent(event)
                    return True
        return super().eventFilter(source, event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            paths = [u.toLocalFile() for u in urls if u.toLocalFile()]
            self.add_files(paths)

    def add_files(self, file_paths):
        if not file_paths:
            return

        from image_handler import ImageHandler
        temp_handler = ImageHandler()
        supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp', '.heic', '.heif'] + \
                             temp_handler.raw_extensions + temp_handler.video_extensions
        
        added_any = False
        for path in file_paths:
            if os.path.isfile(path) and path not in self.gallery_images:
                _, ext = os.path.splitext(path)
                if ext.lower() in supported_extensions:
                    # Generate thumbnail
                    try:
                        if temp_handler.load_image(path):
                            pixmap = self._generate_pixmap_from_array(temp_handler.original_image_data)
                            if pixmap:
                                self.gallery_images[path] = pixmap
                                self.manual_paths.add(path)
                                if path in self.removed_paths:
                                    self.removed_paths.remove(path)
                                added_any = True
                    except Exception as e:
                        print(f"Error generating thumbnail for {path}: {e}")

        if added_any:
            self.populate()

    def keyPressEvent(self, event: QKeyEvent):
        if not self.thumbnail_items:
            super().keyPressEvent(event)
            return

        key = event.key()
        current_index = self.focused_index if self.focused_index != -1 else 0

        # 1. Navigation and selection keys
        if key in (Qt.Key.Key_Down, Qt.Key.Key_Up, Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            new_index = current_index
            
            # Simple Grid/List navigation
            if key in (Qt.Key.Key_Down, Qt.Key.Key_Right):
                new_index = min(current_index + 1, len(self.thumbnail_items) - 1)
            elif key in (Qt.Key.Key_Up, Qt.Key.Key_Left):
                new_index = max(current_index - 1, 0)

            modifiers = event.modifiers()
            is_shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)

            # Move focus
            if new_index != current_index:
                self._set_focused_item(new_index)
                
                if is_shift:
                     # Extend selection
                     anchor = self.last_selection_anchor if self.last_selection_anchor != -1 else current_index
                     self._select_range(anchor, new_index)
                     self._emit_selection_change()
                else:
                     self._select_single(new_index)
                     self.last_selection_anchor = new_index
                     self._emit_selection_change()
                     
            # Enter/Return/Space: Toggle
            elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
                if modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
                     self._toggle_selection(current_index)
                else:
                     self._select_single(current_index)
                self.last_selection_anchor = current_index
                self._emit_selection_change()
        
        # 2. Deletion keys
        elif key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            # Only remove if items are removable
            self.remove_selected_items()
            
        else:
            # Pass all other keys to parent (ImageViewer) for global shortcuts
            super().keyPressEvent(event)

    def remove_selected_items(self):
        """Remove selected items from the gallery and add them to the removed filter."""
        # Filter for removable items only
        selected_items = [item for item in self.thumbnail_items 
                          if item.is_selected and item.removable]
        if not selected_items:
            return
            
        for item in selected_items:
            self.remove_item(item.file_path, refresh=False)

        # Update sidebar
        self.populate()
        
        # If gallery is now TRULY empty (no removable or synced items), clear viewer
        if not self.gallery_images:
             if self.parent() and hasattr(self.parent(), 'display_montage'):
                 self.parent().display_montage([], is_manual=False)
        else:
            # Otherwise, just sync the viewer to whatever is left selected 
            # (e.g. non-removable items that were part of the selection)
            self._emit_selection_change() 

    def remove_item(self, path, refresh=True):
        """Remove a single item from the gallery."""
        self.removed_paths.add(path)
        if path in self.gallery_images:
            del self.gallery_images[path]
        if path in self.manual_paths:
            self.manual_paths.remove(path)
            
        if refresh:
            self.populate()
            self._emit_selection_change()
            
            # If gallery is now empty, clear main viewer
            if not self.gallery_images:
                if self.parent() and hasattr(self.parent(), 'display_montage'):
                    self.parent().display_montage([])

    def unremove_files(self, file_paths):
        """Explicitly restore files to the sidebar if they were previously removed."""
        if not file_paths:
            return
        
        changed = False
        for path in file_paths:
            if path in self.removed_paths:
                self.removed_paths.remove(path)
                changed = True
        
        if changed:
            self.populate()

    def add_to_manual_paths(self, file_paths):
        """Mark files as manually added (removable)."""
        if not file_paths:
            return
        for path in file_paths:
            self.manual_paths.add(path)
        self.populate()

    def populate(self, windows=None):
        """Populate thumbnail pane from persistent gallery and open windows."""
        # Don't populate if blocked (during selection-driven view changes)
        if hasattr(self, 'block_populate') and self.block_populate:
            return
        
        # 0. Track paths found in currently open windows
        seen_in_windows = set()
        
        # 1. Update gallery with images from all windows
        for widget in QApplication.topLevelWidgets():
            if widget.__class__.__name__ == 'ImageViewer':
                if hasattr(widget, 'stacked_widget') and hasattr(widget, 'montage_widget'):
                    if widget.stacked_widget.currentWidget() == widget.montage_widget and widget.montage_labels:
                        # Montage mode: add all montage images to gallery
                        for label in widget.montage_labels:
                            if hasattr(label, 'file_path') and label.current_pixmap and label.file_path:
                                self.gallery_images[label.file_path] = label.current_pixmap
                                seen_in_windows.add(label.file_path)
                    elif widget.image_label.current_pixmap and widget.current_file_path:
                        # Single image mode: add to gallery
                        # Check if this is an NPZ file with multiple keys
                        if hasattr(widget.image_handler, 'npz_keys') and len(widget.image_handler.npz_keys) > 1:
                            # Add thumbnails for each valid NPZ key
                            for key in widget.image_handler.npz_keys.keys():
                                if widget.image_handler.npz_keys[key]:  # Only valid keys
                                    # Create a unique identifier for this NPZ key
                                    npz_key_path = f"{widget.current_file_path}#{key}"
                                    seen_in_windows.add(npz_key_path)
                                    # Get or generate pixmap for this key
                                    if key == widget.image_handler.current_npz_key:
                                        # Current key - use existing pixmap
                                        self.gallery_images[npz_key_path] = widget.image_label.current_pixmap
                                    else:
                                        # Other keys - generate thumbnail
                                        pixmap = self._generate_pixmap_from_array(widget.image_handler.npz_data[key])
                                        if pixmap:
                                            self.gallery_images[npz_key_path] = pixmap
                        else:
                            # Regular file
                            self.gallery_images[widget.current_file_path] = widget.image_label.current_pixmap
                            seen_in_windows.add(widget.current_file_path)

        # 1.5 Prune "synced" images that are no longer in any window
        # (Rule: If it's NOT manual and NOT in a window, it's stale)
        stale_paths = []
        for path in list(self.gallery_images.keys()):
            if path not in self.manual_paths and path not in seen_in_windows:
                stale_paths.append(path)
        
        for path in stale_paths:
            if path in self.gallery_images:
                del self.gallery_images[path]
            # Also clean up from removed_paths if it's no longer even synced
            if path in self.removed_paths:
                self.removed_paths.remove(path)
        
        # 2. Capture current selection
        selected_paths = {item.file_path for item in self.thumbnail_items if item.is_selected}
        
        # 3. Clear all existing thumbnail widgets
        for item in self.thumbnail_items:
            self.thumbnail_layout.removeWidget(item)
            item.deleteLater()
        self.thumbnail_items.clear()

        # 4. Create thumbnail items from gallery
        for file_path, pixmap in self.gallery_images.items():
            if file_path in self.removed_paths:
                continue
            
            # Removable IF it was manually added 
            # (Rule 1 & 2: Added to main pane or sidebar directly)
            is_removable = file_path in self.manual_paths
                
            item = ThumbnailItem(file_path, pixmap, removable=is_removable)
            item.installEventFilter(self)
            item.clicked.connect(lambda event, i=item: self._on_thumbnail_clicked(i, event))
            item.remove_requested.connect(lambda p=file_path: self.remove_item(p))
            item.overlay_changed.connect(lambda alpha, p=file_path: self.overlay_changed.emit(p, alpha))
            
            # Restore selection
            if file_path in selected_paths:
                item.set_selected(True)
                
            self.thumbnail_items.append(item)
            self.thumbnail_layout.addWidget(item)

        # 5. Handle focus
        if self.thumbnail_items:
             if self.focused_index >= len(self.thumbnail_items):
                 self.focused_index = len(self.thumbnail_items) - 1
             if self.focused_index == -1:
                 self.focused_index = 0
             self._set_focused_item(self.focused_index)
    
    def _generate_pixmap_from_array(self, array_data):
        """Generate a QPixmap for a numpy array's data (used for thumbnails)."""
        try:
            import numpy as np
            from PyQt6.QtGui import QImage, QPixmap
            
            data = array_data.copy()
            
            # Normalize to 0-255 range
            if data.dtype != np.uint8:
                data_min = data.min()
                data_max = data.max()
                if data_max > data_min:
                    data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
                else:
                    data = np.zeros_like(data, dtype=np.uint8)
            
            # Convert to QImage
            if data.ndim == 2:
                # Grayscale
                height, width = data.shape
                bytes_per_line = width
                qimage = QImage(data.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
            elif data.ndim == 3:
                height, width, channels = data.shape
                if channels == 3:
                    # RGB
                    bytes_per_line = width * 3
                    qimage = QImage(data.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                elif channels == 4:
                    # RGBA
                    bytes_per_line = width * 4
                    qimage = QImage(data.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888)
                elif channels == 2:
                    # RG - convert to RGB by adding zeros for B
                    rgb_data = np.zeros((height, width, 3), dtype=np.uint8)
                    rgb_data[:, :, :2] = data
                    bytes_per_line = width * 3
                    qimage = QImage(rgb_data.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                else:
                    return None
            else:
                return None
            
            return QPixmap.fromImage(qimage.copy())
        except Exception as e:
            print(f"Error generating NPZ key pixmap: {e}")
            return None
        
    # (Resuming actual content for replacement)
    
    def _on_thumbnail_clicked(self, item, event):
        if item not in self.thumbnail_items:
            return
        index = self.thumbnail_items.index(item)
        self._handle_selection(index, event.modifiers())

    def _handle_selection(self, index, modifiers):
        self._set_focused_item(index)

        is_shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        is_ctrl = bool(modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier))

        if is_shift:
            anchor = self.last_selection_anchor if self.last_selection_anchor != -1 else index
            self._select_range(anchor, index)
        elif is_ctrl:
            # Toggle (Cmd/Ctrl + Click)
            self._toggle_selection(index)
            self.last_selection_anchor = index
        else:
            # Default: Select Single (Exclusive)
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
        
        # Update Select All checkbox state (block signals to prevent loop)
        self.select_all_cb.blockSignals(True)
        if not self.thumbnail_items:
            self.select_all_cb.setCheckState(Qt.CheckState.Unchecked)
        else:
            selected_count = len(selected_files)
            total_count = len(self.thumbnail_items)
            if selected_count == 0:
                self.select_all_cb.setCheckState(Qt.CheckState.Unchecked)
            elif selected_count == total_count:
                self.select_all_cb.setCheckState(Qt.CheckState.Checked)
            else:
                self.select_all_cb.setCheckState(Qt.CheckState.PartiallyChecked)
        self.select_all_cb.blockSignals(False)
    
    def _on_select_all_changed(self, state):
        """Handle Select All checkbox changes"""
        if not self.thumbnail_items:
            return
        
        # Check current selection state to determine action
        currently_selected = sum(1 for item in self.thumbnail_items if item.is_selected)
        
        # Block signals to prevent recursion
        self.select_all_cb.blockSignals(True)
        
        # If nothing or partial selected, select all. If all selected, deselect all.
        if currently_selected < len(self.thumbnail_items):
            # Select all
            for item in self.thumbnail_items:
                item.set_selected(True)
            self.select_all_cb.setCheckState(Qt.CheckState.Checked)
        else:
            # Deselect all
            for item in self.thumbnail_items:
                item.set_selected(False)
            self.select_all_cb.setCheckState(Qt.CheckState.Unchecked)
        
        self.select_all_cb.blockSignals(False)
        self._emit_selection_change()

    def _set_focused_item(self, index):
        if not self.thumbnail_items:
            return
        
        # Bounds check
        if index < 0 or index >= len(self.thumbnail_items):
            return
        
        if self.focused_index != -1 and self.focused_index < len(self.thumbnail_items):
            self.thumbnail_items[self.focused_index].set_focused(False)

        self.focused_index = index

        if self.focused_index != -1 and self.focused_index < len(self.thumbnail_items):
            self.thumbnail_items[self.focused_index].set_focused(True)
            self.scroll_area.ensureWidgetVisible(self.thumbnail_items[self.focused_index])





class SmartSortFilterProxyModel(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_root_path = ""
        self.hide_folders = False

    def set_current_root_path(self, path):
        self.current_root_path = os.path.normpath(path) if path else ""
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row, source_parent):
        model = self.sourceModel()
        index = model.index(source_row, 0, source_parent)
        
        # Check if it is an ancestor of the current view root (or root itself)
        if model.isDir(index):
            path = os.path.normpath(model.filePath(index))
            if self.current_root_path:
                if self.current_root_path == path:
                    return True
                
                # Check strict ancestry
                # Add separator to path if missing to prevent partial matches (e.g. /usr matching /usr_local)
                # But do NOT add if it is the root path (/) which already has it
                check_path = path
                if not check_path.endswith(os.sep):
                    check_path += os.sep
                    
                if self.current_root_path.startswith(check_path):
                    return True
                
                # If we are here, it's a directory that is NOT an ancestor
                if self.hide_folders:
                    return False
                
        return super().filterAcceptsRow(source_row, source_parent)

class BreadcrumbBar(QWidget):
    path_clicked = pyqtSignal(str)
    edit_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(0)
        self.setFixedHeight(28) # Reasonable height

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.edit_requested.emit()
        super().mousePressEvent(event)

    def set_path(self, path):
        # Clear existing
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if not path:
             return

        # Normalize and split
        path = os.path.normpath(path)
        parts = []
        if os.name == 'nt': # Windows
             # Handle drive letter
             drive, tail = os.path.splitdrive(path)
             if drive: parts.append(drive + os.sep)
             parts.extend([p for p in tail.split(os.sep) if p])
        else:
             if path.startswith(os.sep):
                 parts.append(os.sep) # Root
             parts.extend([p for p in path.split(os.sep) if p])

        current_build_path = ""
        for i, part in enumerate(parts):
            if i == 0 and os.name != 'nt':
                 if part == os.sep:
                      current_build_path = os.sep
                      btn_text = "/"
                 else:
                      current_build_path = part
                      btn_text = part
            else:
                 current_build_path = os.path.join(current_build_path, part)
                 btn_text = part
            
            # Button
            btn = QPushButton(btn_text)
            btn.setFlat(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            
            # Styling to make it look like text/breadcrumb
            btn.setStyleSheet("""
                QPushButton {
                    border: none;
                    padding: 2px 5px;
                    text-align: left;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: rgba(128, 128, 128, 50);
                    border-radius: 3px;
                }
            """)
            
            # Capture path for closure
            # Lambda default argument trick
            btn.clicked.connect(lambda checked, p=current_build_path: self.path_clicked.emit(p))
            
            self.layout.addWidget(btn)
            
            # Arrow (if not last)
            if i < len(parts) - 1:
                arrow = QLabel(">")
                arrow.setStyleSheet("color: gray; padding: 0 2px;")
                self.layout.addWidget(arrow)

        # Edit Button to make manual entry discoverable
        edit_btn = QPushButton("✎") 
        edit_btn.setFlat(True)
        edit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        edit_btn.setToolTip("Edit Address")
        edit_btn.setFixedWidth(20)
        edit_btn.setStyleSheet("""
            QPushButton { 
                border: none; 
                color: #666; 
                font-size: 14px;
            } 
            QPushButton:hover { 
                color: #333; 
                background-color: rgba(0,0,0,0.1); 
                border-radius: 3px; 
            }
        """)
        edit_btn.clicked.connect(self.edit_requested.emit)
        self.layout.addWidget(edit_btn)

        # History Button (Dropdown Arrow)
        hist_btn = QPushButton("▼") 
        hist_btn.setFlat(True)
        hist_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        hist_btn.setToolTip("Recent Folders")
        hist_btn.setFixedWidth(20)
        hist_btn.setStyleSheet("""
            QPushButton { 
                border: none; 
                color: #666; 
                font-size: 10px;
                padding-top: 2px;
            } 
            QPushButton:hover { 
                color: #333; 
                background-color: rgba(0,0,0,0.1); 
                border-radius: 3px; 
            }
        """)
        hist_btn.clicked.connect(self.edit_requested.emit)
        self.layout.addWidget(hist_btn)

        self.layout.addStretch()

class SmartAddressBar(QWidget):
    path_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QStackedLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(0)
        
        # Breadcrumb View
        self.breadcrumbs = BreadcrumbBar()
        self.breadcrumbs.path_clicked.connect(self._on_breadcrumb_clicked)
        self.breadcrumbs.edit_requested.connect(self.show_edit)
        self.layout.addWidget(self.breadcrumbs)
        
        # Edit View Container (Combo + Cancel Button)
        self.edit_container = QWidget()
        edit_layout = QHBoxLayout(self.edit_container)
        edit_layout.setContentsMargins(0, 0, 0, 0)
        edit_layout.setSpacing(2)

        self.combo = QComboBox()
        self.combo.setEditable(True)
        self.combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.combo.setMinimumWidth(50)
        self.combo.lineEdit().returnPressed.connect(self._on_combo_entered)
        self.combo.activated.connect(self._on_combo_activated)
        self.combo.installEventFilter(self)
        self.combo.lineEdit().installEventFilter(self)
        edit_layout.addWidget(self.combo)
        
        self.cancel_btn = QToolButton()
        self.cancel_btn.setText("✖") 
        self.cancel_btn.setToolTip("Cancel Edit")
        self.cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_btn.clicked.connect(self._cancel_edit)
        self.cancel_btn.setStyleSheet("""
            QToolButton { border: none; color: #888; font-weight: bold; } 
            QToolButton:hover { color: #d00; background-color: rgba(0,0,0,0.1); border-radius: 3px; }
        """)
        edit_layout.addWidget(self.cancel_btn)

        self.layout.addWidget(self.edit_container)
        
        self.recent_paths = load_folder_history()
        self.combo.addItems(self.recent_paths)
        self._current_path = ""

    def show_edit(self):
        self.layout.setCurrentWidget(self.edit_container)
        self.combo.setFocus()
        self.combo.lineEdit().selectAll()
        if self.recent_paths:
             self.combo.showPopup()

    def set_path(self, path):
        self._current_path = path
        # Update Breadcrumbs
        self.breadcrumbs.set_path(path)
        
        # Update Combo Text
        self.combo.blockSignals(True)
        self.combo.setEditText(path)
        self.combo.blockSignals(False)
        
        # Add to recent
        if path and os.path.exists(path):
             self._add_recent(path)
             
        # Switch to Breadcrumbs
        self.layout.setCurrentWidget(self.breadcrumbs)

    def _add_recent(self, path):
        if path in self.recent_paths:
             self.recent_paths.remove(path)
        self.recent_paths.insert(0, path)
        self.recent_paths = self.recent_paths[:10] 
        
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItems(self.recent_paths)
        self.combo.setEditText(self._current_path)
        self.combo.blockSignals(False)
        save_folder_history(self.recent_paths)

    def _on_breadcrumb_clicked(self, path):
        self.path_changed.emit(path)

    def _on_combo_entered(self):
        path = self.combo.currentText()
        if path and os.path.isdir(path):
             self.path_changed.emit(path)
             self.layout.setCurrentWidget(self.breadcrumbs)
        else:
             pass 

    def _on_combo_activated(self, index):
        path = self.combo.itemText(index)
        if path and os.path.isdir(path):
             self.path_changed.emit(path)
             self.layout.setCurrentWidget(self.breadcrumbs)

    def _cancel_edit(self):
        self.layout.setCurrentWidget(self.breadcrumbs)
        self.combo.setEditText(self._current_path)

    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
             self._cancel_edit()
             return True

        if source == self.combo and event.type() == QEvent.Type.FocusOut:
            if not self.combo.view().isVisible():
                 # Don't cancel immediately if focus went to cancel button
                 # But cancel button click triggers logic anyway.
                 # If focus went to cancel button, we rely on click.
                 # If we return, we might close before click?
                 # Actually if we click Cancel, focus out happens first.
                 # If we close edit view on focus out, cancel button might move or disappear.
                 # Let's check focus widget.
                 focus_widget = QApplication.focusWidget()
                 if focus_widget != self.cancel_btn:
                     self._cancel_edit()
            return False
            
        return super().eventFilter(source, event)

class FileExplorerPane(QDockWidget):
    files_selected = pyqtSignal(list)

    def sizeHint(self):
        # Suggest a wider initial size (450px)
        return QSize(450, 600)

    def __init__(self, parent=None):
        super().__init__("Explorer", parent)
        self.setObjectName("FileExplorerPane")
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)
        self.setMinimumWidth(300)
        self.target_selection_path = None

        # Initialize Models FIRST to prevent AttributeError on widget signal triggers
        # File System Model
        self.model = QFileSystemModel()
        self.model.setRootPath(QDir.rootPath())
        self.model.setNameFilters(["*"]) 
        self.model.setNameFilterDisables(False) 
        
        # Proxy Model
        self.proxy_model = SmartSortFilterProxyModel()
        self.proxy_model.setSourceModel(self.model)
        self.proxy_model.setFilterKeyColumn(0) # Name column
        self.proxy_model.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.proxy_model.setDynamicSortFilter(True)
        # self.proxy_model.layoutChanged.connect(self._select_first_item)
        self.model.directoryLoaded.connect(self._on_directory_loaded)

        self.content_widget = QWidget()
        self.content_widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.layout = QVBoxLayout(self.content_widget)

        # Navigation Bar (Up Button + Path Input)
        nav_layout = QHBoxLayout()
        
        self.up_button = QToolButton()
        self.up_button.setIcon(QIcon.fromTheme("go-up"))
        if self.up_button.icon().isNull(): # Fallback if theme icon missing
             self.up_button.setText("..") 
        self.up_button.clicked.connect(self._on_up_clicked)
        nav_layout.addWidget(self.up_button)
        
        self.path_input = SmartAddressBar()
        self.path_input.path_changed.connect(self.set_root_path)
        nav_layout.addWidget(self.path_input)
        
        self.layout.addLayout(nav_layout)

        # Filter Input with History
        self.filter_input = QComboBox()
        self.filter_input.setEditable(True)
        self.filter_input.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.filter_input.lineEdit().setPlaceholderText("Filter files (e.g. *.png, image_*)")
        self.filter_input.editTextChanged.connect(self._on_filter_changed)
        self.filter_input.lineEdit().returnPressed.connect(self._add_filter_history)
        self.layout.addWidget(self.filter_input)
        
        self.filter_history = load_filter_history()
        self.filter_input.addItems(self.filter_history)
        self.filter_input.setEditText("")
        
        # Options Bar (Hide Folders Checkbox)
        options_layout = QHBoxLayout()
        self.hide_folders_cb = QCheckBox("Hide Folders")
        self.hide_folders_cb.stateChanged.connect(self._on_hide_folders_changed)
        options_layout.addWidget(self.hide_folders_cb)
        options_layout.addStretch()
        self.layout.addLayout(options_layout)
        
        # Tree View
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.proxy_model)
        self.tree_view.setRootIsDecorated(True)
        self.tree_view.setSortingEnabled(True)
        self.tree_view.setHeaderHidden(False)
        self.tree_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tree_view.setColumnWidth(0, 450)
        for i in range(1, self.model.columnCount()):
            self.tree_view.setColumnHidden(i, True)
        self.tree_view.header().setStretchLastSection(True)
        self.tree_view.setMinimumWidth(300)
        
        self.tree_view.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self.tree_view.activated.connect(self._on_item_activated) # Double click or enter
        self.tree_view.installEventFilter(self)
        self.layout.addWidget(self.tree_view)

        # Video Creation
        self.create_video_btn = QPushButton("Create Video from Selection")
        self.create_video_btn.setEnabled(False)
        self.create_video_btn.clicked.connect(self._create_video_from_selection)
        self.layout.addWidget(self.create_video_btn)

        self.setWidget(self.content_widget)
        
        self.set_root_path(QDir.homePath())

    def showEvent(self, event):
        super().showEvent(event)
        # Ensure focus goes to tree when shown so selection is "active" (e.g. blue not gray)
        self.tree_view.setFocus()

    def set_supported_extensions(self, extensions):
        """Set filename filters for the file system model.
           extensions: list of strings like ['*.png', '*.jpg']
        """
        if extensions:
            self.model.setNameFilters(extensions)
            self.model.setNameFilterDisables(False) # Hide filtered files
        else:
             self.model.setNameFilters(["*"])
             
    def eventFilter(self, source, event):
        if source == self.tree_view and event.type() == QEvent.Type.KeyPress:
            # Handle Enter/Return explicitly for navigation (Folder Dive-in)
            if event.key() in [Qt.Key.Key_Return, Qt.Key.Key_Enter]:
                indexes = self.tree_view.selectionModel().selectedRows()
                if indexes:
                    self._on_item_activated(indexes[0])
                    return True

            # Handle Backspace explicitly for navigation (Folder Up)
            if event.key() == Qt.Key.Key_Backspace:
                self._on_up_clicked()
                return True

            # Keys allowed to remain in the tree view for navigational use
            # Strictly: Up, Down, Enter, and Backspace
            allowed_keys = [
                Qt.Key.Key_Up, 
                Qt.Key.Key_Down, 
                Qt.Key.Key_Return, 
                Qt.Key.Key_Enter,
                Qt.Key.Key_Backspace
            ]
            
            if event.key() not in allowed_keys:
                # Forward to main window (shortcuts, pan, etc.)
                main_win = self.window()
                if main_win:
                    main_win.keyPressEvent(event)
                    return True # Consume to prevent tree_view from using it (e.g. searching by letter)
                    
        return super().eventFilter(source, event)
             
    def _on_item_activated(self, index):
        source_index = self.proxy_model.mapToSource(index)
        if self.model.isDir(source_index):
            file_path = self.model.filePath(source_index)
            self.set_root_path(file_path)
        else:
            self._emit_selection()

    def _on_up_clicked(self):
        old_root = self.root_path
        parent_dir = os.path.dirname(self.root_path)
        if parent_dir and os.path.exists(parent_dir):
            self.target_selection_path = old_root
            self.set_root_path(parent_dir)

    def set_root_path(self, path):
        if not path:
             path = QDir.homePath()
        elif not os.path.isdir(path):
            path = os.path.dirname(path) if os.path.exists(path) else QDir.homePath()
            
        if hasattr(self, 'root_path') and self.root_path == path:
             return

        self.root_path = path
        self.path_input.set_path(path)
        self.proxy_model.set_current_root_path(path)
        self._set_view_to_path(path)
        self.tree_view.clearSelection()
        
        # Manually trigger selection update if we are not waiting for directory loaded
        # (Though usually QFileSystemModel loads async so directoryLoaded will fire)
        self._restore_selection()

    def _on_directory_loaded(self, path):
         if os.path.normpath(path) == os.path.normpath(self.root_path):
             self._restore_selection()

    def _restore_selection(self):
        root_index = self.tree_view.rootIndex()
        
        # Try to select the target path (context-aware up)
        if self.target_selection_path:
            target_source_index = self.model.index(self.target_selection_path)
            if target_source_index.isValid():
                target_proxy_index = self.proxy_model.mapFromSource(target_source_index)
                if target_proxy_index.isValid() and target_proxy_index.parent() == root_index:
                    self.tree_view.setCurrentIndex(target_proxy_index)
                    self.tree_view.selectionModel().select(target_proxy_index, QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows)
                    self.tree_view.scrollTo(target_proxy_index)
                    self.target_selection_path = None
                    if self.isVisible():
                        self.tree_view.setFocus()
                    return

                    if self.isVisible():
                        self.tree_view.setFocus()
                    return

        # Fallback to first item REMOVED to prevent auto-selection
                # If the explorer is active, make sure the tree is focused for visual clarity
                if self.isVisible():
                    self.tree_view.setFocus()
        
    def _set_view_to_path(self, path):
        root_index = self.model.index(path)
        proxy_root_index = self.proxy_model.mapFromSource(root_index)
        self.tree_view.setRootIndex(proxy_root_index)

    def _on_hide_folders_changed(self, state):
        self.proxy_model.hide_folders = (state == Qt.CheckState.Checked.value or state == 2)
        self.proxy_model.invalidateFilter()

    def _on_filter_changed(self, text):
        if not text:
            self.proxy_model.setFilterRegularExpression("")
            return
            
        pattern = None
        if '*' in text or '?' in text:
             import re
             escaped = re.escape(text)
             regex_pattern = escaped.replace(r'\*', '.*').replace(r'\?', '.')
             pattern = QRegularExpression(regex_pattern, QRegularExpression.PatternOption.CaseInsensitiveOption)
        else:
             pattern = QRegularExpression(text, QRegularExpression.PatternOption.CaseInsensitiveOption)
             if not pattern.isValid():
                  # Fallback to literal search if invalid regex
                  pattern = QRegularExpression(QRegularExpression.escape(text), QRegularExpression.PatternOption.CaseInsensitiveOption)

        self.proxy_model.setFilterRegularExpression(pattern)

    def _add_filter_history(self):
         text = self.filter_input.currentText().strip()
         if not text: return
         
         if text in self.filter_history:
             self.filter_history.remove(text)
         self.filter_history.insert(0, text)
         self.filter_history = self.filter_history[:30]
         
         # Persist current text across update
         self.filter_input.blockSignals(True)
         self.filter_input.clear()
         self.filter_input.addItems(self.filter_history)
         self.filter_input.setEditText(text)
         self.filter_input.blockSignals(False)
         self.filter_input.clearFocus()
         save_filter_history(self.filter_history)

    def _on_selection_changed(self, selected, deselected):
        self._emit_selection()


    def _emit_selection(self):
        indexes = self.tree_view.selectionModel().selectedRows()
        file_paths = []
        for index in indexes:
            source_index = self.proxy_model.mapToSource(index)
            path = self.model.filePath(source_index)
            if not self.model.isDir(source_index):
                file_paths.append(path)
        
        if file_paths:
            self.files_selected.emit(file_paths)
            
        self.create_video_btn.setEnabled(len(file_paths) > 1)

    def _create_video_from_selection(self):
        indexes = self.tree_view.selectionModel().selectedRows()
        # Sort by row to match visual order
        indexes = sorted(indexes, key=lambda i: i.row())
        
        file_paths = []
        for index in indexes:
            source_index = self.proxy_model.mapToSource(index)
            path = self.model.filePath(source_index)
            if not self.model.isDir(source_index):
                file_paths.append(path)
        
        if len(file_paths) < 2: return

        default_dir = os.path.dirname(file_paths[0])
        # Default to .avi (MJPG) which is much more reliable with OpenCV on macOS/Windows default installs
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Video", os.path.join(default_dir, "video.avi"), "AVI Files (*.avi);;MP4 Files (*.mp4)")
        if not save_path: return

        fps, ok = QInputDialog.getInt(self, "Video Settings", "Frames Per Second:", 10, 1, 120)
        if not ok: return

        try:
            import cv2
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except ImportError:
            QMessageBox.critical(self, "Error", "OpenCV (opencv-python) is required for this feature.")
            return

        progress = QProgressDialog("Creating Video...", "Cancel", 0, len(file_paths), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        
        writer = None
        target_size = None # (w, h)
        
        try:
            for i, path in enumerate(file_paths):
                if progress.wasCanceled(): break
                
                # Check extension roughly or just try load
                img = cv2.imread(path)
                if img is None: continue
                
                if writer is None:
                    h, w = img.shape[:2]
                    # Ensure dimensions are multiples of 2 for codecs like H.264
                    target_w = w if w % 2 == 0 else w - 1
                    target_h = h if h % 2 == 0 else h - 1
                    target_size = (target_w, target_h)

                    # FourCC
                    ext = os.path.splitext(save_path)[1].lower()
                    
                    codecs = []
                    # On many systems, especially MacOS with OpenCV, 'MJPG' is the most robust internal codec.
                    # It works for .avi and often for .mp4 containers.
                    codecs = ['MJPG']
                    
                    # If we are on MacOS and user wants mp4, we can try native avc1 if MJPG fails
                    if ext == '.mp4' and sys.platform == 'darwin':
                        codecs.append('avc1')
                    
                    # Final fallback
                    if ext == '.mp4':
                        codecs.append('mp4v')

                    import time
                    for codec in codecs:
                        try:
                            # AVAssetWriter cleanup
                            if os.path.exists(save_path):
                                os.remove(save_path)
                                time.sleep(0.1) 
                            
                            print(f"Trying codec: {codec}")
                            fourcc = cv2.VideoWriter_fourcc(*codec)
                            # FPS must be float for some backends
                            writer = cv2.VideoWriter(save_path, fourcc, float(fps), target_size)
                            if writer.isOpened():
                                print(f"Successfully opened video writer with {codec}")
                                break
                        except Exception as e:
                            print(f"Failed codec {codec}: {e}")
                            continue
                            
                    if not writer or not writer.isOpened():
                         raise Exception(f"Could not open video writer. Tried: {codecs}")
                            
                    if not writer or not writer.isOpened():
                         raise Exception(f"Could not open video writer using codecs: {codecs}")
                         
                # Resize if necessary
                h, w = img.shape[:2]
                if (w, h) != target_size:
                    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

                writer.write(img)
                progress.setValue(i+1)
                QApplication.processEvents() 

            if writer:
                writer.release()
                
            if not progress.wasCanceled():
                 QMessageBox.information(self, "Success", f"Video saved to {save_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create video: {e}")
        finally:
            progress.close()



class CustomGLViewWidget(gl.GLViewWidget):
    sig_double_click = pyqtSignal(object) # Emit pos
    def mouseDoubleClickEvent(self, event):
        self.sig_double_click.emit(event.pos())
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.RightButton:
            event.ignore()
            return
        super().mouseMoveEvent(event)

    def event(self, event):
        if event.type() == QEvent.Type.NativeGesture:
            if event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
                # Zoom In (+val) -> Decrease Distance
                # Zoom Out (-val) -> Increase Distance
                # Boost sensitivity
                factor = 1.0 - (event.value() * 5.0) 
                
                # Apply limit to prevent going negatives or too close
                self.opts['distance'] = max(1.0, self.opts['distance'] * factor)
                self.update()
                return True
        return super().event(event)

    def orbit(self, azim=0, elev=0):
        """
        Orbit the camera around the center point.
        Overridden to allow unrestricted elevation range (no clamping).
        """
        self.opts['azimuth'] += azim
        self.opts['elevation'] += elev
        self.update()


class PointCloudViewer(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D Point Cloud Viewer")
        self.resize(800, 600)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Header Controls
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(10, 5, 10, 5)
        
        self.reset_btn = QPushButton("Reset View")
        self.reset_btn.setToolTip("Reset Camera Position")
        self.reset_btn.clicked.connect(self.reset_view)
        header_layout.addWidget(self.reset_btn)
        
        header_layout.addStretch()
        
        # Meshing / Points Mode
        header_layout.addWidget(QLabel("Render:"))
        self.render_mode_combo = QComboBox()
        self.render_mode_combo.addItems(["Points", "Surface"])
        self.render_mode_combo.currentTextChanged.connect(self.toggle_render_mode)
        header_layout.addWidget(self.render_mode_combo)
        
        header_layout.addSpacing(10)

        self.colormap_mode = "viridis"
        header_layout.addWidget(QLabel("🎨 Color:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["Viridis", "Gray", "Albedo", "Normals"])
        self.colormap_combo.setCurrentText("Viridis")
        self.colormap_combo.setToolTip("Select Colormap")
        self.colormap_combo.currentTextChanged.connect(self.on_colormap_changed)
        header_layout.addWidget(self.colormap_combo)
        
        header_layout.addSpacing(10)
        
        # Lighting Sliders
        # Light X (Horizontal)
        header_layout.addWidget(QLabel("☀️↔️"))
        self.light_x_slider = QSlider(Qt.Orientation.Horizontal)
        self.light_x_slider.setRange(-100, 100)
        self.light_x_slider.setValue(0)
        self.light_x_slider.setFixedWidth(80)
        self.light_x_slider.setToolTip("Light Source Horizontal Position")
        header_layout.addWidget(self.light_x_slider)
        
        # Light Y (Vertical)
        header_layout.addWidget(QLabel("☀️↕️"))
        self.light_y_slider = QSlider(Qt.Orientation.Vertical)
        self.light_y_slider.setRange(-100, 100)
        self.light_y_slider.setValue(0) # Center (Directly above)
        self.light_y_slider.setFixedHeight(80)
        self.light_y_slider.setToolTip("Light Source Vertical Position")
        header_layout.addWidget(self.light_y_slider)
        
        # Z-Scale Slider (Depth Exaggeration)
        header_layout.addWidget(QLabel("🏔️"))
        self.z_scale_slider = QSlider(Qt.Orientation.Vertical)
        self.z_scale_slider.setRange(1, 300) # 1% to 300% of Width
        self.z_scale_slider.setValue(50) # Default to 50% width
        self.z_scale_slider.setFixedHeight(80)
        self.z_scale_slider.setToolTip("Vertical Exaggeration (Depth Scale)")
        header_layout.addWidget(self.z_scale_slider)
        
        self.layout.addLayout(header_layout)
        
        if gl is None:
            self.layout.addWidget(QLabel("Error: PyOpenGL or pyqtgraph.opengl not available."))
            return

        self.view_widget = CustomGLViewWidget()
        self.view_widget.opts['distance'] = 200
        self.layout.addWidget(self.view_widget)
        
        # Navigation Features
        self.view_widget.sig_double_click.connect(self.handle_double_click)
        self.view_widget.sig_double_click.connect(self.handle_double_click)
        
        # State
        self.last_data = None
        self.last_rgb_data = None
        self.scatter = None
        self.mesh = None
        self.base_colors = None
        self.normals = None
        self.pos = None
        self.mesh_faces = None
        self.mesh_vertices = None

    def toggle_render_mode(self):
        if self.last_data is not None:
             self.set_data(self.last_data, rgb_data=self.last_rgb_data)
             self.update_lighting()

    def set_data(self, data, reset_view=False, rgb_data=None):
        if gl is None: return
        
        if data is None: return
        self.last_data = data
        self.last_rgb_data = rgb_data
        processed_data = None
        q_image = None
        
        # Ensure single channel
        if data.ndim == 3:
            # Fallback to luminance
            data = np.mean(data, axis=2)
            
        h, w = data.shape
        
        # No Limit / Downsampling (User requested full resolution)
        sub_data = data
        sh, sw = sub_data.shape
        
        # Create grid normalized to standard range (max dimension = 100)
        VIEW_SCALE = 100.0
        max_dim = max(sh, sw)
        scale_factor = VIEW_SCALE / max_dim
        
        y = np.linspace(sh/2, -sh/2, sh) * scale_factor
        x = -np.linspace(-sw/2, sw/2, sw) * scale_factor
        xv, yv = np.meshgrid(x, y)
        
        # Flatten
        z_vals = sub_data.flatten()
        
        # Identify valid points (ignore 0, NaN, Inf)
        valid_mask = (z_vals != 0) & np.isfinite(z_vals)
        if not np.any(valid_mask):
             # Fallback if everything is filtered
             valid_mask = np.ones_like(z_vals, dtype=bool)
             
        # Filtered data for robust scaling calculations
        z_valid = z_vals[valid_mask]
        if z_valid.size == 0:
             z_valid = np.array([0.0])
        
        # Calculate percentiles for robust scaling (5% to 95%) using ONLY valid data
        with np.errstate(divide='ignore', invalid='ignore'):
            z_p5, z_p95 = np.nanpercentile(z_valid, [5, 95])
            
        # Ensure percentiles are finite
        if not np.isfinite(z_p5): z_p5 = 0.0
        if not np.isfinite(z_p95): z_p95 = 1.0
        
        z_range = z_p95 - z_p5
        if z_range <= 0: z_range = 1.0
        
        # 1. Compute Normals for Lighting (on FULL grid to maintain structure)
        # To make lighting visible even for flat data, we use a relative Z scale 
        # (e.g. scale Z so it has meaningful gradients relative to X/Y steps)
        # We use the percentile range to ensure scaling is robust to outliers.
        
        # FIX: Previous proportional scaling failed because Z units (meters) != XY units (pixels).
        # We decouple the raw range. The Slider now controls "Z Visual Range as % of View Width".
        # scale_factor scales max_dim (XY) to 100.
        # So we want Z to be (slider_value / 100.0) * 100.0 = slider_value.
        z_norm_range = float(self.z_scale_slider.value())
        
        # SANITIZE for np.gradient: fill inf/nan with z_p5 
        # Gradient on non-finite values triggers RuntimeWarnings
        clean_sub_data = sub_data.copy()
        clean_sub_data[~np.isfinite(clean_sub_data)] = z_p5
        
        with np.errstate(divide='ignore', invalid='ignore'):
             visual_z = (clean_sub_data - z_p5) / z_range * z_norm_range
             dzdy, dzdx = np.gradient(visual_z)
             
        # Normalize Z for actual point positions (use the same robust scaling)
        # We use nanmean subtract on the scaled version to keep it centered
        z_vals_scaled = (z_vals - z_p5) / z_range * z_norm_range
        z_vals_centered = z_vals_scaled - np.nanmean(z_vals_scaled[valid_mask])
        
        all_pos = np.vstack((xv.flatten(), yv.flatten(), z_vals_centered)).transpose()

        
        # For gradients, we need to adjust for the X/Y step size if we want true geometric normals
        # However, for visualization, using the normalized visual_z coordinates for gradients 
        # good relative depth effect regardless of resolution.
        # X-axis is flipped (Decreasing), so dz/dx is proportional to -dzdx (array gradient).
        # Normal X is -dz/dx. So Normal X is proportional to -(-dzdx) = +dzdx.
        nx = dzdx.flatten()
        # Flip Y-normal sign because our Y-axis is decreasing (Top to Bottom)
        ny = dzdy.flatten() 
        nz = np.ones_like(nx) * scale_factor
        
        mag = np.sqrt(nx**2 + ny**2 + nz**2)
        # Add epsilon to prevent division by zero/NaN normals
        mag[mag == 0] = 1e-8
        all_normals = np.stack([nx/mag, ny/mag, nz/mag], axis=1)
        
        # 2. Base Colors (on FULL grid)
        norm_z = np.clip((z_vals - z_p5) / z_range, 0, 1)
        
        if rgb_data is not None:
            import cv2
            # Ensure RGB shape matches data shape
            if rgb_data.shape[:2] != data.shape[:2]:
                rgb_resized = cv2.resize(rgb_data, (data.shape[1], data.shape[0]), interpolation=cv2.INTER_LINEAR)
            else:
                rgb_resized = rgb_data
            
            # Ensure it's RGB
            if rgb_resized.ndim == 2:
                rgb_resized = cv2.cvtColor(rgb_resized, cv2.COLOR_GRAY2RGB)
            elif rgb_resized.shape[2] == 4:
                rgb_resized = cv2.cvtColor(rgb_resized, cv2.COLOR_RGBA2RGB)
                
            # Map to 0-1 and add alpha channel
            rgb_flat = rgb_resized.reshape(-1, 3) / 255.0
            all_colors = np.hstack([rgb_flat, np.ones((len(rgb_flat), 1))])
            
        elif self.colormap_mode == "gray":
            all_colors = np.stack([norm_z, norm_z, norm_z, np.ones_like(norm_z)], axis=1)
        elif self.colormap_mode == "albedo":
            all_colors = np.array([[0.6, 0.6, 0.6, 1.0]] * len(norm_z))
        elif self.colormap_mode == "normals":
            # Map normals [-1, 1] to [0, 1] for RGB
            # all_normals is already shape (N, 3)
            rgb_normals = (all_normals + 1.0) / 2.0
            all_colors = np.hstack([rgb_normals, np.ones((len(rgb_normals), 1))])
        else:
            all_colors = cm.viridis(norm_z)
            
        # 3. Apply Validity Mask to filter out 0, NaN, Inf
        
        self.normals_all_grid = all_normals
        
        # Apply Validity Mask to filter out 0, NaN, Inf for Points
        self.pos = all_pos[valid_mask]
        self.normals = all_normals[valid_mask]
        self.base_colors = all_colors[valid_mask]

        # 4. Generate Mesh (If applicable)
        # We pre-calculate faces here in case we need them to render
        render_mode = self.render_mode_combo.currentText()
        if render_mode == "Surface":
             # To create a mesh grid, we need vertex indices
             idx_grid = np.arange(sh * sw).reshape(sh, sw)
             
             # Create two triangles per quad
             #  t1: (r, c), (r+1, c), (r, c+1)
             #  t2: (r+1, c), (r+1, c+1), (r, c+1)
             
             v1 = idx_grid[:-1, :-1].flatten()
             v2 = idx_grid[1:, :-1].flatten()
             v3 = idx_grid[:-1, 1:].flatten()
             v4 = idx_grid[1:, :-1].flatten()
             v5 = idx_grid[1:, 1:].flatten()
             v6 = idx_grid[:-1, 1:].flatten()
             
             # Stack into (NumTriangles, 3) matrix
             faces_t1 = np.vstack([v1, v2, v3]).T
             faces_t2 = np.vstack([v4, v5, v6]).T
             
             # Filter faces that might be crossing invalid (z_val == 0) pixels
             v_mask_flat = valid_mask.flatten()
             
             t1_valid = v_mask_flat[faces_t1[:,0]] & v_mask_flat[faces_t1[:,1]] & v_mask_flat[faces_t1[:,2]]
             t2_valid = v_mask_flat[faces_t2[:,0]] & v_mask_flat[faces_t2[:,1]] & v_mask_flat[faces_t2[:,2]]
             
             valid_faces_t1 = faces_t1[t1_valid]
             valid_faces_t2 = faces_t2[t2_valid]
             
             self.mesh_faces = np.vstack([valid_faces_t1, valid_faces_t2])
             self.mesh_vertices = all_pos
             self.base_colors_mesh = all_colors
             
             # Clear points so we don't render both
             self.pos = np.array([])
             self.base_colors = np.array([])
             self.normals = np.array([])
            
        if self.scatter:
            self.view_widget.removeItem(self.scatter)
            self.scatter = None
        if self.mesh:
            self.view_widget.removeItem(self.mesh)
            self.mesh = None
            
        # Initial draw with lighting
        self.update_lighting()
        
        # Reset Camera ONLY IF requested (e.g. first time or manual reset)
        if reset_view:
            self.reset_view()

    def update_lighting(self):
        """Update point colors based on light direction."""
        if self.base_colors is None or self.normals is None:
            return
            
        # Cartesian Lighting: Project 2D slider pos onto 3D hemisphere
        # Invert both directions as requested
        lx = self.light_x_slider.value() / 100.0
        ly = -self.light_y_slider.value() / 100.0
        
        # Calculate Z component (height of light above surface)
        # sphere equation: x^2 + y^2 + z^2 = 1 => z = sqrt(1 - x^2 - y^2)
        r2 = lx*lx + ly*ly
        if r2 > 1.0:
            # Normalize to unit circle edge if outside
            mag = np.sqrt(r2)
            lx /= mag
            ly /= mag
            lz = 0.0 # Horizon
        else:
            lz = np.sqrt(1.0 - r2)
            
        light_vec = np.array([lx, ly, lz])
        
        # 1. Update Points (if applicable)
        if self.normals is not None and len(self.normals) > 0:
            # Dot product for diffuse lighting
            # Use simple dot (not absolute) to create realistic shadows on the far side
            dot = np.sum(self.normals * light_vec, axis=1)
            
            # If in normals mode, shading might be distracting, but let's keep it subtle
            shading_strength = 0.5 if self.colormap_mode == "normals" else 0.8
            ambient = 0.5 if self.colormap_mode == "normals" else 0.2
            
            # Improve contrast: allow shading to go darker
            shading = np.clip(dot, 0, 1) * shading_strength + ambient
            
            if self.base_colors is not None and len(self.base_colors) > 0:
                shaded_points = self.base_colors.copy()
                # If length doesn't match, or scatter isn't created, we handle below
                if len(shading) == len(shaded_points):
                     shaded_points[:, :3] *= shading[:, np.newaxis]
                
                if self.scatter:
                    self.scatter.setData(color=shaded_points)
                else:
                    self.scatter = gl.GLScatterPlotItem(pos=self.pos, color=shaded_points, size=2, pxMode=True, glOptions='opaque')
                    self.view_widget.addItem(self.scatter)

        # Mesh / Surface
        render_mode = self.render_mode_combo.currentText()
        if render_mode == "Surface" and self.mesh_faces is not None:
             # Apply lighting to mesh vertices directly
             # Since it's a solid surface, simple interpolation works well
             verts = self.mesh_vertices
             colors = self.base_colors_mesh.copy()
             
             # Calculate light on all vertices
             nx = self.normals_all_grid[:, 0] if hasattr(self, 'normals_all_grid') else np.ones(len(verts))
             ny = self.normals_all_grid[:, 1] if hasattr(self, 'normals_all_grid') else np.zeros(len(verts))
             nz = self.normals_all_grid[:, 2] if hasattr(self, 'normals_all_grid') else np.ones(len(verts))
             
             dp = nx * lx + ny * ly + nz * lz
             shd = (dp + 1.0) / 2.0
             shd = 0.3 + 0.7 * shd
             colors[:, :3] *= shd[:, np.newaxis]
             
             if self.mesh:
                  self.mesh.setMeshData(vertexes=verts, faces=self.mesh_faces, vertexColors=colors)
             else:
                  self.mesh = gl.GLMeshItem(vertexes=verts, faces=self.mesh_faces, vertexColors=colors, smooth=False, computeNormals=False, glOptions='opaque')
                  self.view_widget.addItem(self.mesh)

    def reset_view(self):
        if gl is None: return
        # Reset to "Image POV": Top-down view looking at the XY plane (-90 az rotates "up" correctly usually)
        self.view_widget.setCameraPosition(distance=max(self.view_widget.opts['distance'], 200), elevation=-90, azimuth=90)
        self.view_widget.setCameraPosition(distance=max(self.view_widget.opts['distance'], 200), elevation=-90, azimuth=90)
        
        # Reset Lighting
        self.light_x_slider.setValue(0)
        self.light_y_slider.setValue(0)
        
        # Reset Depth Scale
        self.z_scale_slider.setValue(50)
        
        
        # Ideally we compute optimal distance based on bounds

    def on_colormap_changed(self, text):
        """Change Colormap based on dropdown selection."""
        mapping = {
            "Viridis": "viridis",
            "Gray": "gray",
            "Albedo": "albedo",
            "Normals": "normals"
        }
        self.colormap_mode = mapping.get(text, "viridis")
        
        if self.last_data is not None:
            self.set_data(self.last_data, rgb_data=self.last_rgb_data)
            self.update_lighting()

    def resizeEvent(self, event):
        super().resizeEvent(event)
            
    def handle_double_click(self, pos):
        """Center the view on the clicked point."""
        if self.pos is None or len(self.pos) == 0:
            return
            
        # Get View Matrices
        # We try to use the view_widget's internal matrices if possible, or approximate projection
        w = self.view_widget.width()
        h = self.view_widget.height()
        
        # Normalized Device Coordinates (NDC) of click
        # QPoint origin is top-left. OpenGL NDC is center, Y up.
        nx = 2.0 * pos.x() / w - 1.0
        ny = 1.0 - 2.0 * pos.y() / h
        
        try:
             # Getting matrices from the widget's internal state
             pMatrix = self.view_widget.projectionMatrix(region=(0, 0, w, h)) 
             vMatrix = self.view_widget.viewMatrix()
             
             if pMatrix is None or vMatrix is None:
                 print("Error: Matrices are None")
                 return

             mvp = pMatrix * vMatrix
             
             # Project all points? 50k points is fast in numpy.
             # self.pos is (N, 3)
             
             if self.pos is None:
                  print("Error: self.pos is None in double click")
                  return
             
             # Extract matrix elements directly via row/col lists in PyQt6
             m_data = []
             for i in range(4):
                 col = mvp.column(i)
                 m_data.append([col.x(), col.y(), col.z(), col.w()])
             
             mvp_data = np.array(m_data).T  # Transpose needed depending on column/row major layout expectations, PyQt6 .column() gives elements of column vectors
             
             # Homogeneous coordinates
             points_4d = np.hstack([self.pos, np.ones((len(self.pos), 1))])
             
             # Transform
             clip_coords = points_4d @ mvp_data
             
             # Perspective divide
             w_coords = clip_coords[:, 3]
             w_coords[w_coords == 0] = 1e-6
             
             ndc_x = clip_coords[:, 0] / w_coords
             ndc_y = clip_coords[:, 1] / w_coords
             # Z depth can be used for occlusion checking if needed
             
             # Check distance to click (nx, ny)
             dist_sq = (ndc_x - nx)**2 + (ndc_y - ny)**2
             
             # Find minimum distance
             if len(dist_sq) > 0:
                 min_idx = np.argmin(dist_sq)
                 min_dist = dist_sq[min_idx]
                 
                 # Threshold (e.g. within ~5% of screen size)
                 if min_dist < 0.05:
                     target_point = self.pos[min_idx]
                     
                     # Center view on this point
                     self.view_widget.opts['center'] = pg.Vector(*target_point)
                     self.view_widget.update()
                 
        except Exception as e:
            import traceback
            print(f"Error handling double click:")
            traceback.print_exc()
