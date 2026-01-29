import time

import numpy as np
import re
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QPointF, QEvent, QObject, QTimer, QRectF, QSettings, QSortFilterProxyModel, QDir, QRegularExpression, QItemSelectionModel, QSize
from PyQt6.QtGui import QPixmap, QPainter, QNativeGestureEvent, QDoubleValidator, QKeyEvent, QImage, QMouseEvent, QColor, QIcon, QFileSystemModel
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
        # Debounce the expensive image update
        self._region_update_timer.start()

    def _emit_delayed_region_changed(self):
        min_val, max_val = self.region.getRegion()
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
        super().__init__("Math", parent)
        self.setMinimumWidth(100)
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        self.content_widget = QWidget()
        self.content_widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.layout = QVBoxLayout(self.content_widget)

        self.expression_input = QComboBox()
        self.expression_input.setEditable(True)
        self.expression_input.lineEdit().setPlaceholderText("Enter math expression (e.g., x+1, np.log(x))")
        self.expression_input.setMinimumWidth(50)
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
    # Signal now emits on change
    settings_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__("Info", parent)
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.setMinimumWidth(100)

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
        self.color_format_combo.addItems(["Grayscale", "Bayer GRBG", "Bayer RGGB", "Bayer BGGR", "Bayer GBRG", 
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

    def update_info(self, width, height, dtype, dtype_map, file_size=0):
        self.file_size = file_size
        
        # Store defaults
        self.defaults = {
            'width': width,
            'height': height,
            'dtype': str(dtype) if not isinstance(dtype, type) else np.dtype(dtype).name,
            'color_format': self.color_format_combo.currentText() # Preserve current format or reset to Gray? 
            # Usually update_info is called with detected params. 
            # If detected was just "Raw", format is default Grayscale, so getting currentText is risky if user changed it?
            # Ideally update_info should take format. But image_handler only guesses Grayscale usually.
            # Let's assume Grayscale default for new loads.
        }
        # If we want to allow override_settings loopback, we might need to handle format better.
        # For now, default to 'Grayscale' in defaults if not passed, BUT wait,
        # update_info is called from open_file. open_file doesn't pass format.
        self.defaults['color_format'] = "Grayscale"

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
        self.drag_start_position = QPoint()
        self.processed_data = None # Store processed numpy array instead of QPixmap

        self.overlays = [] # List of (QPixmap, opacity)
        self.original_data = None # Raw data before processing
        self.pristine_data = None # Original loaded data (preserved across transforms)
        self.thumbnail_pixmap = None # Cached thumbnail for optimized rendering
        self.contrast_limits = None
        self.colormap = 'gray'

        self.zoom_speed = 1.1
        self.zoom_in_interp = Qt.TransformationMode.FastTransformation
        self.zoom_out_interp = Qt.TransformationMode.SmoothTransformation
        self._pinch_start_scale_factor = None

        # Active Indicator Line
        self.indicator_line = QFrame(self)
        self.indicator_line.setFixedHeight(3)
        self.indicator_line.setStyleSheet("background-color: transparent;")

        # Overlay Label
        self.overlay_label = QLabel(self)
        self.overlay_label.setStyleSheet("background-color: rgba(0, 0, 0, 150); color: white; padding: 5px; border-radius: 5px; font-size: 14px;")
        self.overlay_label.hide()
        self.overlay_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
        
        self.overlays = [] # List of tuples: (QPixmap, opacity)

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

    def _on_shared_crosshair_changed(self):
        self.update()

    def set_active(self, active):
        if self.is_active != active:
            self.is_active = active
            self.update()
            
    def update_transform(self):
        """Force a repaint/update of the transformation."""
        self.update()

    def is_single_channel(self):
        """Check if current image is single channel."""
        if self.original_data is not None:
            # Check shape: (H, W) or (H, W, 1) is single channel
            if self.original_data.ndim == 2:
                return True
            if self.original_data.ndim == 3 and self.original_data.shape[2] == 1:
                return True
        return False
        
    def get_visible_sub_image(self):
        """Return the currently visible portion of the image data."""
        # TODO: Implement true cropping based on view. 
        # For now, returning full image prevents crash and provides data.
        return self.original_data

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
        if self.original_data is None:
            return

        data = self.original_data
        
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
            
            # If data is already RGB (legacy or handled external), just display it
            elif data.ndim == 3 and data.shape[2] >= 3:
                processed_data = data[:, :, :3].astype(np.uint8) if data.dtype != np.uint8 else data[:, :, :3]
                h, w, _ = processed_data.shape
                q_image = QImage(processed_data.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
            else:
                return  # Invalid state for flow mode
        else:
            # Original logic for other colormaps
            # Determine mode: standard RGB or Colormapped (single channel)
            # If RGB and map is 'gray', show as RGB.
            # If RGB and map is NOT 'gray', extract Ch0 and map it.
            is_rgb = (data.ndim == 3 and data.shape[2] == 3)
            treat_as_rgb = is_rgb and (self.colormap == 'gray')

            if treat_as_rgb:  # Color Image
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

            else:  # Grayscale / Colormapped
                if is_rgb:
                     data = data[:, :, 0] # Use Channel 0 for colormapping
                elif data.ndim == 3 and data.shape[2] == 2:
                     # 2-Channel non-flow -> Use Magnitude
                     data = np.linalg.norm(data, axis=2)

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
            # End of else block for non-flow colormaps

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
            self.fit_to_view()
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
        if hasattr(self, 'indicator_line'):
             self.indicator_line.setGeometry(0, self.height() - 3, self.width(), 3)

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
        
        # Debounce the expensive view changed signal (histogram update)
        self._view_update_timer.start()

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
            self.drag_start_position = event.pos()
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
    
    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if os.path.exists(file_path):
                # Try to find load_image in the window hierarchy
                window = self.window()
                if hasattr(window, 'load_image'):
                    window.load_image(file_path)

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
        self.main_layout = QVBoxLayout(self.main_widget)
        
        # Add Select All checkbox
        self.select_all_cb = QCheckBox("Select All")
        self.select_all_cb.setTristate(True)  # Allow indeterminate state
        self.select_all_cb.stateChanged.connect(self._on_select_all_changed)
        self.main_layout.addWidget(self.select_all_cb)

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
        
        # Persistent gallery: {file_path: pixmap}
        self.gallery_images = {}

    def showEvent(self, event):
        """Refresh thumbnail pane contents when shown"""
        super().showEvent(event)
        # Trigger a refresh from the parent ImageViewer
        if self.parent() and hasattr(self.parent(), '_refresh_thumbnail_pane'):
            self.parent()._refresh_thumbnail_pane()

    def eventFilter(self, source, event):
        if source == self.scroll_area and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key in [Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Space]:
                self.keyPressEvent(event)
                return True
        return super().eventFilter(source, event)

    def keyPressEvent(self, event: QKeyEvent):
        if not self.thumbnail_items:
            super().keyPressEvent(event)
            return

        key = event.key()
        current_index = self.focused_index if self.focused_index != -1 else 0

        # Only handle specific navigation and selection keys
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
                     # Just move focus, unless we decide to follow selection?
                     # Standard behavior: Arrows move focus, Space selects.
                     # BUT user asked for "Shift+Arrows".
                     # Usually Arrows without shift simply move focus in file browsers (requiring space to select),
                     # OR they Select Single.
                     # "I want it to work with ... shift+arrows".
                     # Let's make arrows Select Single if no modifier?
                     # That leads to fast browsing.
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
        else:
            # Pass all other keys to parent (ImageViewer) for global shortcuts
            super().keyPressEvent(event)

    def populate(self, windows=None):
        """Populate thumbnail pane from persistent gallery"""
        # Don't populate if blocked (during selection-driven view changes)
        if hasattr(self, 'block_populate') and self.block_populate:
            return
        
        # 1. Update gallery with images from all windows
        for widget in QApplication.topLevelWidgets():
            if widget.__class__.__name__ == 'ImageViewer':
                if hasattr(widget, 'stacked_widget') and hasattr(widget, 'montage_widget'):
                    if widget.stacked_widget.currentWidget() == widget.montage_widget and widget.montage_labels:
                        # Montage mode: add all montage images to gallery
                        for label in widget.montage_labels:
                            if hasattr(label, 'file_path') and label.current_pixmap and label.file_path:
                                self.gallery_images[label.file_path] = label.current_pixmap
                    elif widget.image_label.current_pixmap and widget.current_file_path:
                        # Single image mode: add to gallery
                        # Check if this is an NPZ file with multiple keys
                        if hasattr(widget.image_handler, 'npz_keys') and len(widget.image_handler.npz_keys) > 1:
                            # Add thumbnails for each valid NPZ key
                            for key in widget.image_handler.npz_keys.keys():
                                if widget.image_handler.npz_keys[key]:  # Only valid keys
                                    # Create a unique identifier for this NPZ key
                                    npz_key_path = f"{widget.current_file_path}#{key}"
                                    # Get or generate pixmap for this key
                                    if key == widget.image_handler.current_npz_key:
                                        # Current key - use existing pixmap
                                        self.gallery_images[npz_key_path] = widget.image_label.current_pixmap
                                    else:
                                        # Other keys - generate thumbnail
                                        pixmap = self._generate_npz_key_pixmap(widget.image_handler.npz_data[key])
                                        if pixmap:
                                            self.gallery_images[npz_key_path] = pixmap
                        else:
                            # Regular file
                            self.gallery_images[widget.current_file_path] = widget.image_label.current_pixmap
        
        # 2. Capture current selection
        selected_paths = {item.file_path for item in self.thumbnail_items if item.is_selected}
        
        # 3. Clear all existing thumbnail widgets
        for item in self.thumbnail_items:
            self.thumbnail_layout.removeWidget(item)
            item.deleteLater()
        self.thumbnail_items.clear()

        # 4. Create thumbnail items from gallery
        from widgets import ThumbnailItem
        for file_path, pixmap in self.gallery_images.items():
            item = ThumbnailItem(file_path, pixmap)
            item.clicked.connect(lambda event, i=item: self._on_thumbnail_clicked(i, event))
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
    
    def _generate_npz_key_pixmap(self, array_data):
        """Generate a QPixmap for an NPZ key's array data."""
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
        self._select_first_item()

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
        self._select_first_item()

    def _on_directory_loaded(self, path):
         if os.path.normpath(path) == os.path.normpath(self.root_path):
             self._select_first_item()

    def _select_first_item(self):
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

        # Fallback to first item
        if self.proxy_model.rowCount(root_index) > 0:
            first_index = self.proxy_model.index(0, 0, root_index)
            if first_index.isValid():
                self.tree_view.setCurrentIndex(first_index)
                self.tree_view.selectionModel().select(first_index, QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows)
                self.tree_view.scrollTo(first_index)
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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents)

    def mouseMoveEvent(self, event):
        # Disable Right Button Drag (PyQtGraph uses this for Zoom)
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


class PointCloudViewer(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D Point Cloud Viewer")
        self.resize(800, 600)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        if gl is None:
            self.layout.addWidget(QLabel("Error: PyOpenGL or pyqtgraph.opengl not available."))
            return

        self.view_widget = CustomGLViewWidget()
        self.view_widget.opts['distance'] = 200
        self.layout.addWidget(self.view_widget)
        
        # Controls Overlay
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(10, 10, 10, 10)
        
        self.reset_btn = QPushButton("Reset View")
        self.reset_btn.clicked.connect(self.reset_view)
        controls_layout.addWidget(self.reset_btn)
        
        controls_layout.addStretch()
        
        # We wrap controls in a widget to overlay or just put at bottom? 
        # Bottom is easier.
        self.layout.addLayout(controls_layout)
        
        self.scatter = None

    def set_data(self, data):
        if gl is None: return
        
        if data is None: return
        
        # Ensure single channel
        if data.ndim == 3:
            # Fallback to luminance
            data = np.mean(data, axis=2)
            
        h, w = data.shape
        
        # Downsample for performance (target ~100k points max usually good for Python/GL)
        target_points = 50000
        total_points = h * w
        step = 1
        if total_points > target_points:
            step = int(np.sqrt(total_points / target_points))
            
        sub_data = data[::step, ::step]
        sh, sw = sub_data.shape
        
        # Create grid
        # Center the grid
        y = np.linspace(-sh/2, sh/2, sh)
        x = np.linspace(-sw/2, sw/2, sw)
        xv, yv = np.meshgrid(x, y)
        
        # Flatten
        z_vals = sub_data.flatten()
        
        # Normalize Z for better view scaling (e.g. proportional to X/Y?)
        # User said "values interpreted as depth map". 
        # Usually depth values are in same units as X/Y or arbitrary.
        # Let's keep raw values but maybe center them?
        z_vals = z_vals - np.mean(z_vals)
        
        # Scale Z to be visible? If Data is 0..1 (float) and W is 1000, flat.
        # If Data is 0..255 and W is 1000, ok.
        # Let's verify range.
        z_min, z_max = z_vals.min(), z_vals.max()
        z_range = z_max - z_min
        if z_range == 0: z_range = 1
        
        # Optional: Auto-scale Z to match aspect ratio of Width? 
        # For now, keep raw values to preserve physical meaning if any.
        
        pos = np.vstack((xv.flatten(), yv.flatten(), z_vals)).transpose()
        
        # Color mapping
        # Normalize for colormap
        norm_z = (z_vals - z_min) / z_range
        # Use matplotlib colormap (viridis)
        # viridis returns RGBA floats 0..1
        colors = cm.viridis(norm_z) 
        
        if self.scatter:
            self.view_widget.removeItem(self.scatter)
            self.scatter = None
            
        self.scatter = gl.GLScatterPlotItem(pos=pos, color=colors, size=2, pxMode=True)
        self.view_widget.addItem(self.scatter)
        
        # Reset Camera to fit
        self.reset_view()

    def reset_view(self):
        if gl is None: return
        # Simple reset
        self.view_widget.setCameraPosition(distance=max(self.view_widget.opts['distance'], 200), elevation=45, azimuth=45)
        # Ideally we compute optimal distance based on bounds
