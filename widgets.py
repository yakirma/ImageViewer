import numpy as np
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QPointF, QEvent, QObject
from PyQt6.QtGui import QPixmap, QPainter, QNativeGestureEvent, QDoubleValidator, QKeyEvent, QImage, QMouseEvent
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
)
import pyqtgraph as pg
import os
import matplotlib.cm as cm


class SharedViewState(QObject):
    """An object to hold and synchronize view parameters for multiple labels."""
    state_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._scale_factor = 1.0
        self._offset = QPointF()
        self._crosshair_pos = None

    @property
    def scale_factor(self):
        return self._scale_factor

    @scale_factor.setter
    def scale_factor(self, value):
        if self._scale_factor != value:
            self._scale_factor = value
            self.state_changed.emit()

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value):
        if self._offset != value:
            self._offset = value
            self.state_changed.emit()

    @property
    def crosshair_pos(self):
        return self._crosshair_pos

    @crosshair_pos.setter
    def crosshair_pos(self, value):
        if self._crosshair_pos != value:
            self._crosshair_pos = value
            self.state_changed.emit()

    def restore(self, fit_scale):
        self._scale_factor = fit_scale
        self._offset = QPointF()
        self.state_changed.emit()


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
            self.region.setBounds([min_val, max_val])
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

    def __init__(self, parent=None):
        super().__init__("Math Transform", parent)
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        self.content_widget = QWidget()
        self.layout = QVBoxLayout(self.content_widget)

        self.expression_input = QLineEdit()
        self.expression_input.setPlaceholderText("Enter math expression (e.g., x+1, np.log(x))")
        self.expression_input.returnPressed.connect(self._on_apply)
        self.layout.addWidget(self.expression_input)

        self.apply_button = QPushButton("Apply Transform")
        self.apply_button.clicked.connect(self._on_apply)
        self.layout.addWidget(self.apply_button)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: red;")
        self.layout.addWidget(self.error_label)

        self.setWidget(self.content_widget)

    def _on_apply(self):
        expression = self.expression_input.text()
        if expression:
            self.error_label.clear()
            self.transform_requested.emit(expression)
        else:
            self.error_label.setText("Expression cannot be empty.")

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


class ZoomableDraggableLabel(QLabel):
    hover_moved = pyqtSignal(int, int)
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
            self.shared_state.state_changed.connect(self._on_shared_state_changed)
        else:
            self._scale_factor = 1.0
            self._pixmap_offset = QPointF()
            self._crosshair_pos = None

        self.is_active = False
        self.crosshair_enabled = False
        self.drag_start_position = QPoint()
        self.current_pixmap = None
        self.cached_scaled_pixmap = None

        self.original_data = None
        self.contrast_limits = None
        self.colormap = 'gray'

        self.zoom_speed = 1.1
        self.zoom_in_interp = Qt.TransformationMode.SmoothTransformation
        self.zoom_out_interp = Qt.TransformationMode.SmoothTransformation
        self._pinch_start_scale_factor = None

    def _get_scale_factor(self):
        return self.shared_state.scale_factor if self.shared_state else self._scale_factor

    def _set_scale_factor(self, value):
        if self.shared_state:
            self.shared_state.scale_factor = value
        else:
            self._scale_factor = value

    def _get_offset(self):
        return self.shared_state.offset if self.shared_state else self._pixmap_offset

    def _set_offset(self, value):
        if self.shared_state:
            self.shared_state.offset = value
        else:
            self._pixmap_offset = value

    def _get_crosshair_pos(self):
        return self.shared_state.crosshair_pos if self.shared_state else self._crosshair_pos

    def _set_crosshair_pos(self, value):
        if self.shared_state:
            self.shared_state.crosshair_pos = value
        else:
            self._crosshair_pos = value
            self.update()  # Force repaint for single view

    def _on_shared_state_changed(self):
        self._update_scaled_pixmap()
        self.update()

    def set_active(self, active):
        if self.is_active != active:
            self.is_active = active
            self.update()

    def set_crosshair_enabled(self, enabled):
        self.crosshair_enabled = enabled
        self.setMouseTracking(enabled)
        if not enabled:
            self._set_crosshair_pos(None)
        self.update()

    def set_data(self, data):
        self.original_data = data
        self.contrast_limits = None
        self.apply_colormap(is_new_image=True)

    def is_single_channel(self):
        return self.original_data is not None and self.original_data.ndim == 2

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
            h, w, _ = image_data_8bit.shape
            q_image = QImage(image_data_8bit.data, w, h, 3 * w, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)
        if is_new_image:
            self.current_pixmap = pixmap
            self.fit_to_view()
        else:
            self.update_pixmap_content(pixmap)

    def update_pixmap_content(self, pixmap):
        self.current_pixmap = pixmap
        self._update_scaled_pixmap()
        self.update()

    def fit_to_view(self):
        if self.current_pixmap:
            label_size = self.size()
            pixmap_size = self.current_pixmap.size()
            if pixmap_size.width() > 0 and pixmap_size.height() > 0:
                scale_w = label_size.width() / pixmap_size.width()
                scale_h = label_size.height() / pixmap_size.height()
                fit_scale = min(scale_w, scale_h)
                if self.shared_state:
                    self.shared_state.restore(fit_scale)
                else:
                    self._set_scale_factor(min(1.0, fit_scale))
                    self._set_offset(QPointF())
            else:
                self._set_scale_factor(1.0)
        else:
            self._set_scale_factor(1.0)

        self._update_scaled_pixmap()
        self.update()
        self.zoom_factor_changed.emit(self._get_scale_factor())
        self.view_changed.emit()

    def restore_view(self):
        self.fit_to_view()

    def _update_scaled_pixmap(self):
        if self.current_pixmap is None:
            self.cached_scaled_pixmap = None
            return

        interp_mode = self.zoom_in_interp if self._get_scale_factor() > 1.0 else self.zoom_out_interp
        self.cached_scaled_pixmap = self.current_pixmap.scaled(
            self.current_pixmap.size() * self._get_scale_factor(),
            Qt.AspectRatioMode.KeepAspectRatio,
            interp_mode
        )

    def _apply_zoom(self, new_scale_factor, mouse_pos=None):
        if self.current_pixmap is None:
            return
        old_scale_factor = self._get_scale_factor()
        new_scale_factor = max(0.01, min(100.0, new_scale_factor))

        if abs(new_scale_factor - old_scale_factor) < 1e-9:
            return

        zoom_ratio = new_scale_factor / old_scale_factor
        if mouse_pos:
            mouse_rel_center = mouse_pos - QPointF(self.rect().center())
            new_offset = mouse_rel_center - (mouse_rel_center - self._get_offset()) * zoom_ratio
            self._set_offset(new_offset)
        else:
            self._set_offset(self._get_offset() * zoom_ratio)

        self._set_scale_factor(new_scale_factor)

        if not self.shared_state:
            self._update_scaled_pixmap()
            self.update()

        self.zoom_factor_changed.emit(self._get_scale_factor())
        self.view_changed.emit()

    def wheelEvent(self, event):
        new_scale_factor = self._get_scale_factor()
        if event.angleDelta().y() > 0:
            new_scale_factor *= self.zoom_speed
        else:
            new_scale_factor /= self.zoom_speed
        self._apply_zoom(new_scale_factor, event.position())

    def event(self, event):
        if event.type() == QEvent.Type.Gesture:
            pinch = event.gesture(Qt.GestureType.PinchGesture)
            if pinch:
                if pinch.state() == Qt.GestureState.GestureStarted:
                    self._pinch_start_scale_factor = self._get_scale_factor()
                elif pinch.state() == Qt.GestureState.GestureUpdated:
                    new_scale_factor = self._pinch_start_scale_factor * pinch.totalScaleFactor()
                    self._apply_zoom(new_scale_factor, pinch.centerPoint())
                elif pinch.state() == Qt.GestureState.GestureFinished:
                    self._pinch_start_scale_factor = None
                return True
        return super().event(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.current_pixmap:
            self.drag_start_position = event.pos()
            self.clicked.emit()
            self.setFocus()

    def mouseMoveEvent(self, event):
        if not self.drag_start_position.isNull() and self.current_pixmap:
            delta = event.pos() - self.drag_start_position
            self._set_offset(self._get_offset() + QPointF(delta))
            self.drag_start_position = event.pos()
            if not self.shared_state:
                self.update()
            self.view_changed.emit()

        if self.original_data is not None:
            # Calculate image coordinates from widget coordinates
            if self.cached_scaled_pixmap:
                target_rect = self.cached_scaled_pixmap.rect()
                target_rect.moveCenter(self.rect().center() + self._get_offset().toPoint())

                if target_rect.contains(event.pos()):
                    pixmap_coords = event.pos() - target_rect.topLeft()
                    img_height, img_width = self.original_data.shape[0], self.original_data.shape[1]

                    if target_rect.width() > 0 and target_rect.height() > 0:
                        x = int(pixmap_coords.x() * (img_width / target_rect.width()))
                        y = int(pixmap_coords.y() * (img_height / target_rect.height()))

                        if 0 <= x < img_width and 0 <= y < img_height:
                            if self.crosshair_enabled:
                                self._set_crosshair_pos(QPoint(x, y))
                            self.hover_moved.emit(x, y)
                        else:
                            self.hover_left.emit()
                else:
                    self.hover_left.emit()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start_position = QPoint()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.cached_scaled_pixmap is None:
            return
        painter = QPainter(self)
        target_rect = self.cached_scaled_pixmap.rect()
        target_rect.moveCenter(self.rect().center() + self._get_offset().toPoint())
        painter.drawPixmap(target_rect, self.cached_scaled_pixmap)

        if self.crosshair_enabled and self._get_crosshair_pos():
            pos = self._get_crosshair_pos()
            # Transform image coords to widget coords
            scale = self._get_scale_factor()
            offset = self._get_offset()

            view_x = (pos.x() * scale) + target_rect.left()
            view_y = (pos.y() * scale) + target_rect.top()

            pen = painter.pen()
            pen.setColor(Qt.GlobalColor.green)
            pen.setWidth(1)
            painter.setPen(pen)

            painter.drawLine(int(view_x), self.rect().top(), int(view_x), self.rect().bottom())
            painter.drawLine(self.rect().left(), int(view_y), self.rect().right(), int(view_y))

        if self.is_active:
            pen = painter.pen()
            pen.setColor(Qt.GlobalColor.green)
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawRect(self.rect().adjusted(1, 1, -1, -1))


class ThumbnailItem(QWidget):
    clicked = pyqtSignal(QEvent)

    def __init__(self, file_path, pixmap, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.is_selected = False
        self.is_focused = False

        self.setFixedSize(150, 150)
        layout = QVBoxLayout(self)
        self.image_label = QLabel()
        self.image_label.setPixmap(
            pixmap.scaled(128, 128, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.filename_label = QLabel(os.path.basename(file_path))
        self.filename_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.filename_label.setWordWrap(True)

        layout.addWidget(self.image_label)
        layout.addWidget(self.filename_label)
        self.setLayout(layout)

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

    def __init__(self, parent=None):
        super().__init__("Thumbnails", parent)
        self.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.setFloating(False)

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