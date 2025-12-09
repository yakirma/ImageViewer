import numpy as np
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QPointF, QEvent
from PyQt6.QtGui import QPixmap, QPainter, QNativeGestureEvent, QDoubleValidator
from PyQt6.QtWidgets import (
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
)
import pyqtgraph as pg


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
                y, x = np.histogram(channel_data, bins=256)
                hist = self.plot_widget.plot(x, y, stepMode="center", fillLevel=0, brush=colors[i])
                self.histograms.append(hist)
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
            # No need to emit here, setRegion triggers _on_region_changed

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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.grabGesture(Qt.GestureType.PinchGesture)

        self.scale_factor = 1.0
        self.drag_start_position = QPoint()
        self.pixmap_offset = QPointF()
        self.current_pixmap = None
        self.cached_scaled_pixmap = None
        self.image_data_for_hover = None
        self.zoom_speed = 1.1
        self.zoom_in_interp = Qt.TransformationMode.SmoothTransformation
        self.zoom_out_interp = Qt.TransformationMode.SmoothTransformation
        self._pinch_start_scale_factor = None

    def set_pixmap(self, pixmap, image_data_for_hover):
        """Used only when a new image is loaded."""
        self.current_pixmap = pixmap
        self.image_data_for_hover = image_data_for_hover
        self.fit_to_view()

    def update_pixmap_content(self, pixmap):
        """Used to update the visual content without changing zoom/pan."""
        self.current_pixmap = pixmap
        self._update_scaled_pixmap()
        self.update()

    def fit_to_view(self):
        """Calculates the initial 'fit-to-view' scale and applies it."""
        self.pixmap_offset = QPointF()
        if self.current_pixmap:
            label_size = self.size()
            pixmap_size = self.current_pixmap.size()
            if pixmap_size.width() > 0 and pixmap_size.height() > 0:
                scale_w = label_size.width() / pixmap_size.width()
                scale_h = label_size.height() / pixmap_size.height()
                fit_scale = min(scale_w, scale_h)
                self.scale_factor = min(1.0, fit_scale)
            else:
                self.scale_factor = 1.0
        else:
            self.scale_factor = 1.0
        self._update_scaled_pixmap()
        self.update()
        self.zoom_factor_changed.emit(self.scale_factor)
        self.view_changed.emit()

    def restore_view(self):
        """Resets the view to the initial 'fit-to-view' state."""
        self.fit_to_view()

    def _update_scaled_pixmap(self):
        if self.current_pixmap is None:
            self.cached_scaled_pixmap = None
            return
        interp_mode = self.zoom_in_interp if self.scale_factor > 1.0 else self.zoom_out_interp
        self.cached_scaled_pixmap = self.current_pixmap.scaled(
            self.current_pixmap.size() * self.scale_factor,
            Qt.AspectRatioMode.KeepAspectRatio,
            interp_mode
        )

    def _apply_zoom(self, new_scale_factor, mouse_pos=None):
        if self.current_pixmap is None:
            return
        old_scale_factor = self.scale_factor
        self.scale_factor = max(0.01, min(100.0, new_scale_factor))
        if abs(self.scale_factor - old_scale_factor) < 1e-9:
            return
        zoom_ratio = self.scale_factor / old_scale_factor
        if mouse_pos:
            mouse_rel_center = mouse_pos - QPointF(self.rect().center())
            self.pixmap_offset = mouse_rel_center - (mouse_rel_center - self.pixmap_offset) * zoom_ratio
        else:
            self.pixmap_offset *= zoom_ratio
        self._update_scaled_pixmap()
        self.update()
        self.zoom_factor_changed.emit(self.scale_factor)
        self.view_changed.emit()

    def wheelEvent(self, event):
        new_scale_factor = self.scale_factor
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
                    self._pinch_start_scale_factor = self.scale_factor
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

    def mouseMoveEvent(self, event):
        if not self.drag_start_position.isNull() and self.current_pixmap:
            delta = event.pos() - self.drag_start_position
            self.pixmap_offset += QPointF(delta)
            self.drag_start_position = event.pos()
            self.update()
            self.view_changed.emit()
        if self.cached_scaled_pixmap and self.image_data_for_hover is not None:
            label_coords = event.pos()
            target_rect = self.cached_scaled_pixmap.rect()
            target_rect.moveCenter(self.rect().center() + self.pixmap_offset.toPoint())
            if target_rect.contains(label_coords):
                pixmap_coords = label_coords - target_rect.topLeft()
                img_height, img_width = self.image_data_for_hover.shape[0], self.image_data_for_hover.shape[1]
                if target_rect.width() > 0 and target_rect.height() > 0:
                    x = int(pixmap_coords.x() * (img_width / target_rect.width()))
                    y = int(pixmap_coords.y() * (img_height / target_rect.height()))
                    if 0 <= x < img_width and 0 <= y < img_height:
                        self.hover_moved.emit(x, y)
                    else:
                        self.hover_left.emit()
            else:
                self.hover_left.emit()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start_position = QPoint()

    def paintEvent(self, event):
        if self.cached_scaled_pixmap is None:
            super().paintEvent(event)
            return
        painter = QPainter(self)
        target_rect = self.cached_scaled_pixmap.rect()
        target_rect.moveCenter(self.rect().center() + self.pixmap_offset.toPoint())
        painter.drawPixmap(target_rect, self.cached_scaled_pixmap)
