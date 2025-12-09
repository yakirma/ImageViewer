import numpy as np
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QAction, QPixmap, QImage, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QStatusBar,
    QPushButton,
    QStackedWidget,
    QMessageBox,
    QComboBox,
    QToolBar,
    QSizePolicy,
    QSlider,
)
import matplotlib.cm as cm

from widgets import ZoomableDraggableLabel, InfoPane, MathTransformPane, ZoomSettingsDialog, HistogramWidget
from image_handler import ImageHandler
import settings


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Viewer")
        QApplication.instance().installEventFilter(self)

        self.image_handler = ImageHandler()
        self.display_data = None
        self.recent_files = settings.load_recent_files()

        screen_geometry = self.screen().geometry()
        self.resize(screen_geometry.width() // 2, screen_geometry.height() // 2)
        self.move(screen_geometry.center() - self.rect().center())

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self._create_welcome_screen()
        self._create_image_display()

        self.current_colormap = "gray"
        self.contrast_limits = None
        self.zoom_settings = {"zoom_speed": 1.1, "zoom_in_interp": "Nearest", "zoom_out_interp": "Smooth"}

        self._create_menus_and_toolbar()
        self._create_info_pane()
        self._create_math_transform_pane()
        self._create_histogram_window()

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.image_label.hover_moved.connect(self.update_status_bar)
        self.image_label.hover_left.connect(self.status_bar.clearMessage)
        self.image_label.zoom_factor_changed.connect(self._on_image_label_zoom_changed)
        self.image_label.view_changed.connect(self.update_histogram_data)

    def _create_welcome_screen(self):
        welcome_widget = QWidget()
        welcome_layout = QVBoxLayout(welcome_widget)
        welcome_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # title_label = QLabel("Image Viewer")
        # font = title_label.font()
        # font.setPointSize(24)
        # title_label.setFont(font)
        open_button = QPushButton("Open Image")
        open_button.clicked.connect(self.open_image_dialog)

        drag_drop_label = QLabel("(or drag and drop an image file here)")
        drag_drop_label.setStyleSheet("color: gray; font-style: italic;")

        # welcome_layout.addWidget(title_label)
        welcome_layout.addWidget(open_button)
        welcome_layout.addWidget(drag_drop_label)
        self.stacked_widget.addWidget(welcome_widget)

    def _create_image_display(self):
        self.image_label = ZoomableDraggableLabel()
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(0, 4000)
        self.zoom_slider.setValue(2000)
        self.zoom_slider.setTickInterval(1000)
        self.zoom_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.zoom_slider.valueChanged.connect(self._on_zoom_slider_changed)
        self.zoom_slider.setEnabled(False)
        image_display_container = QWidget()
        image_display_layout = QVBoxLayout(image_display_container)
        image_display_layout.addWidget(self.image_label)
        image_display_layout.addWidget(self.zoom_slider)
        self.stacked_widget.addWidget(image_display_container)

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
        toolbar.addAction(open_action)
        settings_action = QAction(QIcon.fromTheme("preferences-system"), "Settings", self)
        settings_action.triggered.connect(self.open_zoom_settings)
        toolbar.addAction(settings_action)
        restore_action = QAction(QIcon("assets/icons/expand.png"), "Restore View", self)
        restore_action.triggered.connect(self.image_label.restore_view)
        toolbar.addAction(restore_action)
        self.colormap_combo = QComboBox(self)
        self.colormap_combo.addItems(["gray", "turbo", "viridis"])
        self.colormap_combo.setCurrentText(self.current_colormap)
        self.colormap_combo.currentTextChanged.connect(self.set_colormap)
        toolbar.addWidget(QLabel("Colormap:", self))
        toolbar.addWidget(self.colormap_combo)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        self.histogram_action = QAction(QIcon("assets/icons/histogram.png"), "Histogram", self)
        self.histogram_action.triggered.connect(self.toggle_histogram_window)
        self.histogram_action.setEnabled(False)
        toolbar.addAction(self.histogram_action)
        self.math_transform_action = QAction(QIcon.fromTheme("accessories-calculator"), "Math Transform", self)
        self.math_transform_action.triggered.connect(self.toggle_math_transform_pane)
        self.math_transform_action.setEnabled(False)
        toolbar.addAction(self.math_transform_action)
        self.info_action = QAction(QIcon.fromTheme("dialog-information"), "Image Info", self)
        self.info_action.triggered.connect(self.toggle_info_pane)
        self.info_action.setEnabled(False)
        toolbar.addAction(self.info_action)

    def _create_info_pane(self):
        self.info_pane = InfoPane(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.info_pane)
        self.info_pane.hide()
        self.info_pane.apply_clicked.connect(self.reapply_raw_parameters)

    def _create_math_transform_pane(self):
        self.math_transform_pane = MathTransformPane(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.math_transform_pane)
        self.math_transform_pane.hide()
        self.math_transform_pane.transform_requested.connect(self.apply_math_transform)

    def _create_histogram_window(self):
        self.histogram_window = HistogramWidget()
        self.histogram_window.region_changed.connect(self.set_contrast_limits)
        self.histogram_window.hide()

    def toggle_info_pane(self):
        self.info_pane.setVisible(not self.info_pane.isVisible())

    def toggle_math_transform_pane(self):
        self.math_transform_pane.setVisible(not self.math_transform_pane.isVisible())

    def toggle_histogram_window(self):
        if self.histogram_window.isVisible():
            self.histogram_window.hide()
        else:
            self.histogram_window.show()
            self.update_histogram_data()

    def _on_zoom_slider_changed(self, value):
        zoom_factor = 10 ** ((value / 1000.0) - 2.0)
        if abs(self.image_label.scale_factor - zoom_factor) > 1e-5:
            self.image_label.blockSignals(True)
            self.image_label._apply_zoom(zoom_factor)
            self.image_label.blockSignals(False)

    def _on_image_label_zoom_changed(self, scale_factor):
        if scale_factor > 0:
            slider_value = int((np.log10(scale_factor) + 2.0) * 1000.0)
            self.zoom_slider.blockSignals(True)
            self.zoom_slider.setValue(slider_value)
            self.zoom_slider.blockSignals(False)

    def reapply_raw_parameters(self, settings):
        try:
            self.image_handler.load_image(self.image_handler.current_file_path, override_settings=settings)
            self.display_data = self.image_handler.original_image_data.copy()
            self.apply_colormap(is_new_image=True)
            self.update_histogram_data(new_image=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying parameters:\n{e}")

    def apply_math_transform(self, expression):
        try:
            self.display_data = self.image_handler.apply_math_transform(expression)
            self.colormap_combo.setEnabled(True)
            self.apply_colormap()
            self.math_transform_pane.set_error_message("")
            self.update_histogram_data(new_image=True)
        except Exception as e:
            self.math_transform_pane.set_error_message(f"Error: {e}")

    def open_zoom_settings(self):
        dialog = ZoomSettingsDialog(self)
        dialog.set_settings(self.zoom_settings)
        if dialog.exec():
            self.zoom_settings = dialog.get_settings()
            self.apply_zoom_settings()

    def apply_zoom_settings(self):
        self.image_label.zoom_speed = self.zoom_settings["zoom_speed"]
        interp_map = {"Smooth": Qt.TransformationMode.SmoothTransformation,
                      "Nearest": Qt.TransformationMode.FastTransformation}
        self.image_label.zoom_in_interp = interp_map[self.zoom_settings["zoom_in_interp"]]
        self.image_label.zoom_out_interp = interp_map[self.zoom_settings["zoom_out_interp"]]

    def set_colormap(self, name):
        self.current_colormap = name
        self.apply_colormap()

    def set_contrast_limits(self, min_val, max_val):
        self.contrast_limits = (min_val, max_val)
        self.apply_colormap()

    def apply_colormap(self, is_new_image=False):
        if self.display_data is None:
            return

        data = self.display_data

        if len(data.shape) == 3 and data.shape[2] == 3:  # Color Image
            if self.contrast_limits:
                min_val, max_val = self.contrast_limits
                stretched_channels = []
                for i in range(3):
                    channel = data[..., i].astype(np.float32)
                    stretched = 255 * (channel - min_val) / (max_val - min_val)
                    stretched_channels.append(np.clip(stretched, 0, 255))
                processed_data = np.stack(stretched_channels, axis=-1).astype(np.uint8)
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

            colored_data = cm.get_cmap(self.current_colormap)(norm_data)
            image_data_8bit = (colored_data[:, :, :3] * 255).astype(np.uint8)
            h, w, _ = image_data_8bit.shape
            q_image = QImage(image_data_8bit.data, w, h, 3 * w, QImage.Format.Format_RGB888)

        if is_new_image:
            self.image_label.set_pixmap(QPixmap.fromImage(q_image), self.display_data)
        else:
            self.image_label.update_pixmap_content(QPixmap.fromImage(q_image))

    def open_image_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "",
                                                   f"Image Files (*.png *.jpg *.jpeg *.bmp *.tiff {' '.join(['*' + ext for ext in self.image_handler.raw_extensions])})")
        if file_path:
            self.open_file(file_path)

    def open_file(self, file_path):
        try:
            self.image_handler.load_image(file_path)
            self.display_data = self.image_handler.original_image_data.copy()

            self.info_action.setEnabled(self.image_handler.is_raw)
            self.info_pane.set_raw_mode(self.image_handler.is_raw)
            self.math_transform_action.setEnabled(True)
            self.zoom_slider.setEnabled(True)
            self.histogram_action.setEnabled(True)
            self.colormap_combo.setEnabled(self.image_handler.is_single_channel_image())

            if self.image_handler.is_raw:
                self.info_pane.update_info(self.image_handler.width, self.image_handler.height,
                                           self.image_handler.dtype, self.image_handler.dtype_map)
            else:
                self.info_pane.hide()

            self.contrast_limits = None
            self.apply_colormap(is_new_image=True)

            if self.image_label.current_pixmap:
                self.stacked_widget.setCurrentIndex(1)
                self.recent_files = settings.add_to_recent_files(self.recent_files, file_path)
                self._update_recent_files_menu()
            self.update_histogram_data(new_image=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error opening image:\n{e}")
            self.math_transform_action.setEnabled(False)
            self.info_action.setEnabled(False)
            self.info_pane.set_raw_mode(False)
            self.zoom_slider.setEnabled(False)
            self.histogram_action.setEnabled(False)

    def save_view(self):
        if self.image_label.current_pixmap is None:
            return
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Image (*.png)")
        if file_name:
            self.image_label.current_pixmap.save(file_name, "png")

    def update_status_bar(self, x_coord, y_coord):
        if self.display_data is not None and y_coord < self.display_data.shape[0] and x_coord < self.display_data.shape[
            1]:
            value = self.display_data[y_coord, x_coord]
            self.status_bar.showMessage(f"({x_coord}, {y_coord}): {value}")

    def _update_recent_files_menu(self):
        self.recent_files_menu.clear()
        if not self.recent_files:
            self.recent_files_menu.setEnabled(False)
            return
        self.recent_files_menu.setEnabled(True)
        for file_path in self.recent_files:
            action = QAction(file_path, self)
            action.triggered.connect(lambda checked, path=file_path: self.open_file(path))
            self.recent_files_menu.addAction(action)

    def update_histogram_data(self, new_image=False):
        visible_data = self._get_visible_image_data()

        try:
            self.histogram_window.region_changed.disconnect(self.set_contrast_limits)
        except TypeError:
            pass

        self.histogram_window.update_histogram(visible_data, new_image)

        self.histogram_window.region_changed.connect(self.set_contrast_limits)

    def keyPressEvent(self, event):
        if self.display_data is not None:
            if event.key() == Qt.Key.Key_M:
                if event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
                    self._apply_histogram_preset(5, 95)
                else:
                    self._apply_histogram_preset(0, 100)
        super().keyPressEvent(event)

    def _apply_histogram_preset(self, min_percent, max_percent):
        visible_data = self._get_visible_image_data()
        if visible_data is not None and visible_data.size > 0:
            if len(visible_data.shape) == 3:
                hist_data = np.dot(visible_data[..., :3], [0.2989, 0.5870, 0.1140])
            else:
                hist_data = visible_data

            min_val = np.percentile(hist_data, min_percent)
            max_val = np.percentile(hist_data, max_percent)
            self.set_contrast_limits(min_val, max_val)
            self.histogram_window.region.setRegion([min_val, max_val])

    def _get_visible_image_data(self):
        if self.image_handler.original_image_data is None or not self.image_label.cached_scaled_pixmap:
            return None

        data_for_hist = self.image_handler.original_image_data

        visible_rect = self.image_label.rect()
        pixmap_rect = self.image_label.cached_scaled_pixmap.rect()
        pixmap_rect.moveCenter(self.image_label.rect().center() + self.image_label.pixmap_offset.toPoint())
        intersection = visible_rect.intersected(pixmap_rect)
        if intersection.isEmpty():
            return None
        scale_factor = self.image_label.scale_factor
        if scale_factor <= 0: return None
        x1 = int((intersection.left() - pixmap_rect.left()) / scale_factor)
        y1 = int((intersection.top() - pixmap_rect.top()) / scale_factor)
        x2 = int((intersection.right() - pixmap_rect.left()) / scale_factor)
        y2 = int((intersection.bottom() - pixmap_rect.top()) / scale_factor)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.image_handler.width, x2)
        y2 = min(self.image_handler.height, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return data_for_hist[y1:y2, x1:x2]

    def restore_image_view(self):
        self.contrast_limits = None
        self.display_data = self.image_handler.original_image_data.copy()
        self.colormap_combo.setEnabled(self.image_handler.is_single_channel_image())
        self.apply_colormap(is_new_image=True)
        self.update_histogram_data(new_image=True)

    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.DragEnter:
            if event.mimeData().hasUrls():
                urls = event.mimeData().urls()
                if urls:
                    file_path = urls[0].toLocalFile()
                    _, ext = os.path.splitext(file_path)
                    supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp',
                                            '.tiff'] + self.image_handler.raw_extensions
                    if ext.lower() in supported_extensions:
                        event.acceptProposedAction()
        elif event.type() == QEvent.Type.Drop:
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()
                self.open_file(file_path)
                return True
        return super().eventFilter(source, event)