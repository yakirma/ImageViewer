import os

import numpy as np
from PyQt6.QtCore import Qt, QEvent, QPoint
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
    QGridLayout,
)
import matplotlib.cm as cm

from widgets import ZoomableDraggableLabel, InfoPane, MathTransformPane, ZoomSettingsDialog, HistogramWidget, \
    ThumbnailPane, SharedViewState
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

        self.image_handler = ImageHandler()
        self.recent_files = settings.load_recent_files()
        self.current_file_path = None
        self.montage_shared_state = None
        self.montage_labels = []
        self.active_label = None

        screen_geometry = self.screen().geometry()
        self.resize(screen_geometry.width() // 2, screen_geometry.height() // 2)
        self.move(screen_geometry.center() - self.rect().center())

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.current_colormap = "gray"
        self.zoom_settings = {"zoom_speed": 1.1, "zoom_in_interp": "Nearest", "zoom_out_interp": "Smooth"}

        self._create_welcome_screen()
        self._create_image_display()
        self._create_montage_view()



        self._create_menus_and_toolbar()
        self._create_info_pane()
        self._create_math_transform_pane()
        self._create_histogram_window()
        self._create_thumbnail_pane()

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.zoom_status_label = QLabel("Zoom: 100%")
        self.status_bar.addPermanentWidget(self.zoom_status_label)

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
        image_display_layout = QVBoxLayout(self.image_display_container)
        image_display_layout.addWidget(self.image_label)
        image_display_layout.addWidget(self.zoom_slider)


        self.stacked_widget.addWidget(self.image_display_container)
        self.apply_zoom_settings()  # Apply default settings immediately

    def _create_montage_view(self):
        self.montage_widget = QWidget()
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
        self.colormap_combo.addItems(["gray", "turbo", "viridis"])
        self.colormap_combo.setCurrentText(self.current_colormap)
        self.colormap_combo.currentTextChanged.connect(self.set_colormap)
        toolbar.addWidget(QLabel("Colormap:", self))
        toolbar.addWidget(self.colormap_combo)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        self.histogram_action = QAction(QIcon(resource_path("assets/icons/histogram.png")), "Histogram", self)
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

        thumbnail_action = QAction(QIcon(resource_path("assets/icons/opened_images.png")), "Opened Images", self)
        thumbnail_action.triggered.connect(self.toggle_thumbnail_pane)
        toolbar.addAction(thumbnail_action)

    def _create_info_pane(self):
        self.info_pane = InfoPane(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.info_pane)
        self.info_pane.hide()
        self.info_pane.apply_clicked.connect(self.reapply_raw_parameters)

    def _create_math_transform_pane(self):
        self.math_transform_pane = MathTransformPane(self)
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
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.thumbnail_pane)
        self.thumbnail_pane.hide()
        self.thumbnail_pane.selection_changed.connect(self.display_montage)
        self.thumbnail_pane.overlay_changed.connect(self._on_overlay_changed)

        self.overlay_alphas = {} # path -> alpha
        self.overlay_cache = {}  # path -> QPixmap (resized to match current active view)

    def display_montage(self, file_paths):
        for i in reversed(range(self.montage_layout.count())):
            widget = self.montage_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        self.montage_labels.clear()

        if not file_paths:
            if self.current_file_path:
                self.open_file(self.current_file_path)
            return


        self.montage_shared_state = SharedViewState()
        max_montage_zoom = 100.0

        row, col = 0, 0
        for file_path in file_paths:
            temp_handler = ImageHandler()
            temp_handler.load_image(file_path)
            data = temp_handler.original_image_data

            if data is not None and data.size > 0:
                # Update Max Zoom Limit
                dims = max(temp_handler.width, temp_handler.height)
                if dims > max_montage_zoom: max_montage_zoom = float(dims)

                image_label = ZoomableDraggableLabel(shared_state=self.montage_shared_state)
                image_label.set_data(data, is_pristine=True)
                image_label.file_path = file_path # Store for overlay restoration

                # Check if this file is already open in another window and inherit its state
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

                self.montage_layout.addWidget(image_label, row, col)
                image_label.hover_moved.connect(lambda x, y, label=image_label: self._set_active_montage_label(label))
                
                col += 1
                if col % 3 == 0:
                    row += 1
                    col = 0
        if self.montage_labels:
            self.montage_shared_state.max_zoom_limit = max_montage_zoom
            self._set_active_montage_label(self.montage_labels[0])

        self.stacked_widget.setCurrentWidget(self.montage_widget)

    def _set_active_montage_label(self, label):
        if self.active_label:
            self.active_label.set_active(False)
        self.active_label = label
        if self.active_label:
            self.active_label.set_active(True)
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

        self.colormap_combo.setEnabled(self.active_label.is_single_channel())
        
        # Sync the colormap combo box with the active label's current colormap
        self.colormap_combo.blockSignals(True)
        self.colormap_combo.setCurrentText(self.active_label.colormap)
        self.colormap_combo.blockSignals(False)

        self.update_histogram_data(new_image=reset_histogram)

    def toggle_info_pane(self):
        self.info_pane.setVisible(not self.info_pane.isVisible())

    def toggle_math_transform_pane(self):
        visible = not self.math_transform_pane.isVisible()
        self.math_transform_pane.setVisible(visible)
        
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
                     # For now, let's assume they want filenames if they close the pane?
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
        if self.thumbnail_pane.isVisible():
            self.thumbnail_pane.hide()
        else:
            self.thumbnail_pane.populate(self.window_list)
            self.thumbnail_pane.show()
            self.thumbnail_pane.setFocus()

    def _on_zoom_slider_changed(self, value):
        if self.sender() is not self.zoom_slider: return
        if self.active_label:
            zoom_factor = 10 ** ((value / 1000.0) - 2.0)
            if abs(self.active_label._get_scale_factor() - zoom_factor) > 1e-5:
                self.active_label._apply_zoom(zoom_factor)

    def _on_image_label_zoom_changed(self, scale_factor):
        if self.sender() is not self.active_label: return
        if scale_factor > 0:
            slider_value = int((np.log10(scale_factor) + 2.0) * 1000.0)
            self.zoom_slider.blockSignals(True)
            self.zoom_slider.setValue(slider_value)
            self.zoom_slider.blockSignals(False)
            
            self.zoom_status_label.setText(f"Zoom: {int(scale_factor * 100)}%")

    def reapply_raw_parameters(self, settings):
        try:
            self.image_handler.load_image(self.current_file_path, override_settings=settings)
            if self.active_label:
                self.active_label.set_data(self.image_handler.original_image_data)
            self.update_histogram_data(new_image=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying parameters:\n{e}")

    def apply_math_transform(self, expression):
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
            self.update_histogram_data(new_image=True)
            self.math_transform_pane.set_error_message("")
        except Exception as e:
            self.math_transform_pane.set_error_message(str(e))

    def restore_original_image(self):
        if self.active_label and self.active_label.pristine_data is not None:
            self.active_label.set_data(self.active_label.pristine_data, reset_view=False)
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
        if self.active_label:
            self.active_label.set_colormap(name)

    def set_contrast_limits(self, min_val, max_val):
        # Apply only to the active label, ensuring independence in montage view
        if self.active_label:
            self.active_label.set_contrast_limits(min_val, max_val)

    def open_image_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "",
                                                   f"Image Files (*.png *.jpg *.jpeg *.bmp *.tiff {' '.join(['*' + ext for ext in self.image_handler.raw_extensions])})")
        if file_path:
            self.open_file(file_path)

    def open_file(self, file_path):
        self.current_file_path = file_path
        try:
            self.image_handler.load_image(file_path)

            self.info_action.setEnabled(self.image_handler.is_raw)
            self.info_pane.set_raw_mode(self.image_handler.is_raw)
            self.math_transform_action.setEnabled(True)
            self.zoom_slider.setEnabled(True)
            self.histogram_action.setEnabled(True)

            if self.image_handler.is_raw:
                self.info_pane.update_info(self.image_handler.width, self.image_handler.height,
                                           self.image_handler.dtype, self.image_handler.dtype_map)
            else:
                self.info_pane.hide()
            
            # Load Data (triggers render with set attributes)
            self.image_label.set_data(self.image_handler.original_image_data, is_pristine=True)
            self._set_active_montage_label(self.image_label)
            
            # Apply default Min-Max contrast stretch for new image
            self._apply_histogram_preset(0, 100)
            
            # Clear overlay cache as dimensions/proxy might have changed
            self.overlay_cache.clear()
            self._update_overlays() 
    
            if self.image_label.current_pixmap:
                self.stacked_widget.setCurrentWidget(self.image_display_container)
                self.recent_files = settings.add_to_recent_files(self.recent_files, file_path)
                self._update_recent_files_menu()

                # Update path overlay
                self.image_label.set_overlay_text(file_path)

        except Exception as e:
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
            action.triggered.connect(lambda checked, path=file_path: self.open_file(path))
            self.recent_files_menu.addAction(action)

    def update_histogram_data(self, new_image=False):
        if not self.active_label:
            return
            
        use_visible_only = self.histogram_window.use_visible_checkbox.isChecked()
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
                    hist_data = np.dot(visible_data[..., :3], [0.2989, 0.5870, 0.1140])
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

        if event.key() == Qt.Key.Key_N and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
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

    def resizeEvent(self, event):
        super().resizeEvent(event)


    def _apply_histogram_preset(self, min_percent, max_percent, use_visible_only=False):
        visible_data = self._get_visible_image_data(use_visible_only)
        if visible_data is not None and visible_data.size > 0:
            if len(visible_data.shape) == 3:
                hist_data = np.dot(visible_data[..., :3], [0.2989, 0.5870, 0.1140])
            else:
                hist_data = visible_data

            min_val = np.percentile(hist_data, min_percent)
            max_val = np.percentile(hist_data, max_percent)
            self.set_contrast_limits(min_val, max_val)
            self.histogram_window.region.setRegion([min_val, max_val])

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
             # Reset data which clears contrast limits
             self.active_label.set_data(self.active_label.original_data)
             # Reset View
             self.active_label.restore_view()
             # Reset Colormap
             self.set_colormap("gray")
             self.colormap_combo.setCurrentText("gray")
             # Reset Histogram/UI
             self._update_active_view(reset_histogram=True)
             
             # Apply default Min-Max contrast again to ensure it's not raw/black
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
                                                '.tiff'] + self.image_handler.raw_extensions
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
                    # Load and resize logic
                    try:
                        handler = ImageHandler()
                        handler.load_image(path)
                        if handler.original_image_data is not None:
                            # Create Pixmap from original data
                            # Note: We need to resize to target_size (which might be a Proxy)
                            # QImage construction
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

        self.active_label.set_overlays(overlays_to_draw)

    def closeEvent(self, event):
        """Remove window from the list when closed."""
        if self in self.window_list:
            self.window_list.remove(self)
        super().closeEvent(event)