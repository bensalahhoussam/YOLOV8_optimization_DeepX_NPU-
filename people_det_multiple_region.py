import sys
import cv2 as cv
import numpy as np
from queue import Queue
from threading import Lock
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QFileDialog, QComboBox,
                             QMessageBox, QListWidget, QListWidgetItem)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QColor
from ultralytics import YOLO
import os
from ultralytics.utils.plotting import Annotator, colors
import random
from collections import defaultdict

names = {0: 'person'}
person_colors = {}
track_history = defaultdict(list)


def draw_dashed_line(img, p1, p2, color, thickness=2, dash=10):
    """Draw a dashed line between two points"""
    p1 = np.array(p1)
    p2 = np.array(p2)
    dist = np.linalg.norm(p2 - p1)
    if dist == 0:
        return

    direction = (p2 - p1) / dist

    for i in range(0, int(dist), dash * 2):
        start = p1 + direction * i
        end = p1 + direction * min(i + dash, dist)
        cv.line(img, tuple(start.astype(int)), tuple(end.astype(int)), color, thickness)


def draw_dashed_polygon(img, pts, color, thickness=2, dash=10):
    """Draw a dashed polygon"""
    if len(pts) < 2:
        return

    # Draw dashed lines between consecutive points
    for i in range(len(pts)):
        draw_dashed_line(img, pts[i], pts[(i + 1) % len(pts)], color, thickness, dash)

    # Draw closing line if we have more than 2 points
    if len(pts) >= 3:
        draw_dashed_line(img, pts[-1], pts[0], color, thickness, dash)


def fill_polygon_transparent(img, pts, color, alpha=0.2):
    """Fill polygon with transparent color"""
    if len(pts) < 3:
        return

    # Create overlay
    overlay = img.copy()

    # Fill polygon with color
    pts_array = np.array(pts, np.int32).reshape((-1, 1, 2))
    cv.fillPoly(overlay, [pts_array], color)

    # Blend with original image
    cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


class ROI:
    """Class to represent a single ROI region"""
    # Predefined distinct colors for ROIs (BGR format for OpenCV)
    roi_colors = [
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (0, 255, 0),  # Green
        (255, 0, 0),  # Blue
        (0, 165, 255),  # Orange
        (255, 192, 203),  # Pink
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (128, 128, 0),  # Olive
        (128, 0, 0),  # Maroon
        (0, 128, 0),  # Dark Green
    ]

    color_counter = 0  # Class variable to keep track of color assignment

    def __init__(self, name="ROI"):
        self.name = name
        self.points = []
        self.defined = False

        # Assign a unique color from the palette
        self.color_idx = ROI.color_counter % len(ROI.roi_colors)
        self.color = ROI.roi_colors[self.color_idx]
        self.alpha = 0.15  # Opacity for fill

        # Increment counter for next ROI
        ROI.color_counter += 1

        self.count = 0
        self.track_ids = set()  # Track IDs currently inside this ROI
        self.entry_count = 0  # Total entries into ROI
        self.exit_count = 0  # Total exits from ROI

    def add_point(self, point):
        self.points.append(point)

    def finish(self):
        if len(self.points) >= 3:
            self.defined = True
            return True
        return False

    def clear(self):
        self.points = []
        self.defined = False
        self.count = 0
        self.track_ids.clear()

    def contains_point(self, point):
        """Check if a point is inside this ROI polygon"""
        if len(self.points) < 3 or not self.defined:
            return False
        roi_np = np.array(self.points, np.int32)
        return cv.pointPolygonTest(roi_np, point, False) >= 0

    def draw(self, img, is_active=False):
        """Draw the ROI on the image with transparent fill"""
        if not self.points:
            return

        # Fill ROI with transparent color if defined
        if self.defined and len(self.points) >= 3:
            fill_polygon_transparent(img, self.points, self.color, self.alpha)

        # Draw points
        point_color = (0, 0, 255) if is_active else self.color
        for i, point in enumerate(self.points):
            cv.circle(img, point, 8, point_color, -1)  # Larger points
            cv.circle(img, point, 8, (255, 255, 255), 2)  # White border

            # Label points
            cv.putText(img, str(i + 1),
                       (point[0] + 12, point[1] - 12),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)  # White background
            cv.putText(img, str(i + 1),
                       (point[0] + 12, point[1] - 12),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, point_color, 2)  # Colored text

        # Draw dashed polygon with thicker lines for visibility
        line_color = (0, 255, 255) if is_active else self.color
        if len(self.points) >= 2:
            draw_dashed_polygon(img, self.points, line_color, thickness=3, dash=20)

            # Draw closing line if ROI is defined
            if self.defined and len(self.points) >= 3:
                draw_dashed_line(img, self.points[-1], self.points[0],
                                 line_color, thickness=3, dash=20)

        # Draw ROI name and statistics with background for readability
        if self.points and self.defined:
            center_x = sum(p[0] for p in self.points) // len(self.points)
            center_y = sum(p[1] for p in self.points) // len(self.points)

            # Draw background rectangle for text
            text = f"{self.name}: {self.count}"
            text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]

            # Semi-transparent black background
            overlay = img.copy()
            """cv.rectangle(overlay,
                         (center_x - 70, center_y - 30),
                         (center_x + text_size[0] + 10, center_y + 15),
                         (0, 0, 0), -1)"""
            cv.addWeighted(overlay, 0.6, img, 0.4, 0, img)

            # White border
            """cv.rectangle(img,
                         (center_x - 70, center_y - 30),
                         (center_x + text_size[0] + 10, center_y + 15),
                         (255, 255, 255), 2)"""

            # Draw text
            """cv.putText(img, text, (center_x - 60, center_y),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, self.color, 3)"""


class VideoLabel(QLabel):
    clicked = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def mousePressEvent(self, event):
        self.clicked.emit(event)


class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = False
        self.source = 0
        self.cap = None
        self.lock = Lock()

    def set_source(self, source):
        with self.lock:
            self.source = source
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    def run(self):
        with self.lock:
            if isinstance(self.source, str) and self.source != "0":
                if not os.path.exists(self.source):
                    self.error_occurred.emit(f"Video file not found: {self.source}")
                    return

            self.cap = cv.VideoCapture(self.source)
            if not self.cap.isOpened():
                error_msg = f"Error: Could not open video source {self.source}"
                self.error_occurred.emit(error_msg)
                return

        self.running = True

        while self.running:
            with self.lock:
                if self.cap is None:
                    break
                ret, frame = self.cap.read()

            if not ret:
                if isinstance(self.source, str) and self.source != "0":
                    with self.lock:
                        self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            self.frame_ready.emit(frame)
            self.msleep(10)

    def stop(self):
        self.running = False
        if self.isRunning():
            self.wait()
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None


class InferenceThread(QThread):
    result_ready = pyqtSignal(np.ndarray, object)

    def __init__(self):
        super().__init__()
        try:
            self.model = YOLO("yolov8n.pt")
            self.model.eval()
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            try:
                self.model = YOLO("yolov8n.pt")
                self.model.eval()
            except:
                raise Exception("Could not load YOLO model.")

        self.queue = Queue(maxsize=1)
        self.running = True
        self.frame_lock = Lock()
        self.current_frame = None

    def submit(self, frame):
        if not self.queue.full():
            self.queue.put(frame)

    def run(self):
        while self.running:
            try:
                frame = self.queue.get(timeout=0.01)
                with self.frame_lock:
                    self.current_frame = frame.copy()
                """ run method need to change as infernce engine function for npu"""
                results = self.model.track(frame, persist=True, tracker="botsort.yaml", classes=[0], verbose=False)
                print(f"res:{results}")
                self.result_ready.emit(frame, results)

            except Exception:
                if self.running:
                    self.msleep(5)
                continue

    def stop(self):
        self.running = False
        self.wait()


class PersonCounterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Person Counter App - Multiple ROIs with Tracking")
        self.resize(1400, 800)

        # Multiple ROIs
        self.rois = []  # List of ROI objects
        self.active_roi_index = -1  # Index of currently active ROI for drawing
        self.drawing_roi = False

        # Track previous positions for each ROI
        self.previous_roi_states = []  # Store previous track_ids for each ROI

        self.current_frame = None
        self.fps = 0
        self.frame_count = 0

        self.init_ui()

        self.video_thread = VideoThread()
        self.yolo_thread = InferenceThread()

        self.video_thread.frame_ready.connect(self.yolo_thread.submit)
        self.video_thread.error_occurred.connect(self.handle_video_error)
        self.yolo_thread.result_ready.connect(self.process_result)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_fps)
        self.timer.start(1000)

        # Create first ROI
        self.add_new_roi()

    def init_ui(self):
        main_layout = QHBoxLayout()

        # Left panel for video and controls
        left_panel = QVBoxLayout()

        # Controls
        controls = QHBoxLayout()
        self.source_box = QComboBox()
        self.source_box.addItem("Webcam", "0")
        self.source_box.addItem("Video File", None)

        demo_video_path = "/home/houssem/Projects/yolov8_kd/VIRAT.mp4"
        self.source_box.addItem("Demo Video", demo_video_path)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.toggle_video)

        self.add_roi_btn = QPushButton("Add New ROI")
        self.add_roi_btn.clicked.connect(self.add_new_roi)
        self.add_roi_btn.setEnabled(False)

        self.start_draw_btn = QPushButton("Draw ROI")
        self.start_draw_btn.clicked.connect(self.start_drawing)
        self.start_draw_btn.setEnabled(False)

        self.finish_draw_btn = QPushButton("Finish Drawing")
        self.finish_draw_btn.clicked.connect(self.finish_drawing)
        self.finish_draw_btn.setEnabled(False)

        self.clear_roi_btn = QPushButton("Clear Current ROI")
        self.clear_roi_btn.clicked.connect(self.clear_current_roi)
        self.clear_roi_btn.setEnabled(False)

        self.delete_roi_btn = QPushButton("Delete Current ROI")
        self.delete_roi_btn.clicked.connect(self.delete_current_roi)
        self.delete_roi_btn.setEnabled(False)

        controls.addWidget(self.source_box)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.add_roi_btn)
        controls.addWidget(self.start_draw_btn)
        controls.addWidget(self.finish_draw_btn)
        controls.addWidget(self.clear_roi_btn)
        controls.addWidget(self.delete_roi_btn)

        # Video display
        self.video = VideoLabel()
        self.video.setStyleSheet("border:2px solid black")
        self.video.clicked.connect(self.on_mouse_click)

        # Status labels
        stats_layout = QHBoxLayout()
        self.total_counter = QLabel("Total People in ROIs: 0")
        self.total_counter.setStyleSheet("font-size:18px; font-weight:bold; color: white")

        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("font-size:14px; font-weight:bold; color: white")

        self.status_label = QLabel("Status: Stopped")
        self.status_label.setStyleSheet("font-size:14px; color: white")

        stats_layout.addWidget(self.total_counter)
        stats_layout.addWidget(self.fps_label)
        stats_layout.addWidget(self.status_label)

        left_panel.addLayout(controls)
        left_panel.addWidget(self.video)
        left_panel.addLayout(stats_layout)

        # Right panel for ROI list
        right_panel = QVBoxLayout()
        right_panel.setContentsMargins(10, 0, 0, 0)

        roi_list_label = QLabel("ROI List")
        roi_list_label.setStyleSheet("font-size:16px; font-weight:bold; color: white")
        right_panel.addWidget(roi_list_label)

        self.roi_list = QListWidget()
        self.roi_list.itemClicked.connect(self.on_roi_selected)
        self.roi_list.setMaximumWidth(300)

        # Set custom colors for list items
        self.roi_list.setStyleSheet("""
            QListWidget {
                background-color: #2b2b2b;
                color: white;
                border: 2px solid #555;
                border-radius: 5px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #555;
                font-weight: bold;
            }
            QListWidget::item:selected {
                background-color: #4a4a4a;
                border: 2px solid #888;
            }
        """)

        right_panel.addWidget(self.roi_list)

        # ROI summary
        summary_label = QLabel("ROI Summary")
        summary_label.setStyleSheet("font-size:16px; font-weight:bold; color: white")
        right_panel.addWidget(summary_label)

        self.summary_text = QLabel("No ROIs defined")
        self.summary_text.setWordWrap(True)
        self.summary_text.setMaximumWidth(300)
        self.summary_text.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                color: white;
                padding: 10px;
                border: 2px solid #555;
                border-radius: 5px;
                font-size: 12px;
            }
        """)
        right_panel.addWidget(self.summary_text)

        right_panel.addStretch()

        main_layout.addLayout(left_panel, 3)
        main_layout.addLayout(right_panel, 1)

        # Set main window background
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: white;
            }
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: 1px solid #555;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
                border: 1px solid #777;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
            QPushButton:disabled {
                background-color: #333;
                color: #777;
            }
            QComboBox {
                background-color: #4a4a4a;
                color: white;
                border: 1px solid #555;
                padding: 5px;
                border-radius: 4px;
            }
            QComboBox:hover {
                border: 1px solid #777;
            }
            QLabel {
                color: white;
            }
        """)

        self.setLayout(main_layout)

    def closeEvent(self, event):
        self.video_thread.stop()
        self.yolo_thread.stop()
        event.accept()

    @pyqtSlot(str)
    def handle_video_error(self, error_msg):
        QMessageBox.critical(self, "Video Error", error_msg)
        self.stop_video()

    @pyqtSlot()
    def toggle_video(self):
        if not self.video_thread.isRunning():
            source = self.source_box.currentData()
            if source is None:
                source, _ = QFileDialog.getOpenFileName(
                    self, "Select Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov)"
                )
                if not source:
                    return

            if source == 0:
                source = "0"

            self.video_thread.set_source(source)
            self.video_thread.start()
            self.yolo_thread.start()
            self.start_btn.setText("Stop")
            self.add_roi_btn.setEnabled(True)
            self.start_draw_btn.setEnabled(True)
            self.status_label.setText("Status: Running - Select an ROI to draw")
        else:
            self.stop_video()

    def stop_video(self):
        self.video_thread.stop()
        self.yolo_thread.stop()
        self.start_btn.setText("Start")
        self.add_roi_btn.setEnabled(False)
        self.start_draw_btn.setEnabled(False)
        self.finish_draw_btn.setEnabled(False)
        self.clear_roi_btn.setEnabled(False)
        self.delete_roi_btn.setEnabled(False)
        self.status_label.setText("Status: Stopped")

    def add_new_roi(self):
        """Add a new ROI to the list"""
        roi_count = len(self.rois) + 1
        new_roi = ROI(name=f"ROI {roi_count}")
        self.rois.append(new_roi)
        self.active_roi_index = len(self.rois) - 1

        # Add to list widget with color indicator
        item = QListWidgetItem(f"ROI {roi_count} (Not defined)")
        item.setData(Qt.ItemDataRole.UserRole, len(self.rois) - 1)

        # Set item color based on ROI color
        roi_color = new_roi.color
        qcolor = QColor(roi_color[2], roi_color[1], roi_color[0])  # BGR to RGB
        item.setForeground(qcolor)

        self.roi_list.addItem(item)
        self.roi_list.setCurrentItem(item)

        # Initialize previous state
        self.previous_roi_states.append(set())

        self.update_roi_list()
        self.status_label.setText(f"Status: Added ROI {roi_count}. Select it to start drawing.")

    def on_roi_selected(self, item):
        """Handle ROI selection from list"""
        roi_index = item.data(Qt.ItemDataRole.UserRole)
        if 0 <= roi_index < len(self.rois):
            self.active_roi_index = roi_index
            self.update_button_states()
            self.status_label.setText(f"Status: Selected {self.rois[roi_index].name}")

    def update_button_states(self):
        """Update button states based on current ROI state"""
        if self.active_roi_index >= 0 and self.active_roi_index < len(self.rois):
            current_roi = self.rois[self.active_roi_index]

            self.start_draw_btn.setEnabled(True)
            self.clear_roi_btn.setEnabled(True)
            self.delete_roi_btn.setEnabled(True)

            if self.drawing_roi:
                self.finish_draw_btn.setEnabled(True)
                self.start_draw_btn.setEnabled(False)
            else:
                self.finish_draw_btn.setEnabled(False)

    @pyqtSlot()
    def start_drawing(self):
        """Start drawing the currently selected ROI"""
        if self.active_roi_index >= 0 and self.current_frame is not None:
            current_roi = self.rois[self.active_roi_index]
            current_roi.clear()  # Clear existing points
            self.drawing_roi = True
            self.update_button_states()
            self.status_label.setText(f"Status: Drawing {current_roi.name} - Click points on video")

    @pyqtSlot()
    def finish_drawing(self):
        """Finish drawing the current ROI"""
        if self.active_roi_index >= 0 and self.drawing_roi:
            current_roi = self.rois[self.active_roi_index]
            if current_roi.finish():
                self.drawing_roi = False
                self.update_button_states()
                self.update_roi_list()
                self.status_label.setText(f"Status: {current_roi.name} defined with {len(current_roi.points)} points")
            else:
                self.status_label.setText(f"Status: Need at least 3 points for {current_roi.name}")

    @pyqtSlot()
    def clear_current_roi(self):
        """Clear the currently selected ROI"""
        if self.active_roi_index >= 0:
            current_roi = self.rois[self.active_roi_index]
            current_roi.clear()
            self.drawing_roi = False
            self.update_button_states()
            self.update_roi_list()
            self.status_label.setText(f"Status: {current_roi.name} cleared")

    @pyqtSlot()
    def delete_current_roi(self):
        """Delete the currently selected ROI"""
        if self.active_roi_index >= 0:
            roi_name = self.rois[self.active_roi_index].name

            # Remove from list
            self.roi_list.takeItem(self.active_roi_index)

            # Remove from ROIs list
            del self.rois[self.active_roi_index]

            # Remove from previous states
            if self.active_roi_index < len(self.previous_roi_states):
                del self.previous_roi_states[self.active_roi_index]

            # Update indices in list items
            for i in range(self.roi_list.count()):
                item = self.roi_list.item(i)
                item.setData(Qt.ItemDataRole.UserRole, i)

            # Reset active ROI
            if self.rois:
                self.active_roi_index = 0
                self.roi_list.setCurrentRow(0)
            else:
                self.active_roi_index = -1
                # Reset color counter when all ROIs are deleted
                ROI.color_counter = 0
                self.add_new_roi()  # Add a new empty ROI

            self.update_button_states()
            self.status_label.setText(f"Status: {roi_name} deleted")

    def on_mouse_click(self, event):
        if not self.drawing_roi or self.current_frame is None or self.active_roi_index < 0:
            return

        pixmap = self.video.pixmap()
        if pixmap is None:
            return

        # Calculate click position
        lw, lh = self.video.width(), self.video.height()
        pw, ph = pixmap.width(), pixmap.height()

        x_offset = (lw - pw) // 2
        y_offset = (lh - ph) // 2

        x = event.position().x() - x_offset
        y = event.position().y() - y_offset

        if x < 0 or y < 0 or x >= pw or y >= ph:
            return

        # Convert to frame coordinates
        fh, fw = self.current_frame.shape[:2]
        fx = int(x * fw / pw)
        fy = int(y * fh / ph)

        # Add point to current ROI
        if event.button() == Qt.MouseButton.LeftButton:
            current_roi = self.rois[self.active_roi_index]
            current_roi.add_point((fx, fy))
            self.status_label.setText(f"Status: {current_roi.name} - Point {len(current_roi.points)} added")

    def update_roi_list(self):
        """Update the ROI list display"""
        for i in range(self.roi_list.count()):
            item = self.roi_list.item(i)
            roi_index = item.data(Qt.ItemDataRole.UserRole)
            if 0 <= roi_index < len(self.rois):
                roi = self.rois[roi_index]
                status = "Defined" if roi.defined else "Not defined"
                count_text = f" ({roi.count} people)" if roi.defined else ""
                item.setText(f"{roi.name}: {status}{count_text}")

                # Set item color to match ROI color
                roi_color = roi.color
                qcolor = QColor(roi_color[2], roi_color[1], roi_color[0])  # BGR to RGB
                item.setForeground(qcolor)

                # Make defined ROIs bold
                font = item.font()
                font.setBold(roi.defined)
                item.setFont(font)

    def update_summary(self):
        """Update the ROI summary text"""
        if not self.rois:
            self.summary_text.setText("No ROIs defined")
            return

        total_people = sum(roi.count for roi in self.rois if roi.defined)
        defined_rois = [roi for roi in self.rois if roi.defined]

        summary = f"Total People in All ROIs: {total_people}\n\n"
        summary += f"Defined ROIs: {len(defined_rois)}/{len(self.rois)}\n\n"

        for i, roi in enumerate(self.rois):
            if roi.defined:
                # Get color name
                color_names = ["Yellow", "Magenta", "Cyan", "Green", "Blue",
                               "Orange", "Pink", "Purple", "Teal", "Olive",
                               "Maroon", "Dark Green"]
                color_name = color_names[roi.color_idx % len(color_names)]
                summary += f"{roi.name}: {roi.count} people ({color_name})\n"

        self.summary_text.setText(summary)
        self.total_counter.setText(f"Total People in ROIs: {total_people}")

    @pyqtSlot(np.ndarray, object)
    def process_result(self, frame, results):
        self.current_frame = frame.copy()
        display = frame.copy()

        # Reset all ROI counts and track IDs for current frame
        for roi in self.rois:
            roi.track_ids.clear()
            roi.count = 0

        # Get all defined ROIs
        defined_rois = [roi for roi in self.rois if roi.defined and len(roi.points) >= 3]

        # If no ROIs are defined, don't process any detections
        if not defined_rois:
            # Just draw the ROIs (if any are being drawn)
            for i, roi in enumerate(self.rois):
                is_active = (i == self.active_roi_index and self.drawing_roi)
                roi.draw(display, is_active)

            # Update UI with zero counts
            self.update_roi_list()
            self.update_summary()
            self.show_frame(display)
            self.fps += 1
            return

        # Process detections with tracking, but only for those inside defined ROIs
        if results[0].boxes.is_track:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            annotator = Annotator(display, line_width=2, example=str(names))

            for box, track_id, cls in zip(boxes, track_ids, clss):
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)

                # Check which ROIs contain this person
                containing_rois = []
                for roi in defined_rois:
                    if roi.contains_point((float(cx), float(cy))):
                        containing_rois.append(roi)
                        roi.track_ids.add(track_id)

                # Only process if inside at least one ROI
                if not containing_rois:
                    continue

                # Assign color to track ID if not already assigned (independent of ROI color)
                if track_id not in person_colors:
                    person_colors[track_id] = (random.randint(50, 200),
                                               random.randint(50, 200),
                                               random.randint(50, 200))

                color = person_colors[track_id]

                status = f"ID {track_id}"
                if len(containing_rois) == 1:
                    # Add ROI name to status if in only one ROI
                    status = f"ID {track_id} - {containing_rois[0].name}"
                elif len(containing_rois) > 1:
                    # Indicate multiple ROIs
                    status = f"ID {track_id} - Multiple ROIs"

                # Draw bounding box with tracking
                annotator.box_label(box, status, color=color)

                # Draw tracking trail only if we have history
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                track = track_history[track_id]
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                if len(track) > 30:
                    track.pop(0)

                # Draw tracking trail
                """if len(track) > 1:
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv.polylines(display, [points], isClosed=False, color=color, thickness=2)"""

        # Update ROI counts and track entry/exit events
        for i, roi in enumerate(self.rois):
            if roi.defined:
                roi.count = len(roi.track_ids)

                # Check for entry/exit events
                current_tracks = roi.track_ids
                previous_tracks = self.previous_roi_states[i] if i < len(self.previous_roi_states) else set()

                # Entries: tracks in current but not in previous
                entries = current_tracks - previous_tracks
                if entries:
                    roi.entry_count += len(entries)

                # Exits: tracks in previous but not in current
                exits = previous_tracks - current_tracks
                if exits:
                    roi.exit_count += len(exits)

                # Update previous state
                if i < len(self.previous_roi_states):
                    self.previous_roi_states[i] = current_tracks.copy()

        # Draw all ROIs with transparent fill
        for i, roi in enumerate(self.rois):
            is_active = (i == self.active_roi_index and self.drawing_roi)
            roi.draw(display, is_active)

        # Update UI
        self.update_roi_list()
        self.update_summary()
        self.show_frame(display)
        self.fps += 1

    def show_frame(self, frame):
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(img)
        scaled_pixmap = pixmap.scaled(
            self.video.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video.setPixmap(scaled_pixmap)

    def update_fps(self):
        self.fps_label.setText(f"FPS: {self.fps}")
        self.fps = 0


if __name__ == "__main__":
    app = QApplication(sys.argv)

    model_path = "yolov8n.pt"
    if not os.path.exists(model_path):
        reply = QMessageBox.question(
            None,
            "Model Not Found",
            "yolov8n.pt not found. Would you like to download it?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                from ultralytics import YOLO

                model = YOLO("yolov8n.pt")
                QMessageBox.information(None, "Success", "Model downloaded successfully!")
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Failed to download model: {e}")
                sys.exit(1)

    win = PersonCounterApp()
    win.show()
    sys.exit(app.exec())