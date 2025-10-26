"""
Clock Detection Application
A minimalistic desktop application for detecting time from clock images.
"""
import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox, QFrame
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap, QImage, QFont

from clock_detector import detect_clock_time


class ClockDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clock Time Detector")
        self.setGeometry(100, 100, 800, 600)
        self.setMinimumSize(600, 400)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QLabel {
                color: #333;
            }
            QFrame {
                border: 2px solid #ddd;
                border-radius: 8px;
                background-color: white;
            }
        """)
        
        self.current_image = None
        self.detected_time = None
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QLabel("Clock Time Detector")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setStyleSheet("color: #2c3e50; margin: 10px;")
        main_layout.addWidget(title_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        button_layout.addWidget(self.load_button)
        
        self.detect_button = QPushButton("Detect Time")
        self.detect_button.clicked.connect(self.detect_time)
        self.detect_button.setEnabled(False)
        button_layout.addWidget(self.detect_button)
        
        button_layout.addStretch()
        main_layout.addLayout(button_layout)
        
        # Image display area
        self.image_frame = QFrame()
        self.image_frame.setFixedHeight(400)
        self.image_frame.setStyleSheet("""
            QFrame {
                border: 2px dashed #ccc;
                background-color: #fafafa;
            }
        """)
        
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                color: #999;
                font-size: 16px;
            }
        """)
        
        image_layout = QVBoxLayout(self.image_frame)
        image_layout.addWidget(self.image_label)
        
        main_layout.addWidget(self.image_frame)
        
        # Time display area
        time_frame = QFrame()
        time_frame.setFixedHeight(80)
        time_frame.setStyleSheet("""
            QFrame {
                background-color: #e8f5e8;
                border: 2px solid #4CAF50;
            }
        """)
        
        time_layout = QVBoxLayout(time_frame)
        
        self.time_label = QLabel("Detected Time: --:--")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.time_label.setStyleSheet("color: #2c3e50;")
        time_layout.addWidget(self.time_label)
        
        main_layout.addWidget(time_frame)
    
    def load_image(self):
        """Load an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Clock Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        if file_path:
            try:
                # Load image using OpenCV
                self.current_image = cv2.imread(file_path)
                
                if self.current_image is None:
                    QMessageBox.critical(self, "Error", "Failed to load image file.")
                    return
                
                # Display image
                self.display_image(self.current_image)
                
                # Enable detect button
                self.detect_button.setEnabled(True)
                
                # Reset time display
                self.time_label.setText("Detected Time: --:--")
                self.detected_time = None
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading image: {str(e)}")
    
    def display_image(self, img):
        """Display image in the image label."""
        if img is None:
            return
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        # Create QImage
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Create QPixmap and scale to fit
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.image_frame.size() - QSize(20, 20),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setText("")
    
    def detect_time(self):
        """Detect time from the loaded image."""
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return
        
        try:
            # Detect clock time
            hour, minute, result_image = detect_clock_time(self.current_image)
            
            if hour is not None and minute is not None:
                # Update time display
                time_text = f"Detected Time: {hour:02d}:{minute:02d}"
                self.time_label.setText(time_text)
                self.detected_time = (hour, minute)
                
                # Display result image with detections
                self.display_image(result_image)
                
                # Update window title
                self.setWindowTitle(f"Clock Time Detector - {hour:02d}:{minute:02d}")
                
            else:
                QMessageBox.information(
                    self,
                    "Detection Result",
                    "No clock detected in the image. Please try with a different image."
                )
                self.time_label.setText("Detected Time: --:--")
                self.detected_time = None
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during detection: {str(e)}")
            self.time_label.setText("Detected Time: --:--")
            self.detected_time = None


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Clock Time Detector")
    app.setApplicationVersion("1.0")
    
    # Set application style
    app.setStyle("Fusion")
    
    window = ClockDetectionApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
