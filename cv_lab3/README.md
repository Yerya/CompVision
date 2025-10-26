# Clock Time Detector

A desktop application for detecting time from clock images using computer vision techniques.

## Features

- Load clock images from file
- Detect clock face using Hough circle detection
- Identify clock hands using line detection
- Display detected time (hours and minutes)
- Minimalistic and user-friendly interface

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- PySide6

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python app/main.py
```

1. Click "Load Image" to select a clock image
2. Click "Detect Time" to analyze the image and detect the time
3. The detected time will be displayed on screen

## Algorithm

The application uses the following approach:

1. **Preprocessing**: Convert to grayscale and enhance contrast
2. **Clock Face Detection**: Use Hough circle detection to find the clock face
3. **Hand Detection**: Apply edge detection and line detection within the clock face
4. **Line Clustering**: Group similar lines to identify individual hands
5. **Time Calculation**: Convert hand angles to clock time

## Supported Image Formats

- PNG
- JPEG/JPG
- BMP
- TIFF

## Notes

- Works best with clear clock images
- Handles clocks with additional decorative elements
- Detects both hour and minute hands
- Visualizes detection results with colored overlays
