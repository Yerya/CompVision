#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify all imports work correctly
"""
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'cv_lab_step1', 'app'))

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        print("1. Testing basic imports...")
        import numpy as np
        import cv2
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QImage, QPixmap
        from PySide6.QtWidgets import QApplication
        print("   ✅ Basic imports successful")
        
        print("2. Testing app modules...")
        from image_ops import read_color_bgr, to_gray_mean, to_gray_hsv_value
        from qt_image import cv_to_qpixmap
        from hough import hough_lines, hough_circles
        from stats import local_statistics
        from texture import texture_segmentation
        print("   ✅ App modules imported successfully")
        
        print("3. Testing Lab 3 modules...")
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), 'cv_lab_step3', 'app'))
            from clock_detection import detect_clock_time
            print("   ✅ Lab 3 modules imported successfully")
        except ImportError as e:
            print(f"   ⚠️  Lab 3 modules not available: {e}")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    if not success:
        sys.exit(1)



