#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify image loading functionality
"""
import sys
import os
import numpy as np
import cv2

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'cv_lab_step1', 'app'))

from image_ops import read_color_bgr
from qt_image import cv_to_qpixmap

def test_image_loading(image_path):
    """Test image loading and conversion"""
    print(f"Testing image loading: {image_path}")
    
    try:
        # Test OpenCV loading
        print("1. Testing OpenCV loading...")
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            print("   ERROR: OpenCV failed to load image")
            return False
        print(f"   SUCCESS: OpenCV loaded image with shape {img_cv.shape}")
        
        # Test our read_color_bgr function
        print("2. Testing read_color_bgr function...")
        img_bgr = read_color_bgr(image_path)
        print(f"   SUCCESS: read_color_bgr loaded image with shape {img_bgr.shape}")
        
        # Test conversion to QPixmap
        print("3. Testing conversion to QPixmap...")
        pixmap = cv_to_qpixmap(img_bgr)
        if pixmap.isNull():
            print("   ERROR: QPixmap is null")
            return False
        print(f"   SUCCESS: QPixmap created with size {pixmap.size().width()}x{pixmap.size().height()}")
        
        return True
        
    except Exception as e:
        print(f"   ERROR: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            success = test_image_loading(image_path)
            if success:
                print("\n✅ All tests passed!")
            else:
                print("\n❌ Some tests failed!")
        else:
            print(f"Image file not found: {image_path}")
    else:
        print("Usage: python test_image_loading.py <image_path>")
        print("Example: python test_image_loading.py test_image.jpg")



