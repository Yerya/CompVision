# -*- coding: utf-8 -*-
"""
Qt image helpers.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from PySide6.QtGui import QImage, QPixmap

def cv_to_qimage(img: np.ndarray) -> QImage:
    """
    Convert a NumPy image (BGR or gray, dtype uint8) to QImage.
    - BGR (OpenCV default) -> converted to RGB for Qt.
    - Grayscale -> QImage.Format_Grayscale8.
    The returned QImage owns its data (copy()) to avoid dangling buffer issues.
    """
    if img.ndim == 2:
        h, w = img.shape
        bytes_per_line = img.strides[0]
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        return qimg.copy()
    elif img.ndim == 3 and img.shape[2] == 3:
        # Convert BGR (OpenCV) to RGB (Qt)
        rgb = img[:, :, ::-1].copy()
        h, w, _ = rgb.shape
        bytes_per_line = rgb.strides[0]
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return qimg.copy()
    else:
        raise ValueError("cv_to_qimage expects uint8 gray or BGR images with shape (H,W) or (H,W,3).")

def cv_to_qpixmap(img: np.ndarray) -> QPixmap:
    """
    Convenience wrapper to get a QPixmap from a NumPy image.
    """
    return QPixmap.fromImage(cv_to_qimage(img))
