# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple
import numpy as np
import cv2


def local_statistics(gray: np.ndarray, window_size: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute local statistical features: mean, variance, skewness, kurtosis.
    Returns (mean_map, variance_map, skewness_map, kurtosis_map).
    """
    if gray.ndim != 2:
        raise ValueError("local_statistics expects a grayscale image with shape (H, W).")

    img_float = gray.astype(np.float32)

    mean_map = cv2.boxFilter(img_float, -1, (window_size, window_size), normalize=True)

    img_squared = img_float ** 2
    mean_squared = cv2.boxFilter(img_squared, -1, (window_size, window_size), normalize=True)
    var_map = mean_squared - (mean_map ** 2)
    var_map = np.maximum(var_map, 0)

    std_map = np.sqrt(var_map)
    median_map = cv2.medianBlur(img_float.astype(np.uint8), window_size).astype(np.float32)
    skew_map = (mean_map - median_map) / (std_map + 1e-6)

    laplacian = cv2.Laplacian(img_float, cv2.CV_32F, ksize=3)
    kurt_map = np.abs(laplacian) / (std_map + 1e-6)

    mean_norm = cv2.normalize(mean_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    var_norm = cv2.normalize(var_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    skew_norm = cv2.normalize(skew_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    kurt_norm = cv2.normalize(kurt_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return mean_norm, var_norm, skew_norm, kurt_norm



