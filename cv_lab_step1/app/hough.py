# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple
import numpy as np
import cv2


def calculate_adaptive_hough_params(gray: np.ndarray) -> dict:
    """
    Calculate adaptive Hough parameters based on image characteristics.
    Returns dictionary with adaptive parameters for both lines and circles.
    """
    h, w = gray.shape[:2]
    diagonal = int(np.sqrt(h * h + w * w))

    params = {
        # Line parameters
        'line_threshold': max(50, min(200, int(diagonal * 0.1))),
        'line_min_length': max(20, min(100, int(diagonal * 0.05))),
        'line_max_gap': max(5, min(30, int(diagonal * 0.02))),

        # Circle parameters
        'circle_dp': 1.0,
        'circle_min_dist': max(20, min(150, int(diagonal * 0.08))),
        'circle_param1': max(30, min(150, int(diagonal * 0.15))),
        'circle_param2': max(20, min(100, int(diagonal * 0.1))),
        'circle_min_radius': max(5, int(diagonal * 0.02)),
        'circle_max_radius': min(200, int(diagonal * 0.3)),
    }

    return params


def hough_lines(gray: np.ndarray, rho: float = 1.0, theta: float = np.pi/180, threshold: int = 100,
                min_line_length: int = 50, max_line_gap: int = 10, adaptive: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect lines using Hough Line Transform.
    If adaptive=True, automatically calculates optimal parameters based on image size.
    Returns (image_with_lines, lines_array).
    """
    if gray.ndim != 2:
        raise ValueError("hough_lines expects a grayscale image with shape (H, W).")

    if adaptive:
        adaptive_params = calculate_adaptive_hough_params(gray)
        threshold = adaptive_params['line_threshold']
        min_line_length = adaptive_params['line_min_length']
        max_line_gap = adaptive_params['line_max_gap']

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho, theta, threshold,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)

    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return result, lines if lines is not None else np.array([])


def hough_circles(gray: np.ndarray, dp: float = 1.0, min_dist: float = 50.0,
                  param1: float = 50.0, param2: float = 30.0, min_radius: int = 0,
                  max_radius: int = 0, adaptive: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect circles using Hough Circle Transform.
    If adaptive=True, automatically calculates optimal parameters based on image size.
    Returns (image_with_circles, circles_array).
    """
    if gray.ndim != 2:
        raise ValueError("hough_circles expects a grayscale image with shape (H, W).")

    if adaptive:
        adaptive_params = calculate_adaptive_hough_params(gray)
        min_dist = adaptive_params['circle_min_dist']
        param1 = adaptive_params['circle_param1']
        param2 = adaptive_params['circle_param2']
        min_radius = adaptive_params['circle_min_radius']
        max_radius = adaptive_params['circle_max_radius']

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, min_dist,
                               param1=param1, param2=param2,
                               minRadius=min_radius, maxRadius=max_radius)

    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(result, (x, y), r, (0, 255, 0), 2)
            cv2.circle(result, (x, y), 2, (0, 0, 255), 3)

    return result, circles if circles is not None else np.array([])



