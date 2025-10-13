# -*- coding: utf-8 -*-
"""
Image operations for Step 1:
- Read color image
- Convert to grayscale (two ways):
  1) Channel averaging (manual RGB mean)
  2) HSV model (use the V channel as brightness)
"""
from __future__ import annotations
from typing import Tuple
import numpy as np
import cv2

def read_color_bgr(path: str) -> np.ndarray:

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img

def to_gray_mean(bgr: np.ndarray) -> np.ndarray:
 
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("to_gray_mean expects a BGR image with shape (H, W, 3).")
    # Convert to float for precise mean, then back to uint8
    bgr_f = bgr.astype(np.float32)
    # BGR -> average -> gray
    gray = bgr_f.mean(axis=2)
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    return gray_u8

def to_gray_hsv_value(bgr: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to grayscale using the HSV model.
    We take the V (value/brightness) channel as the grayscale intensity.
    Returns uint8 grayscale image.
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("to_gray_hsv_value expects a BGR image with shape (H, W, 3).")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]  # Value channel
    return v.copy()


# --- Binarization ---
def threshold_fixed(gray: np.ndarray, threshold: int = 128, invert: bool = False) -> np.ndarray:
    """
    Fixed threshold binarization for uint8 grayscale images.
    Returns a binary uint8 image with values {0, 255}.
    """
    if gray.ndim != 2:
        raise ValueError("threshold_fixed expects a grayscale image with shape (H, W).")
    thr = int(max(0, min(255, threshold)))
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, bin_img = cv2.threshold(gray, thr, 255, flag)
    return bin_img


def threshold_otsu(gray: np.ndarray, invert: bool = False) -> Tuple[np.ndarray, float]:
    """
    Otsu thresholding for uint8 grayscale images.
    Returns (binary_image, chosen_threshold).
    """
    if gray.ndim != 2:
        raise ValueError("threshold_otsu expects a grayscale image with shape (H, W).")
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    thr, bin_img = cv2.threshold(gray, 0, 255, flag | cv2.THRESH_OTSU)
    return bin_img, float(thr)


# --- Histogram-based transforms (grayscale) ---
def hist_normalize(gray: np.ndarray) -> np.ndarray:
    """
    Normalize intensities to [0, 255] using min-max normalization.
    """
    if gray.ndim != 2:
        raise ValueError("hist_normalize expects a grayscale image with shape (H, W).")
    norm = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def hist_equalize(gray: np.ndarray) -> np.ndarray:
    """
    Histogram equalization (global) for grayscale images.
    """
    if gray.ndim != 2:
        raise ValueError("hist_equalize expects a grayscale image with shape (H, W).")
    return cv2.equalizeHist(gray)


def hist_contrast_stretch(gray: np.ndarray) -> np.ndarray:
    """
    Linear contrast stretching using image min/max.
    If all pixels are equal, returns a zero image.
    """
    if gray.ndim != 2:
        raise ValueError("hist_contrast_stretch expects a grayscale image with shape (H, W).")
    min_v = int(gray.min())
    max_v = int(gray.max())
    if max_v <= min_v:
        return np.zeros_like(gray)
    scale = 255.0 / float(max_v - min_v)
    stretched = ((gray.astype(np.float32) - float(min_v)) * scale)
    return np.clip(stretched, 0, 255).astype(np.uint8)


# --- Convolutions / Filters ---
def gaussian_blur(gray: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Gaussian blur for grayscale images.
    ksize must be odd and >= 3.
    """
    if gray.ndim != 2:
        raise ValueError("gaussian_blur expects a grayscale image with shape (H, W).")
    k = int(max(3, ksize))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(gray, (k, k), sigmaX=float(sigma), sigmaY=float(sigma))


def laplacian_sharpen(gray: np.ndarray, alpha: float = 1.0, ksize: int = 3) -> np.ndarray:
    """
    Unsharp masking style: sharpened = gray - alpha * Laplacian(gray).
    """
    if gray.ndim != 2:
        raise ValueError("laplacian_sharpen expects a grayscale image with shape (H, W).")
    k = int(max(1, ksize))
    if k % 2 == 0:
        k += 1
    lap = cv2.Laplacian(gray, ddepth=cv2.CV_16S, ksize=k)
    lap_u8 = cv2.convertScaleAbs(lap)
    sharpen = cv2.subtract(gray, cv2.convertScaleAbs(alpha * lap_u8))
    return sharpen


def sobel_edges(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Edge magnitude via Sobel (combine X and Y).
    """
    if gray.ndim != 2:
        raise ValueError("sobel_edges expects a grayscale image with shape (H, W).")
    k = int(max(1, ksize))
    if k % 2 == 0:
        k += 1
    gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=k)
    gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=k)
    ax = cv2.convertScaleAbs(gx)
    ay = cv2.convertScaleAbs(gy)
    mag = cv2.addWeighted(ax, 0.5, ay, 0.5, 0)
    return mag


# --- Geometry ---
def cyclic_shift(img: np.ndarray, dx: int = 0, dy: int = 0) -> np.ndarray:
    """
    Cyclic (wrap-around) shift by dx (cols) and dy (rows).
    Works for gray or color images.
    """
    if img.ndim not in (2, 3):
        raise ValueError("cyclic_shift expects (H,W) or (H,W,3) uint8 image.")
    h, w = img.shape[:2]
    dx_mod = int(dx) % w if w > 0 else 0
    dy_mod = int(dy) % h if h > 0 else 0
    return np.roll(np.roll(img, shift=dy_mod, axis=0), shift=dx_mod, axis=1)


def rotate_about_center(img: np.ndarray, angle_deg: float = 15.0, center: Tuple[float, float] | None = None, scale: float = 1.0) -> np.ndarray:
    """
    Rotate an image around a given center (default: image center).
    Keeps original canvas size (border filled with black).
    """
    if img.ndim not in (2, 3):
        raise ValueError("rotate_about_center expects (H,W) or (H,W,3) uint8 image.")
    h, w = img.shape[:2]
    if center is None:
        center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, float(angle_deg), float(scale))
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rotated


# --- Lab 2: Feature Detection ---

def _calculate_adaptive_hough_params(gray: np.ndarray) -> dict:
    """
    Calculate adaptive Hough parameters based on image characteristics.
    Returns dictionary with adaptive parameters for both lines and circles.
    """
    h, w = gray.shape[:2]
    diagonal = int(np.sqrt(h*h + w*w))
    
    # Adaptive parameters based on image size
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
        'circle_max_radius': min(200, int(diagonal * 0.3))
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
    
    # Use adaptive parameters if requested
    if adaptive:
        adaptive_params = _calculate_adaptive_hough_params(gray)
        threshold = adaptive_params['line_threshold']
        min_line_length = adaptive_params['line_min_length']
        max_line_gap = adaptive_params['line_max_gap']
    
    # Detect edges first
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Hough line detection
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, 
                           minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    # Draw lines on original image
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
    
    # Use adaptive parameters if requested
    if adaptive:
        adaptive_params = _calculate_adaptive_hough_params(gray)
        min_dist = adaptive_params['circle_min_dist']
        param1 = adaptive_params['circle_param1']
        param2 = adaptive_params['circle_param2']
        min_radius = adaptive_params['circle_min_radius']
        max_radius = adaptive_params['circle_max_radius']
    
    # Detect circles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, min_dist,
                              param1=param1, param2=param2,
                              minRadius=min_radius, maxRadius=max_radius)
    
    # Draw circles on original image
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(result, (x, y), r, (0, 255, 0), 2)
            cv2.circle(result, (x, y), 2, (0, 0, 255), 3)
    
    return result, circles if circles is not None else np.array([])


def local_statistics(gray: np.ndarray, window_size: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute local statistical features: mean, variance, skewness, kurtosis.
    Returns (mean_map, variance_map, skewness_map, kurtosis_map).
    """
    if gray.ndim != 2:
        raise ValueError("local_statistics expects a grayscale image with shape (H, W).")
    
    # Convert to float for calculations
    img_float = gray.astype(np.float32)
    h, w = img_float.shape
    
    # Use OpenCV's boxFilter for efficient mean computation
    mean_map = cv2.boxFilter(img_float, -1, (window_size, window_size), normalize=True)
    
    # Compute variance using E[X^2] - E[X]^2
    img_squared = img_float ** 2
    mean_squared = cv2.boxFilter(img_squared, -1, (window_size, window_size), normalize=True)
    var_map = mean_squared - (mean_map ** 2)
    var_map = np.maximum(var_map, 0)  # Ensure non-negative
    
    # For skewness and kurtosis, use a simplified approach
    # Compute local standard deviation
    std_map = np.sqrt(var_map)
    
    # Simplified skewness: measure of asymmetry
    # Use difference between mean and median as proxy
    median_map = cv2.medianBlur(img_float.astype(np.uint8), window_size).astype(np.float32)
    skew_map = (mean_map - median_map) / (std_map + 1e-6)
    
    # Simplified kurtosis: measure of peakedness
    # Use local contrast as proxy
    laplacian = cv2.Laplacian(img_float, cv2.CV_32F, ksize=3)
    kurt_map = np.abs(laplacian) / (std_map + 1e-6)
    
    # Normalize to 0-255 range
    mean_norm = cv2.normalize(mean_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    var_norm = cv2.normalize(var_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    skew_norm = cv2.normalize(skew_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    kurt_norm = cv2.normalize(kurt_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return mean_norm, var_norm, skew_norm, kurt_norm


def texture_segmentation(gray: np.ndarray, seed_point: Tuple[int, int], 
                        threshold: float = 30.0, window_size: int = 15) -> np.ndarray:
    """
    Texture-based region growing segmentation starting from seed point.
    Uses local variance as texture feature.
    Returns segmented image with different regions colored differently.
    """
    if gray.ndim != 2:
        raise ValueError("texture_segmentation expects a grayscale image with shape (H, W).")
    
    h, w = gray.shape
    seed_y, seed_x = seed_point
    
    if not (0 <= seed_x < w and 0 <= seed_y < h):
        raise ValueError(f"Seed point {seed_point} is outside image bounds {(h, w)}")
    
    # Use efficient OpenCV operations for local variance
    img_float = gray.astype(np.float32)
    
    # Compute local mean and variance efficiently
    mean_map = cv2.boxFilter(img_float, -1, (window_size, window_size), normalize=True)
    img_squared = img_float ** 2
    mean_squared = cv2.boxFilter(img_squared, -1, (window_size, window_size), normalize=True)
    var_map = mean_squared - (mean_map ** 2)
    var_map = np.maximum(var_map, 0)
    
    # Get seed texture feature
    seed_texture = var_map[seed_y, seed_x]
    
    # Simple threshold-based segmentation
    # Find pixels with similar texture variance
    texture_diff = np.abs(var_map - seed_texture)
    similar_mask = texture_diff <= threshold
    
    # Create colored result
    result = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Color the segmented region
    if np.any(similar_mask):
        # Use a fixed color for the segmented region
        result[similar_mask] = [0, 255, 0]  # Green for segmented region
        
        # Add some variation based on local variance
        variance_normalized = cv2.normalize(var_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        result[similar_mask, 1] = np.minimum(255, result[similar_mask, 1] + variance_normalized[similar_mask] // 4)
    
    return result