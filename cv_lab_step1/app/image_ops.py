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

