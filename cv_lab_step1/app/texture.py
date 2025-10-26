# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple
import numpy as np
import cv2


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

    img_float = gray.astype(np.float32)

    mean_map = cv2.boxFilter(img_float, -1, (window_size, window_size), normalize=True)
    img_squared = img_float ** 2
    mean_squared = cv2.boxFilter(img_squared, -1, (window_size, window_size), normalize=True)
    var_map = mean_squared - (mean_map ** 2)
    var_map = np.maximum(var_map, 0)

    seed_texture = var_map[seed_y, seed_x]
    texture_diff = np.abs(var_map - seed_texture)
    similar_mask = texture_diff <= threshold

    result = np.zeros((h, w, 3), dtype=np.uint8)

    if np.any(similar_mask):
        result[similar_mask] = [0, 255, 0]
        variance_normalized = cv2.normalize(var_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        result[similar_mask, 1] = np.minimum(255, result[similar_mask, 1] + variance_normalized[similar_mask] // 4)

    return result




