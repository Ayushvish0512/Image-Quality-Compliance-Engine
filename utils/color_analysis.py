"""
Shared HSV Color Analysis Utility
===================================
Single source of truth for color detection via HSV ranges.
Used by cap_detector.py, classification_pipeline.py, and test files.

Exports:
    COLOR_RANGES: dict mapping color names to HSV lower/upper bounds
    detect_dominant_color(cropped_img) -> (color_name, confidence)
"""

import cv2
import numpy as np

# ── HSV Color Ranges ───────────────────────────────────────────────────────
# Single source of truth — edit here to update color detection everywhere
COLOR_RANGES = {
    "Red": [
        (np.array([0, 100, 50], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8)),
        (np.array([160, 100, 50], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8))
    ],
    "Blue": [
        (np.array([100, 100, 50], dtype=np.uint8), np.array([130, 255, 255], dtype=np.uint8))
    ],
    "Green": [
        (np.array([40, 100, 50], dtype=np.uint8), np.array([80, 255, 255], dtype=np.uint8))
    ],
    "Yellow": [
        (np.array([20, 100, 100], dtype=np.uint8), np.array([35, 255, 255], dtype=np.uint8))
    ],
    "Orange": [
        (np.array([10, 100, 100], dtype=np.uint8), np.array([20, 255, 255], dtype=np.uint8))
    ],
    "Purple": [
        (np.array([130, 50, 50], dtype=np.uint8), np.array([160, 255, 255], dtype=np.uint8))
    ],
    "Pink": [
        (np.array([150, 50, 100], dtype=np.uint8), np.array([170, 255, 255], dtype=np.uint8))
    ],
    "Brown": [
        (np.array([5, 50, 50], dtype=np.uint8), np.array([20, 200, 150], dtype=np.uint8))
    ],
    "White": [
        (np.array([0, 0, 200], dtype=np.uint8), np.array([179, 30, 255], dtype=np.uint8))
    ],
    "Gray": [
        (np.array([0, 0, 100], dtype=np.uint8), np.array([179, 40, 200], dtype=np.uint8))
    ],
    "Black": [
        (np.array([0, 0, 0], dtype=np.uint8), np.array([179, 255, 50], dtype=np.uint8))
    ]
}


def detect_dominant_color(cropped_img: np.ndarray) -> tuple:
    """
    Detect dominant color from a cropped region using HSV analysis.

    Args:
        cropped_img: BGR numpy array of the region to analyze

    Returns:
        (color_name: str, confidence: float) where confidence is 0-100
    """
    if cropped_img.size == 0:
        return "Unknown", 0.0

    hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    total_pixels = cropped_img.shape[0] * cropped_img.shape[1]

    best_color = "Unknown"
    best_ratio = 0.0

    for color_name, ranges in COLOR_RANGES.items():
        combined_mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.inRange(hsv_img, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        color_pixels = np.count_nonzero(combined_mask)
        ratio = color_pixels / max(total_pixels, 1)
        if ratio > best_ratio:
            best_ratio = ratio
            best_color = color_name

    # Fallback: use mean hue of saturated pixels
    if best_ratio < 0.15:
        mask_non_black = cv2.inRange(hsv_img, np.array([0, 30, 30]), np.array([179, 255, 255]))
        if np.count_nonzero(mask_non_black) > total_pixels * 0.05:
            mean_hue = np.mean(hsv_img[:, :, 0][mask_non_black > 0])
            if mean_hue < 10 or mean_hue > 160:
                best_color = "Red"
            elif mean_hue < 25:
                best_color = "Orange"
            elif mean_hue < 35:
                best_color = "Yellow"
            elif mean_hue < 85:
                best_color = "Green"
            elif mean_hue < 130:
                best_color = "Blue"
            else:
                best_color = "Purple"
            best_ratio = 0.5

    confidence = round(best_ratio * 100, 1)
    return best_color, confidence
