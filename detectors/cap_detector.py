"""
Cap Detection Module
Hybrid approach: YOLOv8 (best.pt) for cap detection + HSV color analysis on cropped cap region.
Returns: detected (Present / Not Present), color if detected, confidence
"""

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from config.loader import get_cap_rules


# ── Constants ──────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5  # YOLO confidence threshold for cap detection

# Path to YOLO cap detection model
_DETECTOR_DIR = Path(__file__).resolve().parent
_MODEL_PATH = str(_DETECTOR_DIR.parent / "models" / "best.pt")

# Global model cache (load once)
_yolo_model = None


def _get_model():
    """Load and cache the YOLO model."""
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(_MODEL_PATH)
    return _yolo_model


# ── HSV Color Ranges (from Cap-detection repo) ─────────────────────────────
_COLOR_RANGES = {
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


def _detect_dominant_color(cropped_img: np.ndarray) -> tuple:
    """
    Detect the dominant color from a cropped cap region using HSV analysis.

    Args:
        cropped_img: BGR numpy array of the cap region

    Returns:
        (color_name: str, confidence: float) where confidence is 0-100
    """
    if cropped_img.size == 0:
        return "Unknown", 0.0

    hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    total_pixels = cropped_img.shape[0] * cropped_img.shape[1]

    best_color = "Unknown"
    best_ratio = 0.0

    for color_name, ranges in _COLOR_RANGES.items():
        combined_mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.inRange(hsv_img, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        color_pixels = np.count_nonzero(combined_mask)
        ratio = color_pixels / max(total_pixels, 1)
        if ratio > best_ratio:
            best_ratio = ratio
            best_color = color_name

    # Fallback: if no color matched well, use mean hue of saturated pixels
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
            best_ratio = 0.5  # moderate confidence for fallback

    # Convert ratio to 0-100 scale
    confidence = round(best_ratio * 100, 1)
    return best_color, confidence


def detect_cap(image: np.ndarray, face_bbox: dict = None) -> dict:
    """
    Detect cap in the image using YOLOv8 (best.pt) + HSV color analysis.

    Args:
        image: BGR numpy array (H, W, 3)
        face_bbox: dict from face detector (kept for API compatibility, not used by YOLO)

    Returns:
        dict: {
            "detected": "Present" or "Not Present",
            "color": detected color name or "Unknown",
            "confidence": detection confidence (0-100)
        }
    """
    rules = get_cap_rules()
    allowed_colors = rules.get("colors", ["Red"])

    # Load YOLO model
    model = _get_model()

    # Run YOLO inference
    results = model(image, verbose=False)

    cap_detected = False
    max_confidence = 0.0
    cap_color = "Unknown"
    color_confidence = 0.0

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0].item())
            if confidence > CONFIDENCE_THRESHOLD:
                class_id = int(box.cls[0].item())
                if class_id == 0:  # class 0 = 'cap'
                    cap_detected = True
                    if confidence > max_confidence:
                        max_confidence = confidence
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        x1, y1 = max(0, int(x1)), max(0, int(y1))
                        x2, y2 = min(image.shape[1], int(x2)), min(image.shape[0], int(y2))
                        if x2 > x1 and y2 > y1:
                            cropped_cap = image[y1:y2, x1:x2]
                            cap_color, color_confidence = _detect_dominant_color(cropped_cap)

    if cap_detected:
        # Convert YOLO confidence (0-1) to percentage (0-100)
        conf_pct = round(max_confidence * 100, 1)
        return {
            "detected": "Present",
            "color": cap_color,
            "confidence": conf_pct,
        }
    else:
        return {
            "detected": "Not Present",
            "color": "Unknown",
            "confidence": 0.0,
        }
