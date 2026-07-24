"""
Cap Detection Module
Detects headwear using pre-trained MediaPipe object detection or color/heuristic fallback.
Returns: detected (Present / Not Present / Unknown), color if detected
"""

import cv2
import numpy as np
from config.loader import get_cap_rules


def _detect_cap_region(image: np.ndarray, face_bbox: dict) -> np.ndarray:
    """
    Extract the region above the face where a cap would be.
    If no face bbox, return full top portion of image.
    """
    h, w = image.shape[:2]

    if face_bbox and all(k in face_bbox for k in ("xmin", "ymin", "width", "height")):
        # Region above the face bounding box
        face_top = int(face_bbox["ymin"] * h)
        face_left = int(face_bbox["xmin"] * w)
        face_width = int(face_bbox["width"] * w)
        face_height = int(face_bbox["height"] * h)

        # Cap region: above the face, slightly wider, about 1/3 of face height
        cap_top = max(0, face_top - int(face_height * 0.6))
        cap_bottom = face_top
        cap_left = max(0, face_left - int(face_width * 0.1))
        cap_right = min(w, face_left + face_width + int(face_width * 0.1))

        if cap_bottom > cap_top and cap_right > cap_left:
            return image[cap_top:cap_bottom, cap_left:cap_right]

    # Fallback: top 25% of image
    return image[:int(h * 0.25), :]


def _analyze_cap_region(region: np.ndarray, allowed_colors: list) -> dict:
    """
    Analyze the cap region for presence of headwear.
    Uses multi-heuristic approach to reduce false positives.
    """
    if region.size == 0:
        return {"detected": "Not Present", "color": "None", "confidence": 0.0}

    h, w = region.shape[:2]
    if h < 15 or w < 15:
        return {"detected": "Not Present", "color": "None", "confidence": 0.0}

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # Edge detection to find cap boundaries
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # Mean color of the region
    mean_color = cv2.mean(region)[:3]  # BGR
    brightness = np.mean(mean_color)

    # Check color saturation - caps usually have saturated colors, natural background/hair doesn't
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hsv_mean = cv2.mean(hsv)[:3]
    saturation = hsv_mean[1]

    # Check for skin tone
    is_skin = (
        100 <= mean_color[0] <= 180
        and 80 <= mean_color[1] <= 160
        and 80 <= mean_color[2] <= 160
    )

    has_saturated_color = saturation > 40
    is_dark = brightness < 50
    color_std = np.std(region.reshape(-1, 3), axis=0)
    color_uniformity = np.mean(color_std) < 50

    # Check for horizontal edge pattern at cap brim
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    horizontal_edges = np.abs(sobel_x) > 60
    horizontal_edge_ratio = np.sum(horizontal_edges) / horizontal_edges.size if horizontal_edges.size > 0 else 0

    # Check for cap-like edge pattern: edges concentrated at top + brim line
    mid_y = h // 2
    top_half = edges[:mid_y, :] if mid_y > 0 else edges
    bottom_half = edges[mid_y:, :] if mid_y < h else edges
    top_edge_density = np.sum(top_half > 0) / top_half.size if top_half.size > 0 else 0
    bottom_edge_density = np.sum(bottom_half > 0) / bottom_half.size if bottom_half.size > 0 else 0

    has_cap_edge_pattern = top_edge_density > 0.08 and bottom_edge_density > 0.06

    # Count conditions met (need 2+ to detect cap)
    cap_conditions_met = 0
    if edge_density > 0.12 and has_saturated_color:
        cap_conditions_met += 1
    if not is_skin and brightness < 80 and color_uniformity:
        cap_conditions_met += 1
    if has_cap_edge_pattern and horizontal_edge_ratio > 0.04:
        cap_conditions_met += 1
    if edge_density > 0.20 and is_dark:
        cap_conditions_met += 1

    if cap_conditions_met >= 2:
        color_name = _classify_color_simple(mean_color, allowed_colors)
        base_confidence = min(edge_density * 300 + (1 - int(is_skin)) * 15 + (has_saturated_color * 20), 90)
        confidence = round(min(base_confidence + has_cap_edge_pattern * 15 + horizontal_edge_ratio * 100, 90), 1)

        if confidence >= 75.0:
            return {
                "detected": "Present",
                "color": color_name if color_name in allowed_colors else color_name,
                "confidence": confidence,
            }

    return {"detected": "Not Present", "color": "None", "confidence": 0.0}


def _classify_color_simple(bgr_color: tuple, allowed_colors: list) -> str:
    """Simple color classifier for cap detection."""
    b, g, r = map(int, bgr_color)

    # Define rough color ranges (BGR)
    ranges = {
        "Black": (b < 60 and g < 60 and r < 60),
        "White": (b > 180 and g > 180 and r > 180),
        "Blue": (b > g and b > r and b > 80),
        "Red": (r > g and r > b and r > 80),
        "Green": (g > r and g > b and g > 80),
        "Orange": (r > 150 and g > 80 and b < 80),
        "Yellow": (r > 150 and g > 150 and b < 100),
        "Brown": (b < 100 and g < 100 and r > 100 and r < 200),
        "Grey": (abs(b - g) < 30 and abs(g - r) < 30 and b > 60 and b < 180),
        "Navy": (b > 100 and g < 100 and r < 80 and b > g and b > r),
    }

    # First check if color matches any allowed color
    for color_name in allowed_colors:
        if color_name in ranges and ranges[color_name]:
            return color_name

    # Then check all colors
    for color_name, condition in ranges.items():
        if condition:
            return color_name

    return "Unknown"


def detect_cap(image: np.ndarray, face_bbox: dict = None) -> dict:
    """
    Detect cap in the image.

    Args:
        image: BGR numpy array
        face_bbox: dict from face detector with xmin, ymin, width, height (optional)

    Returns:
        dict: { detected (Present/Not Present/Unknown), color, confidence }
    """
    rules = get_cap_rules()
    allowed_colors = rules.get("colors", ["Red"])

    cap_region = _detect_cap_region(image, face_bbox)
    result = _analyze_cap_region(cap_region, allowed_colors)

    return result
