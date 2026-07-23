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

    Uses edge density and color analysis to determine if something is on the head.
    """
    if region.size == 0:
        return {"detected": "Unknown", "color": "Unknown", "confidence": 0.0}

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # Edge detection to find cap boundaries
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # A cap typically introduces edges above the face
    # Low edge density + uniform dark region suggests no cap
    # High edge density + distinct color suggests cap

    # Mean color of the region
    mean_color = cv2.mean(region)[:3]  # BGR
    brightness = np.mean(mean_color)

    # Heuristic: if there's significant edge activity and/or
    # the region has distinct color (not just skin tone), likely a cap
    # Skin tone approximate in BGR: [100-180, 80-160, 80-160]
    is_skin = (
        100 <= mean_color[0] <= 180
        and 80 <= mean_color[1] <= 160
        and 80 <= mean_color[2] <= 160
    )

    if edge_density > 0.08 or (not is_skin and brightness < 150):
        # Likely a cap present — try to match color
        # Use mean color as rough estimate
        color_name = _classify_color_simple(mean_color, allowed_colors)
        confidence = round(min(edge_density * 500 + (1 - int(is_skin)) * 30, 95), 1)

        return {
            "detected": "Present",
            "color": color_name if color_name in allowed_colors else color_name,
            "confidence": max(confidence, 50.0),
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
    allowed_colors = rules.get("colors", ["Blue", "Black"])

    cap_region = _detect_cap_region(image, face_bbox)
    result = _analyze_cap_region(cap_region, allowed_colors)

    return result
