"""
Accessory Detection Module
Detects configurable accessories (apron, gloves, mask, hairnet, id_card, jacket).
Uses heuristic color/region analysis per accessory type.
Supports dynamic accessory list from YAML config — no code changes needed.
"""

import cv2
import numpy as np
from config.loader import get_accessory_rules


def _extract_body_region(image: np.ndarray, region_type: str) -> np.ndarray:
    """
    Extract specific body region for accessory detection.
    Uses image regions as rough approximations (no pose model needed).
    """
    h, w = image.shape[:2]

    regions = {
        "upper_body": image[:int(h * 0.6), :],                    # top 60%
        "torso": image[int(h * 0.15):int(h * 0.55), int(w * 0.2):int(w * 0.8)],
        "face_region": image[:int(h * 0.3), int(w * 0.2):int(w * 0.8)],
        "neck_shoulders": image[int(h * 0.1):int(h * 0.3), :],
    }

    return regions.get(region_type, image)


def _detect_apron(roi: np.ndarray) -> dict:
    """
    Detect apron: look for a solid-colored rectangular covering over torso.
    Aprons are typically darker, uniform color covering the front.
    """
    if roi.size == 0:
        return {"detected": False, "confidence": 0.0}

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Aprons usually create a distinct rectangle in the center of torso
    # Use edge detection + contour analysis
    edges = cv2.Canny(gray, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"detected": False, "confidence": 0.0}

    # Look for large rectangular contour in center
    h, w = roi.shape[:2]
    center_x, center_y = w // 2, h // 2
    best_contour = None
    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (roi.size * 0.1):  # at least 10% of roi
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect_ratio = cw / ch if ch > 0 else 0

        # Apron aspect ratio typically between 0.5 and 1.5
        if 0.4 < aspect_ratio < 2.0:
            # Check if centered
            cx = x + cw // 2
            cy = y + ch // 2
            dist_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            center_proximity = 1 - (dist_from_center / max(w, h))
            area_ratio = area / roi.size
            score = area_ratio * 0.5 + center_proximity * 0.5

            if score > best_score:
                best_score = score
                best_contour = cnt

    if best_contour and best_score > 0.15:
        return {"detected": True, "confidence": round(min(best_score * 100, 95), 1)}
    return {"detected": False, "confidence": 0.0}


def _detect_mask(image: np.ndarray) -> dict:
    """
    Detect mask on lower half of face region.
    Looks for a distinct covering over mouth/nose area.
    Uses multiple heuristics to avoid false positives.
    """
    face_roi = _extract_body_region(image, "face_region")
    if face_roi.size == 0:
        return {"detected": False, "confidence": 0.0}

    h, w = face_roi.shape[:2]
    if h < 20 or w < 20:
        return {"detected": False, "confidence": 0.0}

    # Lower half of face region
    lower_face = face_roi[h // 2:, :]
    if lower_face.size == 0:
        return {"detected": False, "confidence": 0.0}

    # Convert to HSV for color analysis
    hsv = cv2.cvtColor(lower_face, cv2.COLOR_BGR2HSV)

    # Skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)

    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_ratio = np.sum(skin_mask > 0) / skin_mask.size

    # Additional check: edge/texture analysis
    # A mask creates distinct horizontal edge patterns across the lower face
    gray_lower = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_lower, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # Check for horizontal edge patterns typical of mask boundaries
    sobel_x = cv2.Sobel(gray_lower, cv2.CV_64F, 1, 0, ksize=3)
    horizontal_edges = np.abs(sobel_x) > 50
    horizontal_edge_ratio = np.sum(horizontal_edges) / horizontal_edges.size

    # Check color uniformity - masks have more uniform color than skin
    color_std = np.std(lower_face.reshape(-1, 3), axis=0)
    color_uniformity = np.mean(color_std) < 40  # lower std = more uniform

    # Stricter detection: require low skin ratio AND edge evidence of mask boundary
    if skin_ratio < 0.20 and edge_density > 0.05 and horizontal_edge_ratio > 0.03 and color_uniformity:
        confidence = round(min((1 - skin_ratio) * 60 + edge_density * 200 + horizontal_edge_ratio * 200, 90), 1)
        return {"detected": True, "confidence": confidence}
    return {"detected": False, "confidence": 0.0}


def _detect_gloves(image: np.ndarray) -> dict:
    """
    Detect gloves by analyzing hand regions.
    Looks for non-skin colored coverings at the bottom/sides.
    """
    h, w = image.shape[:2]

    # Focus on bottom edges where hands typically are
    bottom_region = image[int(h * 0.7):, :]
    if bottom_region.size == 0:
        return {"detected": False, "confidence": 0.0}

    hsv = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2HSV)

    # Skin color range
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)

    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    non_skin = cv2.bitwise_not(skin_mask)

    # Count non-skin pixels in bottom region
    total_bottom = bottom_region.shape[0] * bottom_region.shape[1]
    non_skin_ratio = np.sum(non_skin > 0) / total_bottom

    # High non-skin ratio could indicate gloves (or other objects)
    # Additional check: look for distinct glove colors (white, blue, etc.)
    if non_skin_ratio > 0.75:
        # Check for common glove colors
        white_lower = np.array([0, 0, 200], dtype=np.uint8)
        white_upper = np.array([180, 30, 255], dtype=np.uint8)
        blue_lower = np.array([100, 50, 50], dtype=np.uint8)
        blue_upper = np.array([130, 200, 200], dtype=np.uint8)

        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

        glove_color_ratio = (np.sum(white_mask > 0) + np.sum(blue_mask > 0)) / total_bottom

        if glove_color_ratio > 0.15:
            return {"detected": True, "confidence": round(min(glove_color_ratio * 200, 95), 1)}

    return {"detected": False, "confidence": 0.0}


def _detect_id_card(image: np.ndarray) -> dict:
    """
    Detect ID card/lanyard on chest area.
    Looks for rectangular object on upper torso.
    """
    torso_roi = _extract_body_region(image, "torso")
    if torso_roi.size == 0:
        return {"detected": False, "confidence": 0.0}

    gray = cv2.cvtColor(torso_roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:  # too small
            continue

        # Approximate contour
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # ID cards are rectangular (4 corners)
        if len(approx) == 4:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect_ratio = cw / ch if ch > 0 else 0
            # ID card aspect ratio ~ 0.6-0.8 (portrait) or 1.3-1.7 (landscape)
            if 0.5 < aspect_ratio < 1.8:
                return {"detected": True, "confidence": 80.0}

    return {"detected": False, "confidence": 0.0}


def _detect_jacket(roi: np.ndarray) -> dict:
    """
    Detect jacket by looking for thick outer layer on upper body.
    """
    if roi.size == 0:
        return {"detected": False, "confidence": 0.0}

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    edge_density = np.sum(edges > 0) / edges.size

    # Jackets typically have many edges (zippers, collars, seams)
    if edge_density > 0.12:
        return {"detected": True, "confidence": round(min(edge_density * 400, 85), 1)}
    return {"detected": False, "confidence": 0.0}


def _detect_hairnet(image: np.ndarray) -> dict:
    """
    Detect hairnet on top of head.
    Hairnets are typically light-colored mesh covering hair.
    """
    # Top 20% of image
    top_region = image[:int(image.shape[0] * 0.2), :]
    if top_region.size == 0:
        return {"detected": False, "confidence": 0.0}

    gray = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY)

    # Hairnets create a fine grid pattern — high frequency edges
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # Very high edge density suggests mesh pattern
    if edge_density > 0.15:
        return {"detected": True, "confidence": round(min(edge_density * 300, 80), 1)}
    return {"detected": False, "confidence": 0.0}


# Registry mapping accessory names to detection functions
_ACCESSORY_DETECTORS = {
    "apron": _detect_apron,
    "gloves": _detect_gloves,
    "mask": _detect_mask,
    "hairnet": _detect_hairnet,
    "id_card": _detect_id_card,
    "jacket": _detect_jacket,
}


def detect_accessories(image: np.ndarray) -> dict:
    """
    Detect all configured accessories.

    Reads accessory config from rules.yaml, only runs detectors for
    enabled accessories. Returns presence status for each.

    Args:
        image: BGR numpy array

    Returns:
        dict: { accessory_name: { detected, confidence, required }, ... }
    """
    accessory_rules = get_accessory_rules()
    results = {}

    upper_roi = _extract_body_region(image, "upper_body")
    torso_roi = _extract_body_region(image, "torso")

    for accessory_name, config in accessory_rules.items():
        enabled = config.get("enabled", False)
        required = config.get("required", False)

        if not enabled:
            continue

        detector = _ACCESSORY_DETECTORS.get(accessory_name)

        if detector is None:
            results[accessory_name] = {
                "detected": "Unknown",
                "confidence": 0.0,
                "required": required,
            }
            continue

        # Choose appropriate ROI
        if accessory_name in ("apron", "jacket", "id_card"):
            roi = torso_roi
        elif accessory_name in ("hairnet",):
            roi = image
        else:
            roi = upper_roi

        detection_result = detector(roi)
        results[accessory_name] = {
            "detected": detection_result["detected"],
            "confidence": detection_result["confidence"],
            "required": required,
        }

    return results
