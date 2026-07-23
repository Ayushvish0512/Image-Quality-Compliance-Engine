"""
Face Distance Estimation Module
Determines if the employee is Too Close / Good / Too Far based on face bounding box size relative to frame.
Uses configurable min/max face size percentages from YAML config.
"""

import numpy as np
from config.loader import get_thresholds


def estimate_distance(face_bbox: dict, image_shape: tuple) -> dict:
    """
    Estimate face distance based on bounding box area ratio.

    Args:
        face_bbox: dict with keys xmin, ymin, width, height (normalized 0-1)
        image_shape: tuple (height, width) of original image

    Returns:
        dict: { distance, face_area_ratio, min_threshold, max_threshold }
    """
    thresholds = get_thresholds()
    min_face = thresholds["minimum_face_size"]  # e.g., 30%
    max_face = thresholds["maximum_face_size"]  # e.g., 70%

    if face_bbox is None:
        return {
            "distance": "Unknown",
            "face_area_ratio": 0.0,
            "min_threshold": min_face,
            "max_threshold": max_face,
        }

    # Face area as percentage of frame
    face_area_pct = (face_bbox["width"] * face_bbox["height"]) * 100

    if face_area_pct < min_face:
        distance = "Too Far"
    elif face_area_pct > max_face:
        distance = "Too Close"
    else:
        distance = "Good"

    return {
        "distance": distance,
        "face_area_ratio": round(face_area_pct, 1),
        "min_threshold": min_face,
        "max_threshold": max_face,
    }
