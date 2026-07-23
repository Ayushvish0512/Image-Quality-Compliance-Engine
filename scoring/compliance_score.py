"""
Compliance Score Engine
Aggregates results from all detectors, applies weighted scoring,
and determines final status: PASS / REVIEW / REJECT.

Scoring weights are configurable via environment variables.
"""

import os
import logging
from config.loader import get_thresholds

logger = logging.getLogger(__name__)

# Default scoring weights (from PRD Section 9)
# Can be overridden via .env variables
DEFAULT_WEIGHTS = {
    "person_count": 15,
    "face_detected": 15,
    "face_visibility": 10,
    "blur": 10,
    "lighting": 10,
    "face_distance": 10,
    "face_orientation": 10,
    "uniform": 15,
    "cap": 5,
    "accessories": 5,
    "screenshot_risk": 5,
}

TOTAL_WEIGHT = sum(DEFAULT_WEIGHTS.values())  # Should be 100


def _get_weight(component: str) -> int:
    """Get weight from env var or fall back to default."""
    env_key = f"SCORE_WEIGHT_{component.upper()}"
    try:
        return int(os.getenv(env_key, DEFAULT_WEIGHTS.get(component, 5)))
    except (ValueError, TypeError):
        return DEFAULT_WEIGHTS.get(component, 5)


def _compute_component_score(component: str, result: dict) -> tuple:
    """
    Compute individual component score (0-100) and failure reasons.

    Returns:
        (score_0_100, [failure_reasons])
    """
    failures = []
    score = 100  # start perfect, deduct

    if component == "person_count":
        count = result.get("count", 0)
        classification = result.get("classification", "No Person")
        if count == 0:
            score = 0
            failures.append("No person detected in frame")
        elif count > 1:
            score = 30
            failures.append(f"Multiple persons detected ({count})")
        else:
            score = 100

    elif component == "face_detected":
        detected = result.get("detected", False)
        confidence = result.get("confidence", 0)
        if not detected:
            score = 0
            failures.append("No face detected")
        else:
            score = min(confidence, 100)
            if confidence < 50:
                failures.append("Face detection confidence too low")

    elif component == "face_visibility":
        visibility = result.get("face_visibility", 0)
        score = visibility
        thresholds = get_thresholds()
        min_vis = thresholds.get("minimum_visibility", 75)
        if visibility < min_vis:
            failures.append(f"Face visibility too low ({visibility} < {min_vis})")

    elif component == "blur":
        blur_score = result.get("blur_score", 0)
        score = blur_score
        thresholds = get_thresholds()
        min_blur = thresholds.get("minimum_blur_score", 60)
        if blur_score < min_blur:
            failures.append(f"Image too blurry ({blur_score} < {min_blur})")

    elif component == "lighting":
        brightness = result.get("brightness_score", 0)
        score = brightness
        thresholds = get_thresholds()
        min_bright = thresholds.get("minimum_brightness", 50)
        if brightness < min_bright:
            failures.append(f"Image too dark ({brightness} < {min_bright})")
        elif brightness > 85:
            failures.append("Image overexposed")

    elif component == "face_distance":
        distance = result.get("distance", "Unknown")
        if distance == "Good":
            score = 100
        elif distance == "Too Close":
            score = 50
            failures.append("Standing too close to camera")
        elif distance == "Too Far":
            score = 50
            failures.append("Standing too far from camera")
        else:
            score = 0
            failures.append("Could not determine face distance")

    elif component == "face_orientation":
        orientation = result.get("orientation", "Unknown")
        if orientation == "Front":
            score = 100
        elif orientation == "Unknown":
            score = 0
            failures.append("Could not determine face orientation")
        else:
            score = 60
            failures.append(f"Not facing the camera (looking {orientation})")

    elif component == "uniform":
        detected = result.get("detected", False)
        if detected:
            score = min(result.get("confidence", 100), 100)
        else:
            score = 0
            failures.append(f"Uniform not detected or wrong color (detected: {result.get('color', 'Unknown')})")

    elif component == "cap":
        detected = result.get("detected", "Not Present")
        if detected == "Present":
            score = 100
        elif detected == "Not Present":
            score = 0
            failures.append("Cap not detected")
        else:
            score = 30
            failures.append("Cap detection inconclusive")

    elif component == "accessories":
        accessories = result
        # Sum up accessory scores
        total_accessories = 0
        detected_accessories = 0
        for name, acc in accessories.items():
            if acc.get("required", False):
                total_accessories += 1
                if acc.get("detected") is True or acc.get("detected") == "Present":
                    detected_accessories += 1
                else:
                    failures.append(f"Required accessory missing: {name}")

        if total_accessories > 0:
            score = (detected_accessories / total_accessories) * 100
        else:
            score = 100  # No required accessories = full marks

    elif component == "screenshot_risk":
        risk_level = result.get("risk_level", "Low")
        if risk_level == "Low":
            score = 100
        elif risk_level == "Medium":
            score = 50
            failures.append("Medium screenshot risk detected")
        else:
            score = 10
            failures.append("High screenshot risk — image may not be a live photo")

    return (max(0, min(score, 100)), failures)


def compute_compliance(all_detections: dict) -> dict:
    """
    Compute overall compliance score from all detection results.

    Args:
        all_detections: dict with keys matching DEFAULT_WEIGHTS
            e.g. {
                "person_count": {...},
                "face_detected": {...},
                ...
                "accessories": {...},
                "screenshot_risk": {...}
            }

    Returns:
        dict: {
            "overall_score": int,
            "status": "EXCELLENT" | "PASS" | "REVIEW" | "REJECT",
            "breakdown": { component: { score, weight, weighted_score, failures } },
            "all_failures": [str]
        }
    """
    total_weighted_score = 0
    total_weight = 0
    breakdown = {}
    all_failures = []

    for component, default_weight in DEFAULT_WEIGHTS.items():
        weight = _get_weight(component)
        result = all_detections.get(component, {})

        comp_score, failures = _compute_component_score(component, result)
        weighted = (comp_score * weight) / 100.0

        breakdown[component] = {
            "score": comp_score,
            "weight": weight,
            "weighted_contribution": round(weighted, 1),
            "failures": failures,
        }
        all_failures.extend(failures)
        total_weighted_score += weighted
        total_weight += weight

    # Normalize if weights don't sum to 100
    if total_weight > 0:
        overall_score = round((total_weighted_score / total_weight) * 100)
    else:
        overall_score = 0

    overall_score = max(0, min(overall_score, 100))

    # Determine status
    if overall_score >= 95:
        status = "EXCELLENT"
    elif overall_score >= 85:
        status = "PASS"
    elif overall_score >= 70:
        status = "REVIEW"
    else:
        status = "REJECT"

    return {
        "overall_score": overall_score,
        "status": status,
        "breakdown": breakdown,
        "all_failures": all_failures,
    }
