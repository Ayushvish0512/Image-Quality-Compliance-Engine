"""
Detection Pipeline Orchestrator
Runs all detectors in sequence, sharing results where dependencies exist.
Manages MediaPipe lifecycle to avoid redundant model loads.

Execution Order (some parallelizable, some sequential due to dependencies):
  Step 1: Face Detection + Person Detection (can run in parallel)
  Step 2: Face Visibility (depends on face landmarks from Step 1)
  Step 3: Face Distance (depends on face bbox from Step 1)
  Step 4: Face Orientation (depends on face mesh from Step 1)
  Step 5: Blur Detection (independent, can run anytime)
  Step 6: Lighting Detection (independent, can run anytime)
  Step 7: Uniform Detection (independent but uses pose)
  Step 8: Cap Detection (uses face_bbox from Step 1)
  Step 9: Accessory Detection (independent)
  Step 10: Screenshot Risk Detection (independent, uses image bytes)
  Step 11: Compliance Score (aggregates all results)
"""

import json
import logging

import numpy as np

from detectors.face_detector import detect_face
from detectors.person_detector import count_persons
from detectors.blur_detector import detect_blur
from detectors.lighting_detector import detect_lighting
from detectors.orientation_detector import estimate_orientation
from detectors.distance_estimator import estimate_distance
from detectors.uniform_detector import detect_uniform
from detectors.cap_detector import detect_cap
from detectors.accessory_detector import detect_accessories
from detectors.screenshot_detector import analyze_screenshot_risk
from scoring.compliance_score import compute_compliance, DEFAULT_WEIGHTS


def _sanitize(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

logger = logging.getLogger(__name__)


def run_pipeline(image: np.ndarray, image_bytes: bytes = None) -> dict:
    """
    Run the full detection pipeline on an image.

    Args:
        image: BGR numpy array (H, W, 3)
        image_bytes: Original image bytes (for EXIF/screenshot analysis)

    Returns:
        dict: Full compliance report matching PRD Section 11 spec
    """
    results = {}
    warnings = []

    # ---- Step 1: Face Detection & Person Detection (can run parallel) ----
    logger.info("Running face detection...")
    face_result = detect_face(image)
    results["face_detected"] = face_result

    logger.info("Running person detection...")
    person_result = count_persons(image)
    results["person_count"] = person_result

    # ---- Step 2: Face Visibility (from face landmarks) ----
    results["face_visibility"] = {
        "score": face_result.get("face_visibility", 0),
        "landmarks": face_result.get("landmarks"),
    }

    # ---- Step 3: Face Distance (from face bbox) ----
    face_bbox = face_result.get("bbox")
    distance_result = estimate_distance(face_bbox, image.shape[:2])
    results["face_distance"] = distance_result

    # ---- Step 4: Face Orientation ----
    orientation_result = estimate_orientation(image)
    results["face_orientation"] = orientation_result

    # ---- Step 5: Blur Detection ----
    logger.info("Running blur detection...")
    blur_result = detect_blur(image)
    results["blur"] = blur_result

    # ---- Step 6: Lighting Detection ----
    logger.info("Running lighting detection...")
    lighting_result = detect_lighting(image)
    results["lighting"] = lighting_result

    # ---- Step 7: Uniform Detection ----
    logger.info("Running uniform detection...")
    uniform_result = detect_uniform(image)
    results["uniform"] = uniform_result

    # ---- Step 8: Cap Detection ----
    logger.info("Running cap detection...")
    cap_result = detect_cap(image, face_bbox)
    results["cap"] = cap_result

    # ---- Step 9: Accessory Detection ----
    logger.info("Running accessory detection...")
    accessory_result = detect_accessories(image)
    results["accessories"] = accessory_result

    # ---- Step 10: Screenshot Risk Detection ----
    logger.info("Running screenshot risk detection...")
    screenshot_result = analyze_screenshot_risk(image, image_bytes)
    results["screenshot_risk"] = screenshot_result

    # ---- Step 11: Compliance Score Calculation ----
    logger.info("Computing compliance score...")
    compliance = compute_compliance(results)

    # ---- Build Final Response (PRD Section 11 Format) ----
    response = {
        "status": compliance["status"],
        "overall_score": compliance["overall_score"],
        "checks": {
            "person_count": person_result.get("count", 0),
            "person_classification": person_result.get("classification", "Unknown"),
            "face_detected": face_result.get("detected", False),
            "face_confidence": face_result.get("confidence", 0),
            "face_visibility": face_result.get("face_visibility", 0),
            "blur_score": blur_result.get("blur_score", 0),
            "blur_classification": blur_result.get("classification", "Unknown"),
            "lighting": {
                "brightness_score": lighting_result.get("brightness_score", 0),
                "contrast_score": lighting_result.get("contrast_score", 0),
                "classification": lighting_result.get("classification", "Unknown"),
            },
            "distance": distance_result.get("distance", "Unknown"),
            "face_area_ratio": distance_result.get("face_area_ratio", 0),
            "orientation": orientation_result.get("orientation", "Unknown"),
            "yaw": orientation_result.get("yaw", 0),
            "pitch": orientation_result.get("pitch", 0),
            "roll": orientation_result.get("roll", 0),
            "uniform": {
                "detected": uniform_result.get("detected", False),
                "color": uniform_result.get("color", "Unknown"),
                "confidence": uniform_result.get("confidence", 0),
                "allowed_colors": uniform_result.get("allowed_colors", []),
            },
            "cap": {
                "detected": cap_result.get("detected", "Unknown"),
                "color": cap_result.get("color", "Unknown"),
                "confidence": cap_result.get("confidence", 0),
            },
            "accessories": {
                name: {
                    "detected": acc.get("detected", False),
                    "confidence": acc.get("confidence", 0),
                    "required": acc.get("required", False),
                }
                for name, acc in accessory_result.items()
            },
            "screenshot_risk": screenshot_result.get("risk_level", "Low"),
            "screenshot_score": screenshot_result.get("score", 0),
        },
        "recommendations": _generate_recommendations(compliance, results),
        "score_breakdown": compliance["breakdown"],
    }

    return _sanitize(response)


def _generate_recommendations(compliance: dict, raw_results: dict) -> list:
    """
    Generate actionable recommendations based on failures.
    """
    recommendations = []

    for component, info in compliance["breakdown"].items():
        for failure in info["failures"]:
            if "person" in component.lower():
                if "multiple" in failure.lower():
                    recommendations.append("Ensure only one person is in the frame")
                elif "no person" in failure.lower():
                    recommendations.append("Position yourself clearly in the frame")
            elif "face" in component.lower():
                if "no face" in failure.lower():
                    recommendations.append("Ensure your face is clearly visible to the camera")
                elif "confidence" in failure.lower():
                    recommendations.append("Move closer and face the camera directly")
                elif "visibility" in failure.lower():
                    recommendations.append("Remove any obstructions from your face")
            elif "blur" in component.lower():
                recommendations.append("Hold the camera steady and avoid movement")
            elif "lighting" in component.lower() or "bright" in failure.lower():
                if "dark" in failure.lower():
                    recommendations.append("Increase lighting in the area")
                elif "overexposed" in failure.lower():
                    recommendations.append("Move away from direct bright light sources")
            elif "distance" in component.lower():
                if "close" in failure.lower():
                    recommendations.append("Move slightly away from the camera")
                elif "far" in failure.lower():
                    recommendations.append("Move slightly closer to the camera")
            elif "orientation" in component.lower():
                recommendations.append("Face the camera directly (front-facing)")
            elif "uniform" in component.lower():
                recommendations.append(
                    f"Wear the required uniform color: "
                    f"{', '.join(raw_results.get('uniform', {}).get('allowed_colors', []))}"
                )
            elif "cap" in component.lower():
                recommendations.append("Wear the required cap")
            elif "accessories" in component.lower():
                rec = failure.replace("Required accessory missing: ", "Wear the required ")
                if rec not in recommendations:
                    recommendations.append(rec)
            elif "screenshot" in component.lower():
                recommendations.append("Take a live selfie, don't use a screenshot")

    # Limit to top 5 most important recommendations
    return recommendations[:5]
