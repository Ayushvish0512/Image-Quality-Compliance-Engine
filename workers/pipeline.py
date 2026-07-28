"""
Detection Pipeline Orchestrator
================================
Runs all detectors in a logical order optimized for Render Free Tier (300 MB RAM).

Execution Flow (Optimized):
  Step 1: Person Detection → Check if exactly one person is present.
  Step 2: Classification Pipeline → Run Cap and Tshirt detection (YOLO + HSV).
  Step 3: Face Detection & Landmarks → Extract visibility and base for other checks.
  Step 4: Quality Checks → Blur, Lighting, Orientation, Distance.
  Step 5: Screenshot Risk Detection.
  Step 6: Accessory Detection.
  Step 7: Compliance Scoring → Aggregates all results with Red=Pass logic.
"""

import logging
import time
import numpy as np

from detectors.person_detector import count_persons
from detectors.classification_pipeline import run_classification_pipeline
from detectors.face_detector import detect_face
from detectors.blur_detector import detect_blur
from detectors.lighting_detector import detect_lighting
from detectors.orientation_detector import estimate_orientation
from detectors.distance_estimator import estimate_distance
from detectors.accessory_detector import detect_accessories
from detectors.screenshot_detector import analyze_screenshot_risk
from scoring.compliance_score import compute_compliance

logger = logging.getLogger(__name__)

def _sanitize(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def run_pipeline(image: np.ndarray, image_bytes: bytes = None) -> dict:
    """
    Run the full compliance detection pipeline.
    """
    results = {}
    start_time = time.perf_counter()

    # ---- Step 1: Person Detection ----
    logger.info("Step 1: Person detection")
    person_result = count_persons(image)
    results["person_count"] = person_result

    # ---- Step 2: Classification Pipeline (Cap & Tshirt) ----
    logger.info("Step 2: Cap and Tshirt detection")
    class_results = run_classification_pipeline(image)
    
    # Map to expected results format
    cap_res = class_results["cap"]
    tshirt_res = class_results["tshirt"]
    
    results["cap"] = {
        "detected": cap_res["detected"],
        "color": cap_res["hsv_color"],
        "confidence": cap_res["model_confidence"]
    }
    
    results["uniform"] = {
        "detected": tshirt_res["detected"],
        "color": tshirt_res["final_color"],
        "confidence": tshirt_res["final_confidence"],
        "allowed_colors": tshirt_res["allowed_colors"]
    }

    # ---- Step 3: Face Detection ----
    logger.info("Step 3: Face detection")
    face_result = detect_face(image)
    results["face_detected"] = face_result
    results["face_visibility"] = {
        "score": face_result.get("face_visibility", 0),
        "landmarks": face_result.get("landmarks")
    }

    # ---- Step 4: Quality & Orientation ----
    logger.info("Step 4: Quality and pose analysis")
    results["blur"] = detect_blur(image)
    results["lighting"] = detect_lighting(image)
    results["face_orientation"] = estimate_orientation(image)
    
    face_bbox = face_result.get("bbox")
    results["face_distance"] = estimate_distance(face_bbox, image.shape[:2])

    # ---- Step 5: Screenshot Risk ----
    logger.info("Step 5: Screenshot risk analysis")
    results["screenshot_risk"] = analyze_screenshot_risk(image, image_bytes)

    # ---- Step 6: Accessory Detection ----
    logger.info("Step 6: Accessory detection")
    results["accessories"] = detect_accessories(image)

    # ---- Step 7: Scoring & Red=Pass Logic ----
    logger.info("Step 7: Computing compliance score")
    compliance = compute_compliance(results)
    
    # Red=Pass Special Logic (Issue 6)
    is_red_tshirt = (
        tshirt_res["detected"] and 
        tshirt_res["final_color"].lower() == "red"
    )
    
    if is_red_tshirt:
        logger.info("Red tshirt detected - applying auto-PASS logic")
        compliance["overall_score"] = max(compliance["overall_score"], 90)
        compliance["status"] = "PASS"

    # Build Response
    response = {
        "status": compliance["status"],
        "overall_score": compliance["overall_score"],
        "process_time_ms": round((time.perf_counter() - start_time) * 1000, 2),
        "checks": {
            "person_count": person_result.get("count", 0),
            "person_classification": person_result.get("classification", "Unknown"),
            "face_detected": face_result.get("detected", False),
            "face_visibility": face_result.get("face_visibility", 0),
            "blur_score": results["blur"].get("blur_score", 0),
            "lighting": {
                "brightness": results["lighting"].get("brightness_score", 0),
                "classification": results["lighting"].get("classification", "Unknown")
            },
            "orientation": results["face_orientation"].get("orientation", "Unknown"),
            "distance": results["face_distance"].get("distance", "Unknown"),
            "uniform": results["uniform"],
            "cap": results["cap"],
            "accessories": results["accessories"],
            "screenshot_risk": results["screenshot_risk"].get("risk_level", "Low")
        },
        "recommendations": _generate_recommendations(compliance, results, is_red_tshirt),
        "score_breakdown": compliance["breakdown"]
    }

    return _sanitize(response)

def _generate_recommendations(compliance: dict, results: dict, is_red_tshirt: bool) -> list:
    recommendations = []
    if is_red_tshirt:
        recommendations.append("✅ Red uniform detected - Compliant")
        
    for component, info in compliance["breakdown"].items():
        if info["score"] < 70:
            for failure in info["failures"]:
                recommendations.append(failure)
                
    return recommendations[:5]
