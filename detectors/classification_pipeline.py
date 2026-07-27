"""
Classification Pipeline — Queue System for Tshirt + Cap Detection
=================================================================
Sequential (queue-based) pipeline that:
  1. Runs tshirt_detection_model.pt → detects tshirt presence + color class
  2. If tshirt found → crop bounding box → HSV color analysis for verification
  3. Runs cap_detection_model.pt → detects cap presence  
  4. If cap found → crop bounding box → HSV color analysis
  5. Returns separate values for both tshirt and cap

This file DOES NOT modify any existing code — it's a standalone addition.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from config.loader import get_uniform_rules, get_cap_rules

logger = logging.getLogger(__name__)

# ── Model Paths ─────────────────────────────────────────────────────────────
_DETECTOR_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _DETECTOR_DIR.parent / "models"

_TSHIRT_MODEL_PATH = str(_MODELS_DIR / "tshirt_detection_model.pt")
_CAP_MODEL_PATH = str(_MODELS_DIR / "cap_detection_model.pt")

# ── Confidence Thresholds ───────────────────────────────────────────────────
TSHIRT_CONFIDENCE_THRESHOLD = 0.5
CAP_CONFIDENCE_THRESHOLD = 0.5

# ── Global Model Cache (load once) ──────────────────────────────────────────
_tshirt_model = None
_cap_model = None


def _get_tshirt_model():
    global _tshirt_model
    if _tshirt_model is None:
        _tshirt_model = YOLO(_TSHIRT_MODEL_PATH)
        logger.info("Loaded tshirt_detection_model.pt — classes: %s", _tshirt_model.names)
    return _tshirt_model


def _get_cap_model():
    global _cap_model
    if _cap_model is None:
        _cap_model = YOLO(_CAP_MODEL_PATH)
        logger.info("Loaded cap_detection_model.pt — classes: %s", _cap_model.names)
    return _cap_model


# ── HSV Color Ranges (for cropped region analysis) ──────────────────────────
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


def _detect_dominant_color_hsv(cropped_img: np.ndarray) -> tuple:
    """
    Detect dominant color from a cropped region using HSV analysis.

    Args:
        cropped_img: BGR numpy array

    Returns:
        (color_name: str, confidence: float) confidence 0-100
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


# ── Tshirt Detection (Queue Step 1) ────────────────────────────────────────

def detect_tshirt(image: np.ndarray) -> dict:
    """
    Detect tshirt in image using tshirt_detection_model.pt.
    Model classes: {0: 'black', 1: 'blue', 2: 'grey', 3: 'white'}

    IMPROVEMENT: Expands the initial tight bounding box by 1.8x to
    capture more torso area for better color detection, then re-runs
    the model on the expanded crop for refined color prediction.

    Args:
        image: BGR numpy array (H, W, 3)

    Returns:
        dict with separate tshirt and cap values
    """
    rules = get_uniform_rules()
    allowed_colors = rules.get("tshirt_colors", ["Red"])
    model = _get_tshirt_model()

    results = model(image, verbose=False)

    best_detection = None
    best_confidence = 0.0
    best_class_id = -1

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0].item())
            if confidence > TSHIRT_CONFIDENCE_THRESHOLD:
                class_id = int(box.cls[0].item())
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_class_id = class_id
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    best_detection = {
                        "x1": max(0, int(x1)),
                        "y1": max(0, int(y1)),
                        "x2": min(image.shape[1], int(x2)),
                        "y2": min(image.shape[0], int(y2)),
                    }

    if best_class_id == -1 or best_detection is None:
        return {
            "detected": False,
            "model_color": "Unknown",
            "model_confidence": 0.0,
            "hsv_color": "Unknown",
            "hsv_confidence": 0.0,
            "bbox": None,
            "color_match": False,
            "allowed_colors": allowed_colors,
        }

    # ── Get initial model prediction (often tight on small collar/patch) ──
    class_names = model.names
    model_color_initial = class_names.get(best_class_id, "Unknown")
    model_confidence_initial = round(best_confidence * 100, 1)

    # ── Expand bounding box 1.8x for more torso context ──
    h_img, w_img = image.shape[:2]
    bbox = best_detection
    bbox_w = bbox["x2"] - bbox["x1"]
    bbox_h = bbox["y2"] - bbox["y1"]
    center_x = (bbox["x1"] + bbox["x2"]) // 2
    center_y = (bbox["y1"] + bbox["y2"]) // 2
    expand_factor = 1.8
    new_w = int(bbox_w * expand_factor)
    new_h = int(bbox_h * expand_factor)
    expanded_bbox = {
        "x1": max(0, center_x - new_w // 2),
        "y1": max(0, center_y - new_h // 2),
        "x2": min(w_img, center_x + new_w // 2),
        "y2": min(h_img, center_y + new_h // 2),
    }
    cropped_expanded = image[expanded_bbox["y1"]:expanded_bbox["y2"],
                             expanded_bbox["x1"]:expanded_bbox["x2"]]

    # ── Re-run model on expanded crop for refined color prediction ──
    model_color = model_color_initial
    model_confidence = model_confidence_initial
    if cropped_expanded.size > 0:
        refined_results = model(cropped_expanded, verbose=False)
        refined_confidence = 0.0
        refined_class_id = -1
        for ref_result in refined_results:
            for ref_box in ref_result.boxes:
                ref_conf = float(ref_box.conf[0].item())
                if ref_conf > TSHIRT_CONFIDENCE_THRESHOLD and ref_conf > refined_confidence:
                    refined_confidence = ref_conf
                    refined_class_id = int(ref_box.cls[0].item())
        if refined_class_id >= 0 and refined_confidence > 0:
            model_color = class_names.get(refined_class_id, model_color_initial)
            model_confidence = round(refined_confidence * 100, 1)

    # ── HSV color analysis on expanded crop (more representative area) ──
    hsv_color, hsv_confidence = _detect_dominant_color_hsv(cropped_expanded)

    # ── Color match logic (case-insensitive) ──
    allowed_colors_lower = [c.lower() for c in allowed_colors]
    model_color_in_allowed = model_color.lower() in allowed_colors_lower
    hsv_color_in_allowed = hsv_color.lower() in allowed_colors_lower

    if model_color_in_allowed:
        color_match = True
        final_color = model_color
        final_confidence = model_confidence
    elif hsv_color_in_allowed:
        color_match = True
        final_color = hsv_color
        final_confidence = hsv_confidence
    else:
        color_match = False
        final_color = model_color
        final_confidence = model_confidence

    return {
        "detected": True,
        "model_color": model_color,
        "model_confidence": model_confidence,
        "model_color_initial": model_color_initial,
        "model_confidence_initial": model_confidence_initial,
        "hsv_color": hsv_color,
        "hsv_confidence": hsv_confidence,
        "final_color": final_color,
        "final_confidence": final_confidence,
        "color_match": color_match,
        "bbox": bbox,
        "expanded_bbox": expanded_bbox,
        "allowed_colors": allowed_colors,
    }


# ── Cap Detection (Queue Step 2) ───────────────────────────────────────────

def detect_cap_classifier(image: np.ndarray) -> dict:
    """
    Detect cap in image using cap_detection_model.pt.
    Model classes: {0: 'cap', 1: 'no_cap'}

    Args:
        image: BGR numpy array (H, W, 3)

    Returns:
        dict with cap presence, confidence, color
    """
    rules = get_cap_rules()
    allowed_colors = rules.get("colors", ["Red"])
    model = _get_cap_model()

    results = model(image, verbose=False)

    cap_detected = False
    max_confidence = 0.0
    cap_bbox = None

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0].item())
            if confidence > CAP_CONFIDENCE_THRESHOLD:
                class_id = int(box.cls[0].item())
                if class_id == 0:  # class 0 = 'cap'
                    if confidence > max_confidence:
                        max_confidence = confidence
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cap_bbox = {
                            "x1": max(0, int(x1)),
                            "y1": max(0, int(y1)),
                            "x2": min(image.shape[1], int(x2)),
                            "y2": min(image.shape[0], int(y2)),
                        }
                        cap_detected = True

    if not cap_detected or cap_bbox is None:
        return {
            "detected": "Not Present",
            "model_confidence": 0.0,
            "hsv_color": "Unknown",
            "hsv_confidence": 0.0,
            "bbox": None,
            "allowed_colors": allowed_colors,
        }

    # Also expand cap bbox slightly (1.4x) for better HSV sampling
    h_img, w_img = image.shape[:2]
    bbox_w = cap_bbox["x2"] - cap_bbox["x1"]
    bbox_h = cap_bbox["y2"] - cap_bbox["y1"]
    center_x = (cap_bbox["x1"] + cap_bbox["x2"]) // 2
    center_y = (cap_bbox["y1"] + cap_bbox["y2"]) // 2
    cap_expand = 1.4
    expanded_cap_bbox = {
        "x1": max(0, center_x - int(bbox_w * cap_expand) // 2),
        "y1": max(0, center_y - int(bbox_h * cap_expand) // 2),
        "x2": min(w_img, center_x + int(bbox_w * cap_expand) // 2),
        "y2": min(h_img, center_y + int(bbox_h * cap_expand) // 2),
    }
    cropped_cap = image[expanded_cap_bbox["y1"]:expanded_cap_bbox["y2"],
                        expanded_cap_bbox["x1"]:expanded_cap_bbox["x2"]]

    hsv_color, hsv_confidence = _detect_dominant_color_hsv(cropped_cap)
    model_confidence = round(max_confidence * 100, 1)

    return {
        "detected": "Present",
        "model_confidence": model_confidence,
        "hsv_color": hsv_color,
        "hsv_confidence": hsv_confidence,
        "bbox": cap_bbox,
        "expanded_bbox": expanded_cap_bbox,
        "allowed_colors": allowed_colors,
    }


# ── Main Queue Pipeline ────────────────────────────────────────────────────

def run_classification_pipeline(image: np.ndarray) -> dict:
    """
    Run the full classification pipeline in sequence (queue order).

    Queue Order:
      1. Tshirt Detection → expand crop → re-run model → HSV color analysis
      2. Cap Detection → expand crop → HSV color analysis

    Args:
        image: BGR numpy array (H, W, 3)

    Returns:
        dict with separate "tshirt" and "cap" result objects
    """
    logger.info("=== Classification Pipeline Started (Queue) ===")

    # ── Step 1: Tshirt Detection ──
    logger.info("[Queue Step 1/2] Running tshirt detection...")
    tshirt_result = detect_tshirt(image)
    if tshirt_result["detected"]:
        logger.info(
            "  → Tshirt: model_color=%s (%.1f%%), hsv=%s (%.1f%%), match=%s",
            tshirt_result["model_color"],
            tshirt_result["model_confidence"],
            tshirt_result["hsv_color"],
            tshirt_result["hsv_confidence"],
            tshirt_result["color_match"],
        )
    else:
        logger.info("  → No tshirt detected")

    # ── Step 2: Cap Detection ──
    logger.info("[Queue Step 2/2] Running cap detection...")
    cap_result = detect_cap_classifier(image)
    if cap_result["detected"] == "Present":
        logger.info(
            "  → Cap: detected, conf=%.1f%%, hsv_color=%s (%.1f%%)",
            cap_result["model_confidence"],
            cap_result["hsv_color"],
            cap_result["hsv_confidence"],
        )
    else:
        logger.info("  → No cap detected")

    logger.info("=== Classification Pipeline Complete ===")

    return {
        "tshirt": tshirt_result,
        "cap": cap_result,
    }
