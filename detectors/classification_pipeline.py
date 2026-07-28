"""
Classification Pipeline — Queue System for Cap + Tshirt Detection
=================================================================
Sequential (queue-based) pipeline that:
  1. Runs cap_detection_model.pt → detects cap presence
  2. If cap found → crop bounding box → HSV color analysis
  3. Runs tshirt_detection_model.pt → detects tshirt presence + color class
  4. If tshirt found → crop bounding box → HSV color analysis
  5. Returns separate values for both cap and tshirt

Uses shared color analysis from utils.color_analysis to avoid code duplication.
Uses ModelRegistry for shared model instances to save memory.
"""

import logging
from pathlib import Path
import numpy as np
from config.loader import get_uniform_rules, get_cap_rules
from utils.color_analysis import detect_dominant_color
from detectors.registry import ModelRegistry

logger = logging.getLogger(__name__)

# ── Confidence Thresholds ───────────────────────────────────────────────────
TSHIRT_CONFIDENCE_THRESHOLD = 0.5
CAP_CONFIDENCE_THRESHOLD = 0.5

# ── Cap Detection (Queue Step 1) ───────────────────────────────────────────

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
    model = ModelRegistry.get_cap_model()

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

    # Expand cap bbox slightly (1.4x) for better HSV sampling
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

    hsv_color, hsv_confidence = detect_dominant_color(cropped_cap)
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


# ── Tshirt Detection (Queue Step 2) ────────────────────────────────────────

def detect_tshirt(image: np.ndarray) -> dict:
    """
    Detect tshirt in image using tshirt_detection_model.pt.
    Model classes: {0: 'black', 1: 'blue', 2: 'grey', 3: 'white'}
    """
    rules = get_uniform_rules()
    allowed_colors = rules.get("tshirt_colors", ["Red", "Black", "Blue", "Grey", "White"])
    model = ModelRegistry.get_tshirt_model()
    
    results = model(image, verbose=False)
    
    tshirt_detected = False
    max_confidence = 0.0
    tshirt_bbox = None
    yolo_color = "Unknown"
    
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0].item())
            if confidence > TSHIRT_CONFIDENCE_THRESHOLD:
                if confidence > max_confidence:
                    max_confidence = confidence
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    tshirt_bbox = {
                        "x1": max(0, int(x1)),
                        "y1": max(0, int(y1)),
                        "x2": min(image.shape[1], int(x2)),
                        "y2": min(image.shape[0], int(y2)),
                    }
                    class_id = int(box.cls[0].item())
                    yolo_color = model.names.get(class_id, "Unknown").capitalize()
                    tshirt_detected = True

    if not tshirt_detected or tshirt_bbox is None:
        return {
            "detected": False,
            "final_color": "Unknown",
            "final_confidence": 0.0,
            "yolo_color": "Unknown",
            "hsv_color": "Unknown",
            "allowed_colors": allowed_colors,
            "color_match": False
        }

    # Expand tshirt bbox 1.8x
    h_img, w_img = image.shape[:2]
    bbox_w = tshirt_bbox["x2"] - tshirt_bbox["x1"]
    bbox_h = tshirt_bbox["y2"] - tshirt_bbox["y1"]
    center_x = (tshirt_bbox["x1"] + tshirt_bbox["x2"]) // 2
    center_y = (tshirt_bbox["y1"] + tshirt_bbox["y2"]) // 2
    tshirt_expand = 1.8
    expanded_bbox = {
        "x1": max(0, center_x - int(bbox_w * tshirt_expand) // 2),
        "y1": max(0, center_y - int(bbox_h * tshirt_expand) // 2),
        "x2": min(w_img, center_x + int(bbox_w * tshirt_expand) // 2),
        "y2": min(h_img, center_y + int(bbox_h * tshirt_expand) // 2),
    }
    cropped_tshirt = image[expanded_bbox["y1"]:expanded_bbox["y2"],
                           expanded_bbox["x1"]:expanded_bbox["x2"]]

    hsv_color, hsv_confidence = detect_dominant_color(cropped_tshirt)
    
    # Priority Logic: If HSV says Red, it's Red (our critical rule)
    if hsv_color == "Red":
        final_color = "Red"
        final_confidence = hsv_confidence
    else:
        final_color = hsv_color if hsv_confidence > 40 else yolo_color
        final_confidence = max(hsv_confidence, round(max_confidence * 100, 1))

    color_match = final_color.lower() in [c.lower() for c in allowed_colors]

    return {
        "detected": True,
        "final_color": final_color,
        "final_confidence": final_confidence,
        "yolo_color": yolo_color,
        "hsv_color": hsv_color,
        "allowed_colors": allowed_colors,
        "color_match": color_match,
        "bbox": tshirt_bbox,
        "expanded_bbox": expanded_bbox
    }


# ── Main Queue Pipeline ────────────────────────────────────────────────────

def run_classification_pipeline(image: np.ndarray) -> dict:
    """
    Run the full classification pipeline in sequence (queue order).

    Queue Order:
      1. Cap Detection → expand crop → HSV color analysis
      2. Tshirt Detection → expand crop → HSV color analysis

    Args:
        image: BGR numpy array (H, W, 3)

    Returns:
        dict with separate "cap" and "tshirt" result objects
    """
    logger.info("=== Classification Pipeline Started (Queue) ===")

    # ── Step 1: Cap Detection ──
    logger.info("[Queue Step 1/2] Running cap detection...")
    cap_result = detect_cap_classifier(image)

    # ── Step 2: Tshirt Detection ──
    logger.info("[Queue Step 2/2] Running tshirt detection...")
    tshirt_result = detect_tshirt(image)

    logger.info("=== Classification Pipeline Complete ===")

    return {
        "cap": cap_result,
        "tshirt": tshirt_result,
    }
