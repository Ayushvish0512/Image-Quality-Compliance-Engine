"""
Uniform/Dress Detection Module
Hybrid approach using pre-trained person detection + color analysis.
Steps:
  1. Get upper-body region from MediaPipe PoseLandmarker
  2. Extract dominant color via K-Means clustering
  3. Match against allowed colors from config
"""

import cv2
import numpy as np
import mediapipe as mp
from sklearn.cluster import KMeans
from config.loader import get_uniform_rules

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
mp_image = mp.Image
ImageFormat = mp.ImageFormat

_pose_landmarker = None


def _get_pose_landmarker():
    global _pose_landmarker
    if _pose_landmarker is None:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=None),
            running_mode=VisionRunningMode.IMAGE,
            min_detection_confidence=0.6,
        )
        _pose_landmarker = PoseLandmarker.create_from_options(options)
    return _pose_landmarker


# Basic color name mapping (BGR -> name)
_COLOR_MAP = {
    "Black": ([0, 0, 0], [50, 50, 50]),
    "Blue": ([90, 60, 0], [140, 120, 80]),
    "Orange": ([0, 100, 200], [30, 160, 255]),
    "White": ([200, 200, 200], [255, 255, 255]),
    "Red": ([0, 0, 100], [60, 60, 200]),
    "Green": ([0, 100, 0], [80, 180, 80]),
    "Yellow": ([0, 180, 180], [60, 255, 255]),
    "Brown": ([30, 50, 80], [80, 100, 150]),
    "Grey": ([100, 100, 100], [180, 180, 180]),
    "Navy": ([60, 30, 0], [100, 60, 40]),
}


def _classify_color(bgr_color: tuple) -> str:
    """Map a BGR color to the nearest named color."""
    b, g, r = map(int, bgr_color)
    best_match = "Unknown"
    min_dist = float("inf")
    for name, (lower, upper) in _COLOR_MAP.items():
        if lower[0] <= b <= upper[0] and lower[1] <= g <= upper[1] and lower[2] <= r <= upper[2]:
            return name
        center = np.array([
            (lower[0] + upper[0]) / 2,
            (lower[1] + upper[1]) / 2,
            (lower[2] + upper[2]) / 2,
        ])
        dist = np.linalg.norm(np.array([b, g, r]) - center)
        if dist < min_dist:
            min_dist = dist
            best_match = name
    return best_match


def detect_uniform(image: np.ndarray) -> dict:
    """
    Detect uniform/tshirt color.

    Args:
        image: RGB or BGR numpy array (H, W, 3)

    Returns:
        dict: { detected, color, confidence, method }
    """
    rules = get_uniform_rules()
    allowed_colors = rules.get("tshirt_colors", ["Blue", "Orange", "Black"])
    confidence_threshold = rules.get("confidence_threshold", 70)

    # Ensure RGB for MediaPipe
    if image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image
    else:
        rgb = image
    h, w = image.shape[:2]

    # Step 1: Get upper body via pose landmarks
    mp_img = mp_image(image_format=ImageFormat.SRGB, data=rgb)
    pose_model = _get_pose_landmarker()
    result = pose_model.detect(mp_img)

    if not result.pose_landmarks:
        return {"detected": False, "color": "Unknown", "confidence": 0.0, "method": "pose_failed"}

    landmarks = result.pose_landmarks[0]

    # PoseLandmarker uses named indices for landmarks
    # Left shoulder ~ 11, Right shoulder ~ 12, Left hip ~ 23, Right hip ~ 24
    landmark_indices = {
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_hip": 23,
        "right_hip": 24,
    }

    def _get_lm(idx):
        if idx < len(landmarks):
            return landmarks[idx]
        return None

    pts = {}
    for name, idx in landmark_indices.items():
        lm = _get_lm(idx)
        if lm:
            pts[name] = (lm.x, lm.y)

    if len(pts) < 4:
        return {"detected": False, "color": "Unknown", "confidence": 0.0, "method": "pose_failed"}

    x_coords = [pts[k][0] for k in pts]
    y_coords = [pts[k][1] for k in pts]

    x_min = max(0, int(min(x_coords) * w))
    x_max = min(w, int(max(x_coords) * w))
    y_min = max(0, int(min(y_coords) * h))
    y_max = min(h, int(max(y_coords) * h))

    # Expand upward to cover chest
    y_min = max(0, y_min - int(h * 0.05))
    y_max = min(h, y_max + int(h * 0.05))

    if x_max <= x_min or y_max <= y_min:
        return {"detected": False, "color": "Unknown", "confidence": 0.0, "method": "invalid_bbox"}

    torso_roi = image[y_min:y_max, x_min:x_max]
    if torso_roi.size == 0:
        return {"detected": False, "color": "Unknown", "confidence": 0.0, "method": "empty_roi"}

    # Step 2: Extract dominant color via K-Means
    pixels = torso_roi.reshape(-1, 3)
    if len(pixels) < 10:
        return {"detected": False, "color": "Unknown", "confidence": 0.0, "method": "too_few_pixels"}

    if len(pixels) > 10000:
        idx = np.random.choice(len(pixels), 10000, replace=False)
        pixels = pixels[idx]

    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
    dominant_color = tuple(dominant_color.astype(int))

    # Step 3: Classify the color
    classified_name = _classify_color(dominant_color)

    # Step 4: Check against allowed colors
    if classified_name in allowed_colors:
        detected = True
        color = classified_name
        cluster_counts = np.bincount(kmeans.labels_)
        dominant_pct = max(cluster_counts) / len(kmeans.labels_)
        confidence = round(min(dominant_pct * 100, 100), 1)
    else:
        detected = False
        color = classified_name
        confidence = 0.0

    return {
        "detected": detected,
        "color": color,
        "confidence": confidence,
        "allowed_colors": allowed_colors,
        "method": "kmeans_color_extraction",
    }
