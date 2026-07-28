"""
Face Detection & Visibility Module
Uses MediaPipe FaceDetector + FaceLandmarker (tasks.vision API).
Returns: detected flag, confidence, visibility score (0-100), and facial landmarks.
"""

import os
from pathlib import Path

import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
mp_image = mp.Image

# -- Model paths --
_DETECTOR_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _DETECTOR_DIR.parent / "models" / "mediapipe"
_FACE_DETECTOR_MODEL = str(_MODELS_DIR / "blaze_face_short_range.tflite")
_FACE_LANDMARKER_MODEL = str(_MODELS_DIR / "face_landmarker.task")

from detectors.registry import ModelRegistry

def detect_face(image: np.ndarray) -> dict:
    """
    Detect a face in the image and compute visibility score.

    Args:
        image: RGB numpy array (H, W, 3)

    Returns:
        dict with keys: detected, confidence, face_visibility, landmarks, bbox
    """
    # Convert to MediaPipe Image
    mp_img = mp_image(image_format=mp.ImageFormat.SRGB, data=image)

    # --- Step 1: Face Detection ---
    detector = ModelRegistry.get_face_detector()
    detection_result = detector.detect(mp_img)

    if not detection_result.detections:
        return {
            "detected": False,
            "confidence": 0.0,
            "face_visibility": 0.0,
            "landmarks": None,
            "bbox": None,
        }

    detection = detection_result.detections[0]
    # MediaPipe tasks API: Detection.categories[0].score
    confidence = round(detection.categories[0].score * 100, 1)

    # Bounding box
    bbox = detection.bounding_box
    h, w = image.shape[:2]
    bbox_dict = {
        "xmin": round(bbox.origin_x / w, 4),
        "ymin": round(bbox.origin_y / h, 4),
        "width": round(bbox.width / w, 4),
        "height": round(bbox.height / h, 4),
    }

    # --- Step 2: Face Landmarks (for visibility) ---
    landmarker = ModelRegistry.get_face_landmarker()
    landmark_result = landmarker.detect(mp_img)

    if not landmark_result.face_landmarks:
        return {
            "detected": True,
            "confidence": confidence,
            "face_visibility": 50.0,
            "landmarks": None,
            "bbox": bbox_dict,
        }

    landmarks = landmark_result.face_landmarks[0]

    # Key landmark indices for visibility scoring
    EYES = {33, 133, 362, 263}
    NOSE = {1, 2, 98, 327}
    MOUTH = {61, 291, 39, 269}
    CHIN = {152, 175, 200}
    FOREHEAD = {10, 338, 297, 332}

    all_key_points = EYES | NOSE | MOUTH | CHIN | FOREHEAD

    landmark_pts = {}
    for idx in all_key_points:
        if idx < len(landmarks):
            lm = landmarks[idx]
            landmark_pts[idx] = (int(lm.x * w), int(lm.y * h))

    # --- Visibility Heuristic ---
    visible_regions = 0
    total_regions = 5

    if all(idx in landmark_pts for idx in EYES):
        visible_regions += 1
    if all(idx in landmark_pts for idx in NOSE):
        visible_regions += 1
    if all(idx in landmark_pts for idx in MOUTH):
        visible_regions += 1
    if all(idx in landmark_pts for idx in CHIN):
        visible_regions += 1
    if all(idx in landmark_pts for idx in FOREHEAD):
        visible_regions += 1

    visibility_score = round((visible_regions / total_regions) * 100, 1)

    readable_landmarks = {
        "left_eye": landmark_pts.get(33),
        "right_eye": landmark_pts.get(263),
        "nose_tip": landmark_pts.get(1),
        "mouth_left": landmark_pts.get(61),
        "mouth_right": landmark_pts.get(291),
        "chin": landmark_pts.get(152),
    }

    return {
        "detected": True,
        "confidence": confidence,
        "face_visibility": visibility_score,
        "landmarks": readable_landmarks,
        "bbox": bbox_dict,
    }
