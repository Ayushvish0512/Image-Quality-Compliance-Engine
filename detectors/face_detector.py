"""
Face Detection & Visibility Module
Uses MediaPipe FaceDetector + FaceLandmarker (tasks.vision API).
Returns: detected flag, confidence, visibility score (0-100), and facial landmarks.
"""

import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
mp_image = mp.Image

_face_detector = None
_face_landmarker = None


def _get_face_detector():
    global _face_detector
    if _face_detector is None:
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=None),
            running_mode=VisionRunningMode.IMAGE,
            min_detection_confidence=0.5,
        )
        _face_detector = FaceDetector.create_from_options(options)
    return _face_detector


def _get_face_landmarker():
    global _face_landmarker
    if _face_landmarker is None:
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=None),
            running_mode=VisionRunningMode.IMAGE,
            min_detection_confidence=0.5,
            num_faces=1,
        )
        _face_landmarker = FaceLandmarker.create_from_options(options)
    return _face_landmarker


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
    detector = _get_face_detector()
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
    confidence = round(detection.score * 100, 1)

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
    landmarker = _get_face_landmarker()
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
