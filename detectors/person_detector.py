"""
Person Detection Module
Counts persons in the image using MediaPipe PoseLandmarker + FaceDetector (CPU).
Returns: count, classification (No Person / One Person / Multiple Persons)
"""

from pathlib import Path

import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
mp_image = mp.Image
ImageFormat = mp.ImageFormat

# -- Model paths --
_DETECTOR_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _DETECTOR_DIR.parent / "models" / "mediapipe"
_FACE_DETECTOR_MODEL = str(_MODELS_DIR / "blaze_face_short_range.tflite")
_POSE_LANDMARKER_MODEL = str(_MODELS_DIR / "pose_landmarker_lite.task")

from detectors.registry import ModelRegistry

def count_persons(image: np.ndarray) -> dict:
    """
    Count the number of persons visible in the image.

    Uses face detection as primary count (faster) and pose as backup.

    Args:
        image: RGB numpy array (H, W, 3)

    Returns:
        dict: { count, classification, faces_detected, poses_detected }
    """
    mp_img = mp_image(image_format=ImageFormat.SRGB, data=image)

    # Method 1: Count via Face Detection (fast)
    detector = ModelRegistry.get_face_detector()
    face_result = detector.detect(mp_img)
    face_count = len(face_result.detections) if face_result.detections else 0

    # Method 2: Pose detection as supplementary
    pose_model = ModelRegistry.get_pose_landmarker()
    pose_result = pose_model.detect(mp_img)
    pose_count = 0
    if pose_result.pose_landmarks:
        pose_count = len(pose_result.pose_landmarks)

    # Combine: use the higher count (conservative — better to flag extra people)
    count = max(face_count, pose_count)

    if count == 0:
        classification = "No Person"
    elif count == 1:
        classification = "One Person"
    else:
        classification = "Multiple Persons"

    return {
        "count": count,
        "classification": classification,
        "faces_detected": face_count,
        "poses_detected": pose_count,
    }
