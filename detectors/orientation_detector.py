"""
Face Orientation Detection Module
Uses MediaPipe FaceLandmarker (tasks.vision API) landmarks to compute yaw, pitch, roll.
Returns: orientation (Front/Left/Right/Up/Down), yaw, pitch, roll angles.
"""

from pathlib import Path

import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
mp_image = mp.Image
ImageFormat = mp.ImageFormat

# -- Model paths --
_DETECTOR_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _DETECTOR_DIR.parent / "models" / "mediapipe"
_FACE_LANDMARKER_MODEL = str(_MODELS_DIR / "face_landmarker.task")

_face_landmarker = None


def _get_landmarker():
    global _face_landmarker
    if _face_landmarker is None:
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_FACE_LANDMARKER_MODEL),
            running_mode=VisionRunningMode.IMAGE,
            min_face_detection_confidence=0.5,
            num_faces=1,
        )
        _face_landmarker = FaceLandmarker.create_from_options(options)
    return _face_landmarker


def estimate_orientation(image: np.ndarray) -> dict:
    """
    Estimate face orientation from image.

    Uses 6 key face mesh landmarks to compute head pose:
    - Nose tip (1), Chin (152), Left eye outer (33), Right eye outer (263),
      Left mouth corner (61), Right mouth corner (291)

    Args:
        image: RGB numpy array (H, W, 3)

    Returns:
        dict: { orientation, yaw, pitch, roll }
    """
    mp_img = mp_image(image_format=ImageFormat.SRGB, data=image)
    h, w = image.shape[:2]

    landmarker = _get_landmarker()
    result = landmarker.detect(mp_img)

    if not result.face_landmarks:
        return {
            "orientation": "Unknown",
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
        }

    landmarks = result.face_landmarks[0]

    # Key landmark indices for pose estimation
    key_indices = [1, 152, 33, 263, 61, 291]

    pts = []
    for idx in key_indices:
        if idx < len(landmarks):
            lm = landmarks[idx]
            pts.append([lm.x, lm.y, lm.z])
        else:
            pts.append([0.0, 0.0, 0.0])

    pts = np.array(pts)

    # --- Yaw: horizontal turn ---
    nose_tip = pts[0]
    left_eye = pts[2]
    right_eye = pts[3]

    eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
    nose_offset = nose_tip[0] - eye_center_x

    eye_dist = abs(left_eye[0] - right_eye[0])
    if eye_dist > 0:
        yaw_normalized = nose_offset / (eye_dist * 0.5)
    else:
        yaw_normalized = 0.0

    yaw = np.clip(yaw_normalized * 45, -45, 45)

    # --- Pitch: looking up/down ---
    chin = pts[1]
    mouth_left = pts[4]
    mouth_right = pts[5]

    nose_chin_y = chin[1] - nose_tip[1]
    eye_mouth_y = ((left_eye[1] + right_eye[1]) / 2.0) - ((mouth_left[1] + mouth_right[1]) / 2.0)

    if eye_mouth_y != 0:
        pitch_ratio = nose_chin_y / eye_mouth_y
        pitch_deviation = pitch_ratio - 0.66
        pitch = np.clip(pitch_deviation * 180, -30, 30)
    else:
        pitch = 0.0

    # --- Roll: head tilt ---
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    roll = np.degrees(np.arctan2(dy, dx)) if dx != 0 else 0.0

    yaw = round(float(yaw), 1)
    pitch = round(float(pitch), 1)
    roll = round(float(roll), 1)

    # Determine orientation label
    abs_yaw = abs(yaw)
    abs_pitch = abs(pitch)

    if abs_yaw > 25:
        orientation = "Left" if yaw < 0 else "Right"
    elif abs_pitch > 15:
        orientation = "Down" if pitch < 0 else "Up"
    else:
        orientation = "Front"

    return {
        "orientation": orientation,
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
    }
