"""
Model Registry
===============
Centralized registry to ensure models are loaded once and shared across modules.
Reduces RAM usage on Render Free Tier by avoiding duplicate weight allocations.
"""

import logging
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_MP_DIR = _MODELS_DIR / "mediapipe"

class ModelRegistry:
    _instances = {}

    @classmethod
    def get_face_detector(cls):
        if "face_detector" not in cls._instances:
            model_path = str(_MP_DIR / "blaze_face_short_range.tflite")
            options = vision.FaceDetectorOptions(
                base_options=python.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                min_detection_confidence=0.5,
            )
            cls._instances["face_detector"] = vision.FaceDetector.create_from_options(options)
            logger.info("Loaded MediaPipe FaceDetector")
        return cls._instances["face_detector"]

    @classmethod
    def get_face_landmarker(cls):
        if "face_landmarker" not in cls._instances:
            model_path = str(_MP_DIR / "face_landmarker.task")
            options = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                min_face_detection_confidence=0.5,
                num_faces=1,
            )
            cls._instances["face_landmarker"] = vision.FaceLandmarker.create_from_options(options)
            logger.info("Loaded MediaPipe FaceLandmarker")
        return cls._instances["face_landmarker"]

    @classmethod
    def get_pose_landmarker(cls):
        if "pose_landmarker" not in cls._instances:
            model_path = str(_MP_DIR / "pose_landmarker_lite.task")
            options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=0.5,
            )
            cls._instances["pose_landmarker"] = vision.PoseLandmarker.create_from_options(options)
            logger.info("Loaded MediaPipe PoseLandmarker")
        return cls._instances["pose_landmarker"]

    @classmethod
    def get_tshirt_model(cls):
        if "tshirt_model" not in cls._instances:
            model_path = str(_MODELS_DIR / "tshirt_detection_model.pt")
            cls._instances["tshirt_model"] = YOLO(model_path)
            logger.info("Loaded YOLO Tshirt Model")
        return cls._instances["tshirt_model"]

    @classmethod
    def get_cap_model(cls):
        if "cap_model" not in cls._instances:
            model_path = str(_MODELS_DIR / "cap_detection_model.pt")
            cls._instances["cap_model"] = YOLO(model_path)
            logger.info("Loaded YOLO Cap Model")
        return cls._instances["cap_model"]
