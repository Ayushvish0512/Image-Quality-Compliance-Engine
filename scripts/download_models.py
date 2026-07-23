"""
Download required MediaPipe model files for the Vision Compliance Engine.

This script downloads the following model files:
  - face_detector_short_range.tflite  (for FaceDetector)
  - face_landmarker.task              (for FaceLandmarker)
  - pose_landmarker_lite.task         (for PoseLandmarker)

Models are stored in models/mediapipe/ relative to the project root.
"""

import os
import sys
import urllib.request

# Ensure script can be run from project root or scripts/ directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_MODEL_DIR = os.path.join(_PROJECT_ROOT, "models", "mediapipe")

# Model name -> download URL mapping
MODELS = {
    "blaze_face_short_range.tflite": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_detector/blaze_face_short_range/float16/1/"
        "blaze_face_short_range.tflite"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/"
        "face_landmarker.task"
    ),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/latest/"
        "pose_landmarker_lite.task"
    ),
}


def download_file(filename: str, url: str, dest_dir: str) -> bool:
    """Download a file from url to dest_dir if it doesn't already exist."""
    dest_path = os.path.join(dest_dir, filename)

    if os.path.exists(dest_path):
        size_kb = os.path.getsize(dest_path) / 1024
        print(f"  [SKIP] {filename} ({size_kb:.0f} KB) — already exists")
        return True

    print(f"  [DOWNLOAD] {filename}...")
    print(f"    from: {url}")

    try:
        urllib.request.urlretrieve(url, dest_path)
        size_kb = os.path.getsize(dest_path) / 1024
        print(f"    -> saved: {dest_path} ({size_kb:.0f} KB)")
        return True
    except Exception as e:
        print(f"    [ERROR] Failed to download {filename}: {e}", file=sys.stderr)
        # Clean up partial download if it exists
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def main():
    print("=" * 60)
    print("MediaPipe Model Downloader")
    print("=" * 60)

    # Create model directory if needed
    os.makedirs(_MODEL_DIR, exist_ok=True)
    print(f"\nModel directory: {_MODEL_DIR}\n")

    success_count = 0
    fail_count = 0

    for filename, url in MODELS.items():
        if download_file(filename, url, _MODEL_DIR):
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 60)
    print(f"Summary: {success_count} downloaded, {fail_count} failed")

    # Also create a marker file so the app can verify models exist
    marker_path = os.path.join(_MODEL_DIR, ".models_ready")
    if success_count == len(MODELS):
        with open(marker_path, "w") as f:
            f.write("All models downloaded successfully\n")
        print("Models ready for use!")
    else:
        if os.path.exists(marker_path):
            os.remove(marker_path)
        print("WARNING: Some models failed to download. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

