"""
Screenshot Risk Detection Module
Heuristic-based detection for screenshots, edited images, and cropped images.
Checks: EXIF metadata, aspect ratio, pixel duplication, compression artifacts, screen borders.

Returns: risk_level (Low / Medium / High), indicators_found, confidence
"""

import cv2
import numpy as np
from PIL import Image
from io import BytesIO


def analyze_screenshot_risk(image: np.ndarray, image_bytes: bytes = None) -> dict:
    """
    Analyze image for screenshot/edited image indicators.

    Args:
        image: BGR numpy array (H, W, 3)
        image_bytes: Original bytes for EXIF analysis (optional)

    Returns:
        dict: {
            risk_level: "Low" | "Medium" | "High",
            score: 0-100,
            indicators: [string descriptions],
            details: { exif_present, is_screen_ratio, pixel_duplication, ... }
        }
    """
    h, w = image.shape[:2]
    indicators = []
    risk_score = 0

    # --- 1. EXIF Metadata Analysis ---
    exif_present = False
    has_camera_make = False
    has_camera_model = False
    exif_image_unique = False
    datetime_original = None

    if image_bytes:
        try:
            pil_img = Image.open(BytesIO(image_bytes))
            exif_data = pil_img.getexif()

            if exif_data:
                exif_present = True
                # Check for camera-specific tags
                if 271 in exif_data:  # Make
                    has_camera_make = True
                if 272 in exif_data:  # Model
                    has_camera_model = True
                if 36867 in exif_data:  # DateTimeOriginal
                    datetime_original = exif_data[36867]
                if 42016 in exif_data:  # ImageUniqueID
                    exif_image_unique = True

            if not exif_present:
                indicators.append("No EXIF metadata found")
                risk_score += 15  # Reduced from 25
            elif not has_camera_make or not has_camera_model:
                indicators.append("Missing camera metadata (make/model)")
                risk_score += 10  # Reduced from 15
        except Exception:
            indicators.append("Could not read EXIF data")
            risk_score += 5  # Reduced from 10

    else:
        indicators.append("No EXIF data available for analysis")
        risk_score += 5

    # --- 2. Aspect Ratio Analysis ---
    aspect_ratio = w / h if h > 0 else 0
    common_screen_ratios = [
        16 / 9, 9 / 16,       # 16:9
        4 / 3, 3 / 4,         # 4:3
        18.5 / 9, 9 / 18.5,   # 18.5:9 (modern phones)
        19.5 / 9, 9 / 19.5,   # 19.5:9
        20 / 9, 9 / 20,       # 20:9
        21 / 9, 9 / 21,       # 21:9
        16 / 10, 10 / 16,     # 16:10
        5 / 4, 4 / 5,         # 5:4
    ]

    is_screen_ratio = any(abs(aspect_ratio - r) < 0.05 for r in common_screen_ratios)
    if is_screen_ratio:
        indicators.append(f"Image has screen-like aspect ratio ({aspect_ratio:.2f})")
        risk_score += 10  # Reduced from 15

    # --- 3. UI Border Detection ---
    # Screenshots often have solid-color borders at edges
    border_detected = False
    border_colors = set()

    # Check top and bottom edge strips (5px)
    for strip_name, strip in [
        ("top", image[:5, :, :]),
        ("bottom", image[-5:, :, :]),
        ("left", image[:, :5, :]),
        ("right", image[:, -5:, :]),
    ]:
        if strip.size == 0:
            continue
        mean_color = np.mean(strip.reshape(-1, 3), axis=0)
        std_color = np.std(strip.reshape(-1, 3), axis=0)

        # Low variance = solid color
        if np.mean(std_color) < 15:
            border_detected = True
            border_colors.add(tuple(mean_color.astype(int)))

    if border_detected:
        indicators.append(f"Screen UI border(s) detected ({len(border_colors)} distinct)")
        risk_score += 10  # Reduced from 15

    # --- 4. Pixel Duplication (checkerboard sampling) ---
    # Screenshots have less pixel variation than camera photos
    sampled = image[::20, ::20, :]
    unique_pixels = len(np.unique(sampled.reshape(-1, 3), axis=0))
    total_sampled = sampled.shape[0] * sampled.shape[1]
    pixel_variety = unique_pixels / total_sampled if total_sampled > 0 else 0

    if pixel_variety < 0.10:  # Reduced from 0.15 (more lenient)
        indicators.append(f"Low pixel variety ({pixel_variety:.2%}) suggesting screen capture")
        risk_score += 10  # Reduced from 15

    # --- 5. Compression Artifact Detection ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # JPEG artifacts often show as 8x8 block boundaries
    # Check DCT-like blocking
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    edge_std = np.std(laplacian)

    # Very low edge std = overly smooth (heavy compression)
    if edge_std < 10:  # Reduced from 15 (more lenient)
        indicators.append("Heavy compression artifacts detected")
        risk_score += 10

    # --- 6. Overall Risk Calculation ---
    risk_score = min(risk_score, 100)

    # Increased thresholds to reduce false positives (live images less likely flagged High)
    if risk_score >= 65:  # Increased from 50
        risk_level = "High"
    elif risk_score >= 35:  # Increased from 25
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "risk_level": risk_level,
        "score": round(risk_score, 1),
        "indicators": indicators,
        "details": {
            "exif_present": exif_present,
            "has_camera_metadata": has_camera_make or has_camera_model,
            "is_screen_ratio": is_screen_ratio,
            "aspect_ratio": round(aspect_ratio, 3),
            "ui_borders_detected": border_detected,
            "pixel_variety": round(pixel_variety, 3),
            "compression_score": round(edge_std, 1),
        },
    }
