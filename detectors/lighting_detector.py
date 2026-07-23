"""
Lighting Detection Module
Evaluates image brightness (luminance) and contrast.
Returns: brightness_score, contrast_score, classification (Underexposed / Balanced / Overexposed)
"""

import cv2
import numpy as np


def detect_lighting(image: np.ndarray) -> dict:
    """
    Analyze lighting conditions in the image.

    Args:
        image: RGB or BGR numpy array (H, W, 3)

    Returns:
        dict: { brightness_score, contrast_score, classification }
    """
    # Convert to LAB color space for better luminance analysis
    if image.shape[2] == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    else:
        lab = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        lab = cv2.cvtColor(lab, cv2.COLOR_BGR2LAB)

    # L channel (luminance) - range 0-255
    l_channel = lab[:, :, 0]
    mean_luminance = np.mean(l_channel)

    # Brightness score (0-100): map mean luminance 0-255 -> 0-100
    brightness_score = round((mean_luminance / 255.0) * 100, 1)

    # Contrast: standard deviation of L channel
    std_luminance = np.std(l_channel)
    contrast_score = round(min(std_luminance * 2.0, 100.0), 1)

    # Classification based on brightness
    if brightness_score < 35:
        classification = "Underexposed"
    elif brightness_score > 80:
        classification = "Overexposed"
    else:
        classification = "Balanced"

    return {
        "brightness_score": brightness_score,
        "contrast_score": contrast_score,
        "classification": classification,
        "mean_luminance": round(mean_luminance, 1),
    }
