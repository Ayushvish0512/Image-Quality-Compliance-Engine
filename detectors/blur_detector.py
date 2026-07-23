"""
Blur Detection Module
Measures image sharpness using Variance of Laplacian and Tenengrad gradient.
Returns: blur_score (0-100), classification (Sharp / Slightly Blurry / Very Blurry)
"""

import cv2
import numpy as np


def detect_blur(image: np.ndarray) -> dict:
    """
    Detect image blur using multiple methods for robustness.

    Args:
        image: RGB or BGR numpy array (H, W, 3)

    Returns:
        dict: { blur_score, classification, variance_laplacian, tenengrad }
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[2] == 3 else image

    # --- Method 1: Variance of Laplacian ---
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance_laplacian = laplacian.var()

    # --- Method 2: Tenengrad (Sobel gradient magnitude) ---
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    tenengrad = np.mean(gradient_magnitude)

    # Normalize both scores to 0-100 range
    # Variance of Laplacian: typical range 0-1000 for sharp images
    # Tenengrad: typical range 0-100 for sharp images
    norm_laplacian = min(variance_laplacian / 10.0, 100.0)
    norm_tenengrad = min(tenengrad * 2.0, 100.0)

    # Combined score (weighted average)
    blur_score = round((norm_laplacian * 0.6 + norm_tenengrad * 0.4), 1)

    # Classification
    if blur_score >= 70:
        classification = "Sharp"
    elif blur_score >= 40:
        classification = "Slightly Blurry"
    else:
        classification = "Very Blurry"

    return {
        "blur_score": blur_score,
        "classification": classification,
        "variance_laplacian": round(variance_laplacian, 2),
        "tenengrad": round(tenengrad, 2),
    }
