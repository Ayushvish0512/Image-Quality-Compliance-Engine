"""
BBIPL Vision Compliance Engine (VCE)
Main FastAPI Application Entry Point

Endpoints:
  GET  /health              - Health check
  POST /validate            - Full compliance check (accepts image URL)
  POST /check/person        - Person count only
  POST /check/face          - Face detection + visibility
  POST /check/blur          - Blur detection
  POST /check/lighting      - Lighting analysis
  POST /check/distance      - Face distance
  POST /check/orientation   - Face orientation
  POST /check/uniform       - Uniform detection
  POST /check/cap           - Cap detection
  POST /check/accessories   - Accessory detection
  POST /check/screenshot    - Screenshot risk analysis
"""

import io
import logging
import time
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from PIL import Image

from config.loader import load_config
from workers.pipeline import run_pipeline

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-5s | %(message)s",
)
logger = logging.getLogger("vce")

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="BBIPL Vision Compliance Engine",
    description="Validates employee attendance selfies for quality and compliance",
    version="1.0.0",
)

# ── Constants ──────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024      # 10 MB
REQUEST_TIMEOUT = 15                  # seconds


# ── Request Schema ─────────────────────────────────────────────────────────
class ImageUrlRequest(BaseModel):
    image_url: HttpUrl


# ── Helpers ────────────────────────────────────────────────────────────────
def _get_extension(url: str) -> str:
    path = urlparse(url).path
    return path.rsplit(".", 1)[-1].lower() if "." in path else ""


def _download_image(url: str) -> tuple:
    """
    Download image from URL.

    Returns:
        (numpy_array_bgr, raw_bytes)
    """
    headers = {
        "User-Agent": "BBIPL-VisionComplianceEngine/1.0 (attendance-validation; contact: admin@example.com)"
    }
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True, headers=headers)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {exc}")

    chunks = []
    total = 0
    for chunk in resp.iter_content(chunk_size=8192):
        total += len(chunk)
        if total > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Image exceeds 10 MB limit ({total / 1024 / 1024:.1f} MB)",
            )
        chunks.append(chunk)

    raw_bytes = b"".join(chunks)

    # Validate + decode via Pillow
    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        pil_img.verify()
        pil_img = Image.open(io.BytesIO(raw_bytes))
        pil_img = pil_img.convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    # Convert to OpenCV BGR
    np_img = np.array(pil_img)
    bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    return bgr, raw_bytes


# ── Endpoints ──────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    """Health check — confirms the service is running."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "BBIPL Vision Compliance Engine",
    }


@app.post("/validate")
async def validate_image(request: ImageUrlRequest):
    """
    **Full Compliance Check**

    Accept an image URL, run all detection modules, and return a
    detailed compliance report.

    **Request body:** `{ "image_url": "https://..." }`

    **Response:** Compliance report matching PRD Section 11 specification.
    """
    url = str(request.image_url)
    ext = _get_extension(url)

    if ext and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    logger.info("Processing image: %s", url)
    start = time.perf_counter()

    image_bgr, image_bytes = _download_image(url)

    # Run the full detection pipeline
    report = run_pipeline(image_bgr, image_bytes=image_bytes)

    elapsed = round(time.perf_counter() - start, 2)
    logger.info("Completed in %.2f sec — score=%d status=%s", elapsed, report["overall_score"], report["status"])

    report["processing_time_sec"] = elapsed
    report["image_url"] = url

    return report
