"""
Image Upload Module
Handles file upload validation: format, size, and basic integrity checks.
"""

import io
import imghdr
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image

router = APIRouter()

ALLOWED_FORMATS = {"jpg", "jpeg", "png", "webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an attendance selfie image.

    Validates:
    - File format (JPG, JPEG, PNG, WEBP)
    - File size (max 10 MB)
    - Image can be opened/decoded
    """
    # Validate file extension
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_FORMATS))}",
        )

    # Read file content
    content = await file.read()

    # Validate file size
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(content) / 1024:.1f} KB). Maximum allowed: {MAX_FILE_SIZE // (1024 * 1024)} MB",
        )

    # Validate image integrity
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()  # Verify it's a valid image
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image or is corrupted",
        )

    # Also check via imghdr as extra safety
    detected_format = imghdr.what(None, h=content)
    if detected_format and detected_format not in {"jpeg", "png", "gif", "webp"}:
        raise HTTPException(
            status_code=400,
            detail=f"Detected format '{detected_format}' is not allowed",
        )

    return {
        "filename": file.filename,
        "size_bytes": len(content),
        "format": ext,
        "message": "Image uploaded successfully",
    }

