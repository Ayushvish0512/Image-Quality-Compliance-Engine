"""
BBIPL Vision Compliance Engine (VCE)
Main FastAPI Application Entry Point
"""

from fastapi import FastAPI
from app.upload import router as upload_router

app = FastAPI(
    title="BBIPL Vision Compliance Engine",
    description="Validates employee attendance selfies for quality and compliance",
    version="1.0.0",
)

# Register routers
app.include_router(upload_router, tags=["Upload"])


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}

