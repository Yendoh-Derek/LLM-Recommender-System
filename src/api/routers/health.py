"""
health.py
GET /health endpoint.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["health"])
def health():
    """Health check endpoint."""
    return {"status": "ok"}
