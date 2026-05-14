"""
response_models.py
Response models for FastAPI endpoints.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ModelRecommendation(BaseModel):
    model_id: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


class RecommendResponse(BaseModel):
    recommendations: List[ModelRecommendation]


class FeedbackResponse(BaseModel):
    success: bool
    message: Optional[str] = None
