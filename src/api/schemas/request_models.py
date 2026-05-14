"""
request_models.py
Request models for FastAPI endpoints.
"""

from typing import Optional

from pydantic import BaseModel


class RecommendConstraints(BaseModel):
    max_cost: Optional[float] = None
    min_performance: Optional[float] = None


class RecommendRequest(BaseModel):
    task: str
    constraints: Optional[RecommendConstraints] = None
    top_k: int = 5


class FeedbackRequest(BaseModel):
    recommendation_id: str
    model_id: str
    rating: int
    user_id: str
