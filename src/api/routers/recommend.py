"""
recommend.py
POST /recommend endpoint.
"""

from fastapi import APIRouter
from src.recommender.inference import recommend as recommend_models
from ..schemas.request_models import RecommendRequest
from ..schemas.response_models import ModelRecommendation, RecommendResponse

router = APIRouter()


@router.post("/recommend", response_model=RecommendResponse, tags=["recommend"])
def recommend(request: RecommendRequest):
    """Return a ranked recommendation list for a given task request."""
    candidates = recommend_models(None, top_k=request.top_k)
    recommendations = [
        ModelRecommendation(
            model_id=candidate["model_id"],
            score=candidate["score"],
            metadata={"task": request.task},
        )
        for candidate in candidates
    ]
    return RecommendResponse(recommendations=recommendations)
