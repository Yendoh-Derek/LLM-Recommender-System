"""
feedback.py
POST /feedback endpoint.
"""

import json
from pathlib import Path
from fastapi import APIRouter
from ..schemas.request_models import FeedbackRequest
from ..schemas.response_models import FeedbackResponse

router = APIRouter()

FEEDBACK_PATH = Path("data") / "cache" / "feedback.json"


@router.post("/feedback", response_model=FeedbackResponse, tags=["feedback"])
def submit_feedback(request: FeedbackRequest):
    """Store feedback for a recommendation."""
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    feedback_entry = request.dict()

    try:
        if FEEDBACK_PATH.exists():
            existing = json.loads(FEEDBACK_PATH.read_text(encoding="utf-8"))
        else:
            existing = []

        existing.append(feedback_entry)
        FEEDBACK_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")

        return FeedbackResponse(success=True, message="Feedback saved.")
    except Exception as exc:
        return FeedbackResponse(success=False, message=f"Could not save feedback: {exc}")
