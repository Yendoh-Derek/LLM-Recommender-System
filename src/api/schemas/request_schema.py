"""
Pydantic request schemas for API endpoints.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class RecommendationRequest(BaseModel):
    """
    Request schema for the /recommend endpoint.
    Accepts both user context and query text.
    """
    query_text: str = Field(
        ...,
        description="Text describing what the user needs or is looking for",
        min_length=1,
        max_length=1000,
        example="I need a model for generating creative writing and stories"
    )
    user_context: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional user context including preferences, constraints, and metadata",
        example={
            "use_case": "creative_writing",
            "constraints": {
                "max_model_size_gb": 10,
                "preferred_language": "en",
                "quantization": "8bit"
            },
            "preferences": {
                "task": "text-generation",
                "license": "apache-2.0"
            }
        }
    )
    top_k: Optional[int] = Field(
        None,
        description="Number of recommendations to return",
        ge=1,
        le=50,
        example=10
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query_text": "I need a model for generating creative writing and stories",
                "user_context": {
                    "use_case": "creative_writing",
                    "constraints": {
                        "max_model_size_gb": 10,
                        "preferred_language": "en"
                    }
                },
                "top_k": 10
            }
        }

