"""
Pydantic response schemas for API endpoints.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class ModelMetadata(BaseModel):
    """Model metadata information."""
    model_id: str = Field(..., description="Hugging Face model ID")
    model_name: str = Field(..., description="Display name of the model")
    description: str = Field(default="", description="Model description")
    author: str = Field(default="", description="Model author/organization")
    downloads: int = Field(default=0, description="Number of downloads")
    likes: int = Field(default=0, description="Number of likes")
    tags: List[str] = Field(default_factory=list, description="Model tags")
    task: str = Field(default="", description="Primary task")
    library_name: str = Field(default="", description="Library name (e.g., transformers)")
    pipeline_tag: Optional[str] = Field(None, description="Pipeline tag")
    base_model: Optional[str] = Field(None, description="Base model if this is a fine-tuned model")
    model_type: Optional[str] = Field(None, description="Model architecture type")
    size_on_disk_gb: float = Field(default=0.0, description="Model size in GB")
    parameters: Optional[str] = Field(None, description="Number of parameters")
    quantization: Optional[str] = Field(None, description="Quantization type if applicable")
    license: Optional[str] = Field(None, description="Model license")
    huggingface_url: str = Field(..., description="URL to Hugging Face model page")


class ModelRecommendation(BaseModel):
    """Single model recommendation with score and explanation."""
    model: ModelMetadata = Field(..., description="Model metadata")
    score: float = Field(
        ...,
        description="Recommendation score (higher is better)",
        ge=0.0,
        le=1.0
    )
    explanation: str = Field(
        ...,
        description="Explanation of why this model was recommended",
        example="This model is recommended because it excels at creative writing tasks and matches your size constraints."
    )
    rank: int = Field(..., description="Ranking position (1-indexed)", ge=1)


class RecommendationResponse(BaseModel):
    """Response schema for the /recommend endpoint."""
    recommendations: List[ModelRecommendation] = Field(
        ...,
        description="List of ranked model recommendations"
    )
    query_text: str = Field(..., description="The query text that was processed")
    total_results: int = Field(..., description="Total number of recommendations returned")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class MetadataResponse(BaseModel):
    """Response schema for the /metadata endpoint."""
    available_models_count: int = Field(..., description="Total number of available models")
    supported_tasks: List[str] = Field(..., description="List of supported LLM tasks")
    available_features: List[str] = Field(..., description="List of available features for filtering")
    model_catalog_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional catalog information"
    )
    api_version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

