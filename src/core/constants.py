"""
Constants used throughout the LLM Recommender API.
"""
from typing import Dict, List, Any

# Model metadata fields that will be returned in recommendations
MODEL_METADATA_FIELDS: List[str] = [
    "model_id",
    "model_name",
    "description",
    "author",
    "downloads",
    "likes",
    "tags",
    "task",
    "library_name",
    "pipeline_tag",
    "base_model",
    "model_type",
    "size_on_disk_gb",
    "parameters",
    "quantization",
    "license",
    "huggingface_url",
]

# Default model metadata (for mock data)
DEFAULT_MODEL_METADATA: Dict[str, Any] = {
    "model_id": "",
    "model_name": "",
    "description": "",
    "author": "",
    "downloads": 0,
    "likes": 0,
    "tags": [],
    "task": "",
    "library_name": "",
    "pipeline_tag": "",
    "base_model": "",
    "model_type": "",
    "size_on_disk_gb": 0.0,
    "parameters": "",
    "quantization": "",
    "license": "",
    "huggingface_url": "",
}

# Common LLM tasks/categories
LLM_TASKS: List[str] = [
    "text-generation",
    "text2text-generation",
    "question-answering",
    "summarization",
    "translation",
    "conversational",
    "text-classification",
    "zero-shot-classification",
    "fill-mask",
    "token-classification",
]

# Error messages
ERROR_MESSAGES: Dict[str, str] = {
    "INVALID_REQUEST": "Invalid request parameters",
    "MODEL_NOT_FOUND": "Model not found",
    "ARTIFACTS_NOT_LOADED": "Model artifacts not yet loaded",
    "PREPROCESSING_ERROR": "Error during preprocessing",
    "INFERENCE_ERROR": "Error during model inference",
    "RANKING_ERROR": "Error during ranking",
    "INTERNAL_ERROR": "Internal server error",
}

# API response status codes
HTTP_STATUS = {
    "OK": 200,
    "CREATED": 201,
    "BAD_REQUEST": 400,
    "NOT_FOUND": 404,
    "INTERNAL_ERROR": 500,
    "SERVICE_UNAVAILABLE": 503,
}

