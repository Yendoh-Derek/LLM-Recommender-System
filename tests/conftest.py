"""
Pytest configuration and fixtures for testing.
"""
import pytest
from fastapi.testclient import TestClient
from typing import Dict, Any, Generator, List
import numpy as np

from src.api.main import app
from src.core.config import settings


@pytest.fixture
def client() -> TestClient:
    """
    Create a test client for the FastAPI application.
    
    Returns:
        TestClient instance
    """
    return TestClient(app)


@pytest.fixture
def sample_recommendation_request() -> Dict[str, Any]:
    """
    Sample recommendation request payload.
    
    Returns:
        Dictionary with sample request data
    """
    return {
        "query_text": "I need a model for generating creative writing and stories",
        "user_context": {
            "use_case": "creative_writing",
            "constraints": {
                "max_model_size_gb": 10,
                "preferred_language": "en"
            },
            "preferences": {
                "task": "text-generation",
                "license": "apache-2.0"
            }
        },
        "top_k": 5
    }


@pytest.fixture
def sample_model_metadata() -> Dict[str, Any]:
    """
    Sample model metadata dictionary.
    
    Returns:
        Dictionary with sample model metadata
    """
    return {
        "model_id": "gpt2",
        "model_name": "GPT-2",
        "description": "Generative Pre-trained Transformer 2",
        "author": "openai",
        "downloads": 1000000,
        "likes": 50000,
        "tags": ["text-generation", "gpt2", "english"],
        "task": "text-generation",
        "library_name": "transformers",
        "pipeline_tag": "text-generation",
        "base_model": None,
        "model_type": "gpt2",
        "size_on_disk_gb": 0.5,
        "parameters": "124M",
        "quantization": None,
        "license": "mit",
        "huggingface_url": "https://huggingface.co/gpt2"
    }


@pytest.fixture
def sample_model_metadata_list() -> List[Dict[str, Any]]:
    """
    List of sample model metadata dictionaries.
    
    Returns:
        List of model metadata dictionaries
    """
    return [
        {
            "model_id": "gpt2",
            "model_name": "GPT-2",
            "description": "Generative Pre-trained Transformer 2",
            "author": "openai",
            "downloads": 1000000,
            "likes": 50000,
            "tags": ["text-generation", "gpt2", "english"],
            "task": "text-generation",
            "library_name": "transformers",
            "pipeline_tag": "text-generation",
            "base_model": None,
            "model_type": "gpt2",
            "size_on_disk_gb": 0.5,
            "parameters": "124M",
            "quantization": None,
            "license": "mit",
            "huggingface_url": "https://huggingface.co/gpt2"
        },
        {
            "model_id": "distilgpt2",
            "model_name": "DistilGPT-2",
            "description": "Distilled version of GPT-2",
            "author": "huggingface",
            "downloads": 1200000,
            "likes": 60000,
            "tags": ["text-generation", "distil", "lightweight"],
            "task": "text-generation",
            "library_name": "transformers",
            "pipeline_tag": "text-generation",
            "base_model": "gpt2",
            "model_type": "gpt2",
            "size_on_disk_gb": 0.25,
            "parameters": "82M",
            "quantization": None,
            "license": "apache-2.0",
            "huggingface_url": "https://huggingface.co/distilgpt2"
        }
    ]


@pytest.fixture
def sample_query_embedding() -> np.ndarray:
    """
    Sample query embedding vector.
    
    Returns:
        NumPy array with query embedding
    """
    np.random.seed(42)
    return np.random.rand(128).astype(np.float32)


@pytest.fixture
def sample_user_context() -> Dict[str, Any]:
    """
    Sample user context dictionary.
    
    Returns:
        Dictionary with user context
    """
    return {
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

