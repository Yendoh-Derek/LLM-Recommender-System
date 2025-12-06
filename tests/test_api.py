"""
Tests for API endpoints.
"""
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from tests.conftest import client, sample_recommendation_request


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_health_check(self, client: TestClient):
        """Test the /health endpoint."""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
    
    def test_health_check_trailing_slash(self, client: TestClient):
        """Test the /health/ endpoint with trailing slash."""
        response = client.get("/health/")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_readiness_check(self, client: TestClient):
        """Test the /health/ready endpoint."""
        response = client.get("/health/ready")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "ready" in data
        assert "status" in data
        assert "timestamp" in data


class TestRecommendEndpoint:
    """Tests for the /recommend endpoint."""
    
    def test_recommend_basic(self, client: TestClient, sample_recommendation_request):
        """Test basic recommendation request."""
        response = client.post("/recommend", json=sample_recommendation_request)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "recommendations" in data
        assert "query_text" in data
        assert "total_results" in data
        assert "timestamp" in data
        assert isinstance(data["recommendations"], list)
        assert len(data["recommendations"]) > 0
    
    def test_recommend_minimal_request(self, client: TestClient):
        """Test recommendation with minimal request (only query_text)."""
        request = {
            "query_text": "I need a text generation model"
        }
        response = client.post("/recommend", json=request)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "recommendations" in data
        assert data["query_text"] == request["query_text"]
    
    def test_recommend_with_top_k(self, client: TestClient):
        """Test recommendation with top_k parameter."""
        request = {
            "query_text": "I need a model for translation",
            "top_k": 3
        }
        response = client.post("/recommend", json=request)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["recommendations"]) <= 3
    
    def test_recommend_recommendation_structure(self, client: TestClient, sample_recommendation_request):
        """Test that recommendations have the correct structure."""
        response = client.post("/recommend", json=sample_recommendation_request)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        recommendation = data["recommendations"][0]
        assert "model" in recommendation
        assert "score" in recommendation
        assert "explanation" in recommendation
        assert "rank" in recommendation
        
        # Check model metadata structure
        model = recommendation["model"]
        assert "model_id" in model
        assert "model_name" in model
        assert "huggingface_url" in model
    
    def test_recommend_invalid_request_empty_query(self, client: TestClient):
        """Test recommendation with empty query text."""
        request = {
            "query_text": ""
        }
        response = client.post("/recommend", json=request)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_recommend_invalid_request_missing_query(self, client: TestClient):
        """Test recommendation with missing query_text."""
        request = {
            "user_context": {"use_case": "test"}
        }
        response = client.post("/recommend", json=request)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_recommend_trailing_slash(self, client: TestClient, sample_recommendation_request):
        """Test the /recommend/ endpoint with trailing slash."""
        response = client.post("/recommend/", json=sample_recommendation_request)
        assert response.status_code == status.HTTP_200_OK


class TestMetadataEndpoint:
    """Tests for the /metadata endpoint."""
    
    def test_metadata_basic(self, client: TestClient):
        """Test basic metadata request."""
        response = client.get("/metadata")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "available_models_count" in data
        assert "supported_tasks" in data
        assert "available_features" in data
        assert "api_version" in data
        assert "timestamp" in data
        assert isinstance(data["supported_tasks"], list)
        assert isinstance(data["available_features"], list)
    
    def test_metadata_trailing_slash(self, client: TestClient):
        """Test the /metadata/ endpoint with trailing slash."""
        response = client.get("/metadata/")
        assert response.status_code == status.HTTP_200_OK


class TestRootEndpoint:
    """Tests for the root endpoint."""
    
    def test_root_endpoint(self, client: TestClient):
        """Test the root / endpoint."""
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data
        assert "health" in data

