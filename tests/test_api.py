from fastapi.testclient import TestClient

from src.api.fastapi_app import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommend_endpoint():
    payload = {
        "task": "text classification",
        "constraints": {"max_cost": 0.5, "min_performance": 0.8},
        "top_k": 3,
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) == 3
    for recommendation in data["recommendations"]:
        assert "model_id" in recommendation
        assert "score" in recommendation


def test_feedback_endpoint():
    payload = {
        "recommendation_id": "rec_123",
        "model_id": "author/model",
        "rating": 4,
        "user_id": "user_456",
    }
    response = client.post("/feedback", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "Feedback saved" in data["message"]
