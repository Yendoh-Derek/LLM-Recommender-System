# API Design Specification

## Endpoints

### 1. POST /recommend
Get LLM recommendations based on user context.

**Request:**
```json
{
  "user_preferences": {
    "performance_priority": 0.8,
    "cost_priority": 0.2
  },
  "num_recommendations": 5,
  "exclude_models": []
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "model_id": "meta-llama/Llama-3.1-70B-Instruct",
      "score": 0.89,
      "cost_score": 0.05,
      "performance_score": 0.89,
      "diversity_score": 0.74
    }
  ],
  "metadata": {
    "inference_time_ms": 45,
    "model_version": "1.0.0"
  }
}
```

### 2. GET /models
List all available models.

### 3. GET /models/{model_id}
Get details for specific model.

### 4. POST /feedback
Record user feedback for model improvement.

## Implementation Notes

See `deployment_guide.md` for complete FastAPI implementation.
