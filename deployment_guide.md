# 🚀 VS Code Deployment Guide - LLM Recommender System

## 📋 Overview

This guide shows you how to deploy the trained LLM recommender system as a production API using FastAPI in VS Code.

**What you're deploying:**
- Trained SAC agent (19.69% improvement over baseline)
- 95 model database with features
- FAISS similarity search
- REST API for recommendations

---

## 📁 Step 1: Project Setup

### 1.1 Create Project Structure

```bash
# Create project directory
mkdir llm-recommender-api
cd llm-recommender-api

# Create folder structure
mkdir -p app/{models,routes,services,schemas}
mkdir -p data/{models,features,indexes}
mkdir config
mkdir tests
```

### 1.2 Extract Production Artifacts

```bash
# Extract the zip file from Colab
unzip production_artifacts.zip

# Move files to appropriate locations
mv production_artifacts/sac_best.pt data/models/
mv production_artifacts/features_with_scores.parquet data/features/
mv production_artifacts/model_embeddings.* data/indexes/
mv production_artifacts/model_id_mapping.json data/indexes/
mv production_artifacts/*.json config/
```

### 1.3 Final Structure

```
llm-recommender-api/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app
│   ├── models/
│   │   ├── __init__.py
│   │   └── sac_agent.py        # SAC agent class
│   ├── routes/
│   │   ├── __init__.py
│   │   └── recommend.py        # API endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── recommender.py      # Recommendation logic
│   │   └── model_loader.py     # Load artifacts
│   └── schemas/
│       ├── __init__.py
│       └── requests.py         # Pydantic models
├── data/
│   ├── models/
│   │   └── sac_best.pt
│   ├── features/
│   │   └── features_with_scores.parquet
│   └── indexes/
│       ├── model_embeddings.npy
│       ├── model_embeddings.faiss
│       └── model_id_mapping.json
├── config/
│   ├── deployment_config.json
│   ├── sac_config.json
│   └── env_config.json
├── tests/
│   └── test_api.py
├── requirements.txt
├── Dockerfile
├── .env
└── README.md
```

---

## 📦 Step 2: Dependencies

### 2.1 Create `requirements.txt`

```txt
# Web Framework
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
pydantic-settings==2.1.0

# ML/RL
torch==2.1.2
numpy==1.24.3
pandas==2.1.4

# Vector Search
faiss-cpu==1.7.4

# Utilities
python-dotenv==1.0.0
httpx==0.26.0

# Development
pytest==7.4.3
pytest-asyncio==0.23.3
black==23.12.1
```

### 2.2 Install

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🔧 Step 3: Core Implementation

### 3.1 SAC Agent (`app/models/sac_agent.py`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.sigmoid(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(action * (1 - action) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

class SACAgent:
    def __init__(self, state_dim, action_dim, device='cpu'):
        self.device = device
        self.actor = Actor(state_dim, action_dim).to(device)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor.eval()
    
    def select_action(self, state: np.ndarray, evaluate=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate:
                mean, _ = self.actor(state)
                action = torch.sigmoid(mean)
            else:
                action, _ = self.actor.sample(state)
        return action.cpu().numpy()[0]
```

### 3.2 Model Loader (`app/services/model_loader.py`)

```python
import json
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from app.models.sac_agent import SACAgent

class ModelLoader:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.agent = None
        self.features_df = None
        self.embeddings = None
        self.faiss_index = None
        self.model_id_mapping = None
        
    def load_all(self):
        """Load all artifacts."""
        # Load SAC agent
        model_path = self.data_dir / "models" / "sac_best.pt"
        config_path = self.data_dir.parent / "config" / "sac_config.json"
        
        with open(config_path) as f:
            config = json.load(f)
        
        state_dim = config['architecture']['state_dim']
        action_dim = config['architecture']['action_dim']
        
        self.agent = SACAgent(state_dim, action_dim)
        self.agent.load(str(model_path))
        
        # Load features
        features_path = self.data_dir / "features" / "features_with_scores.parquet"
        self.features_df = pd.read_parquet(features_path)
        
        # Load embeddings
        emb_path = self.data_dir / "indexes" / "model_embeddings.npy"
        self.embeddings = np.load(emb_path)
        
        # Load FAISS index
        faiss_path = self.data_dir / "indexes" / "model_embeddings.faiss"
        self.faiss_index = faiss.read_index(str(faiss_path))
        
        # Load model ID mapping
        mapping_path = self.data_dir / "indexes" / "model_id_mapping.json"
        with open(mapping_path) as f:
            self.model_id_mapping = json.load(f)
        
        return self

# Global instance
loader = ModelLoader(Path("data"))
```

### 3.3 Recommender Service (`app/services/recommender.py`)

```python
import numpy as np
from typing import List, Dict
from app.services.model_loader import loader

class RecommenderService:
    def __init__(self):
        self.agent = loader.agent
        self.features_df = loader.features_df
        self.faiss_index = loader.faiss_index
        self.model_id_mapping = loader.model_id_mapping
    
    def get_recommendations(
        self,
        num_recommendations: int = 5,
        user_preferences: Dict = None
    ) -> List[Dict]:
        """Get top-K recommendations using trained SAC agent."""
        
        # Create state from current features
        state = self._create_state()
        
        # Get action from agent
        action = self.agent.select_action(state, evaluate=True)
        
        # Select top-K models based on action weights
        top_k_indices = np.argsort(action)[-num_recommendations:][::-1]
        
        # Build response
        recommendations = []
        for idx in top_k_indices:
            model_info = self.features_df.iloc[idx]
            recommendations.append({
                "model_id": model_info['model_id'],
                "score": float(action[idx]),
                "cost_score": float(model_info.get('cost_score', 0.5)),
                "performance_score": float(model_info.get('performance_score', 0.5)),
                "popularity_score": float(model_info.get('popularity_score', 0.5)),
                "diversity_score": float(model_info.get('diversity_score', 0.5)),
                "num_parameters": float(model_info.get('num_parameters', 0)) if not pd.isna(model_info.get('num_parameters')) else None,
                "quantization": model_info.get('quantization')
            })
        
        return recommendations
    
    def _create_state(self) -> np.ndarray:
        """Create state representation from features."""
        feature_cols = ['cost_score', 'popularity_score', 'recency_score',
                       'diversity_score', 'performance_score']
        state = self.features_df[feature_cols].fillna(0.5).mean().values
        state = np.append(state, 0.0)  # Add step progress
        return state.astype(np.float32)
    
    def find_similar_models(self, model_id: str, k: int = 5) -> List[Dict]:
        """Find similar models using FAISS."""
        # Get model index
        model_indices = {v: int(k) for k, v in self.model_id_mapping.items()}
        
        if model_id not in model_indices:
            return []
        
        idx = model_indices[model_id]
        query_embedding = loader.embeddings[idx:idx+1]
        
        # Search FAISS
        distances, indices = self.faiss_index.search(query_embedding, k + 1)
        
        # Skip first result (query itself)
        similar = []
        for i, dist in zip(indices[0][1:], distances[0][1:]):
            similar_model = self.features_df.iloc[i]
            similar.append({
                "model_id": similar_model['model_id'],
                "similarity": float(1 / (1 + dist))  # Convert distance to similarity
            })
        
        return similar

recommender = RecommenderService()
```

### 3.4 Request Schemas (`app/schemas/requests.py`)

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, List

class RecommendRequest(BaseModel):
    num_recommendations: int = Field(default=5, ge=1, le=20)
    user_preferences: Optional[Dict] = None
    exclude_models: Optional[List[str]] = None

class SimilarModelsRequest(BaseModel):
    model_id: str
    num_similar: int = Field(default=5, ge=1, le=20)

class RecommendResponse(BaseModel):
    recommendations: List[Dict]
    metadata: Dict

class ModelDetailResponse(BaseModel):
    model_id: str
    features: Dict
    benchmarks: Optional[Dict] = None
```

### 3.5 API Routes (`app/routes/recommend.py`)

```python
from fastapi import APIRouter, HTTPException
from app.schemas.requests import (
    RecommendRequest, SimilarModelsRequest,
    RecommendResponse, ModelDetailResponse
)
from app.services.recommender import recommender
import time

router = APIRouter(prefix="/api/v1", tags=["recommendations"])

@router.post("/recommend", response_model=RecommendResponse)
async def get_recommendations(request: RecommendRequest):
    """Get LLM recommendations using trained SAC agent."""
    try:
        start_time = time.time()
        
        recommendations = recommender.get_recommendations(
            num_recommendations=request.num_recommendations,
            user_preferences=request.user_preferences
        )
        
        inference_time = (time.time() - start_time) * 1000
        
        return {
            "recommendations": recommendations,
            "metadata": {
                "inference_time_ms": round(inference_time, 2),
                "model_version": "1.0.0",
                "num_results": len(recommendations)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/similar")
async def find_similar(request: SimilarModelsRequest):
    """Find similar models using FAISS."""
    try:
        similar = recommender.find_similar_models(
            model_id=request.model_id,
            k=request.num_similar
        )
        
        if not similar:
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model_id} not found"
            )
        
        return {"similar_models": similar}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_models():
    """List all available models."""
    models = recommender.features_df[['model_id', 'author', 'downloads', 'likes']].to_dict('records')
    return {"models": models, "total": len(models)}

@router.get("/models/{model_id}")
async def get_model_detail(model_id: str):
    """Get details for a specific model."""
    model_data = recommender.features_df[
        recommender.features_df['model_id'] == model_id
    ]
    
    if len(model_data) == 0:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = model_data.iloc[0].to_dict()
    
    return {
        "model_id": model_id,
        "features": model_info
    }
```

### 3.6 Main Application (`app/main.py`)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.routes import recommend
from app.services.model_loader import loader

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models
    print("Loading models...")
    loader.load_all()
    print("Models loaded successfully!")
    yield
    # Shutdown: Cleanup if needed
    print("Shutting down...")

app = FastAPI(
    title="LLM Recommender API",
    description="RL-powered LLM recommendation system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(recommend.router)

@app.get("/")
async def root():
    return {
        "message": "LLM Recommender API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 🚀 Step 4: Running the API

### 4.1 Development Mode

```bash
# Run with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4.2 Test the API

```bash
# Health check
curl http://localhost:8000/health

# Get recommendations
curl -X POST http://localhost:8000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{"num_recommendations": 5}'

# Find similar models
curl -X POST http://localhost:8000/api/v1/similar \
  -H "Content-Type: application/json" \
  -d '{"model_id": "meta-llama/Llama-3.1-70B-Instruct", "num_similar": 3}'

# List all models
curl http://localhost:8000/api/v1/models
```

### 4.3 Interactive Docs

Visit: `http://localhost:8000/docs`

---

## 🐳 Step 5: Docker Deployment

### 5.1 Create `Dockerfile`

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/
COPY data/ ./data/
COPY config/ ./config/

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5.2 Build and Run

```bash
# Build image
docker build -t llm-recommender:latest .

# Run container
docker run -p 8000:8000 llm-recommender:latest
```

---

## 📊 Step 6: Testing

### 6.1 Create `tests/test_api.py`

```python
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_recommend():
    response = client.post(
        "/api/v1/recommend",
        json={"num_recommendations": 5}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["recommendations"]) == 5
    assert "inference_time_ms" in data["metadata"]

def test_list_models():
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    assert "models" in response.json()
```

### 6.2 Run Tests

```bash
pytest tests/ -v
```

---

## 🌐 Step 7: Production Deployment

### Option A: AWS EC2

```bash
# 1. Launch EC2 instance (t3.medium recommended)
# 2. Install Docker
# 3. Copy artifacts and Dockerfile
# 4. Run container
docker run -d -p 80:8000 llm-recommender:latest
```

### Option B: Google Cloud Run

```bash
# 1. Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/llm-recommender

# 2. Deploy to Cloud Run
gcloud run deploy llm-recommender \
  --image gcr.io/PROJECT_ID/llm-recommender \
  --platform managed \
  --memory 2Gi
```

### Option C: Render/Railway

1. Connect GitHub repo
2. Auto-deploy from main branch
3. Set environment variables

---

## ⚡ Performance Optimization

### Caching Recommendations

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_recommendations(num_recs: int):
    return recommender.get_recommendations(num_recs)
```

### Async Processing

```python
import asyncio

async def batch_recommend(requests: List[RecommendRequest]):
    tasks = [get_recommendations(req) for req in requests]
    return await asyncio.gather(*tasks)
```

---

## 📈 Monitoring

### Add Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@router.post("/recommend")
async def get_recommendations(request: RecommendRequest):
    logger.info(f"Recommendation request: {request.num_recommendations}")
    # ... rest of code
```

### Metrics with Prometheus

```python
from prometheus_client import Counter, Histogram

request_count = Counter('requests_total', 'Total requests')
request_duration = Histogram('request_duration_seconds', 'Request duration')
```

---

## 🎉 Deployment Complete!

Your LLM recommender is now live and ready to serve recommendations!

**Next steps:**
- Set up CI/CD pipeline
- Add authentication (JWT tokens)
- Implement rate limiting
- Set up monitoring dashboards
- Collect user feedback for model improvement