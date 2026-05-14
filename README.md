# LLM Recommender API

A production-ready FastAPI backend for serving LLM recommendations powered by a Reinforcement Learning (SAC) agent trained in Colab.

## Overview

This API serves recommendations for open-source LLMs from Hugging Face. The recommendation engine uses a Soft Actor-Critic (SAC) RL agent that will be trained in Colab. Currently, the API uses mock data and placeholders, making it easy to integrate trained models and artifacts once they're available.

## Features

- FastAPI-based REST API with async support
- Modular architecture for easy integration of trained models
- Structured logging and error handling
- Production-ready configuration management
- Comprehensive request/response validation with Pydantic
- Health and readiness check endpoints
- Comprehensive unit tests (52 tests covering all modules)
- Integration documentation for Colab artifacts

## Project Structure

```
llm-recommender-api/
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI application entry point
│   │   ├── routes/              # API route handlers
│   │   │   ├── recommend.py    # Recommendation endpoint
│   │   │   ├── health.py       # Health check endpoint
│   │   │   └── metadata.py     # Metadata endpoint
│   │   └── schemas/            # Pydantic request/response models
│   │       ├── request_schema.py
│   │       └── response_schema.py
│   ├── core/
│   │   ├── config.py           # Centralized configuration
│   │   ├── logger.py           # Logging setup
│   │   └── constants.py        # Application constants
│   ├── inference/
│   │   ├── load_artifacts.py   # Model/artifact loading (placeholder)
│   │   ├── preprocessing.py    # Feature preprocessing (placeholder)
│   │   ├── rl_policy.py        # SAC agent interface (placeholder)
│   │   └── postprocessing.py   # Result formatting
│   ├── recommender/
│   │   ├── ranker.py           # Scoring and ranking logic
│   │   ├── recipe_generator.py # Model recipe generation
│   │   └── similarity_search.py # Embedding similarity search
│   └── utils/
│       ├── common.py           # Common utilities
│       └── timer.py            # Performance timing
├── artifacts/                  # Trained models and preprocessing artifacts (from Colab)
├── tests/                     # Unit tests
├── scripts/                   # Utility scripts
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
└── README.md                  # This file
```

## Installation

### Prerequisites

- Python 3.11 or higher
- pip or poetry for dependency management

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd llm-recommender-api
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file (copy from `.env.example`):
```bash
cp .env.example .env
```

5. Update `.env` with your configuration (optional for development).

## Running the API

### Development Mode

```bash
python -m src.api.main
```

Or using uvicorn directly:
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at:
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc

## API Endpoints

### Health Check
- `GET /health` - Basic health check
- `GET /health/ready` - Readiness check (checks if artifacts are loaded)

### Recommendations
- `POST /recommend` - Get LLM recommendations based on query and user context
  - Request body: `{"query_text": "...", "user_context": {...}, "top_k": 10}`
  - Returns: Ranked list of model recommendations with scores and explanations

### Metadata
- `GET /metadata` - Get available models, supported tasks, and features

## Configuration

All configuration is managed through `src/core/config.py` and environment variables. Key settings:

- `API_TITLE`: API title
- `API_VERSION`: API version
- `DEBUG`: Enable debug mode
- `HOST`: Server host
- `PORT`: Server port
- `LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)
- `LOG_FORMAT`: Log format (json or text)

## Testing

Run tests with pytest:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src tests/
```

## Integration with Colab Artifacts

This API is designed to integrate seamlessly with trained models and artifacts from Colab. 

**📖 See [docs/INTEGRATION.md](docs/INTEGRATION.md) for detailed integration instructions.**

**📁 See [artifacts/README.md](artifacts/README.md) for expected artifact formats.**

### Quick Integration Steps

1. **Export artifacts from Colab:**
   - SAC agent model (`.pth` file)
   - Model embeddings (`.npy` file)
   - Preprocessing config (`.json` file)
   - Model catalog (`.json` file)

2. **Place artifacts in the `artifacts/` directory:**
   ```
   artifacts/
   ├── models/sac_agent.pth
   ├── embeddings/model_embeddings.npy
   ├── preprocessing/preprocessing_config.json
   └── model_catalog.json
   ```

3. **Update integration points:**
   - `src/inference/load_artifacts.py` - Replace mock loading with real artifact loading
   - `src/inference/rl_policy.py` - Replace mock RLPolicy with actual SAC agent
   - `src/inference/preprocessing.py` - Replace mock preprocessing with real transformations
   - `src/api/routes/recommend.py` - Replace mock recommendations with real inference pipeline

4. **Test the integration:**
   ```bash
   pytest  # Run tests
   uvicorn src.api.main:app --reload  # Start server
   ```

All integration points are clearly marked with `TODO` comments in the code.




