# Integration Guide: Colab Artifacts to FastAPI

This guide explains how to integrate trained models, embeddings, and preprocessing artifacts from your Colab training environment into the FastAPI backend.

## Overview

The API is designed with clear integration points where Colab outputs can be seamlessly plugged in. All placeholder/mock implementations are clearly marked with `TODO` comments and can be replaced with real implementations.

## Integration Points

### 1. Artifact Loading (`src/inference/load_artifacts.py`)

**File to modify:** `src/inference/load_artifacts.py`

**What to replace:**
- `load_sac_model()` - Replace mock loading with actual SAC model loading
- `load_embeddings()` - Replace mock loading with actual embedding loading
- `load_preprocessing_config()` - Replace mock loading with actual config loading
- `load_model_catalog()` - Replace mock loading with actual catalog loading

**Expected Artifact Formats:**

#### SAC Model
- **Format:** PyTorch model file (`.pth` or `.pt`) or TensorFlow SavedModel
- **Location:** `artifacts/models/sac_agent.pth` (or as configured)
- **Expected Interface:**
  ```python
  # PyTorch example
  import torch
  model = torch.load("artifacts/models/sac_agent.pth")
  model.eval()  # Set to evaluation mode
  ```

#### Model Embeddings
- **Format:** NumPy array (`.npy`) or pickle file (`.pkl`)
- **Location:** `artifacts/embeddings/model_embeddings.npy`
- **Expected Shape:** `(num_models, embedding_dim)`
- **Example:**
  ```python
  import numpy as np
  embeddings = np.load("artifacts/embeddings/model_embeddings.npy")
  # Shape: (100, 128) for 100 models with 128-dim embeddings
  ```

#### Preprocessing Config
- **Format:** JSON file
- **Location:** `artifacts/preprocessing/preprocessing_config.json`
- **Expected Structure:**
  ```json
  {
    "normalize": true,
    "encode_features": true,
    "feature_dim": 128,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "normalization_params": {
      "mean": [0.0, ...],
      "std": [1.0, ...]
    }
  }
  ```

#### Model Catalog
- **Format:** JSON file
- **Location:** `artifacts/model_catalog.json`
- **Expected Structure:**
  ```json
  {
    "0": {
      "model_id": "gpt2",
      "model_name": "GPT-2",
      "description": "...",
      "task": "text-generation",
      ...
    },
    "1": {
      "model_id": "distilgpt2",
      ...
    }
  }
  ```

### 2. RL Policy (`src/inference/rl_policy.py`)

**File to modify:** `src/inference/rl_policy.py`

**What to replace:**
- Replace `RLPolicy` class with your actual SAC agent implementation
- Implement `select_action()`, `predict()`, and `get_recommendation_scores()` methods

**Expected Interface:**
```python
class RLPolicy:
    def __init__(self, model):
        self.model = model
        self.is_loaded = True
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action (model index) given state."""
        # Your SAC agent action selection logic
        pass
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Get action probabilities/scores."""
        # Your SAC agent prediction logic
        pass
    
    def get_recommendation_scores(
        self,
        states: List[np.ndarray],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """Get scores for multiple states."""
        # Your SAC agent scoring logic
        pass
```

**Integration Steps:**
1. Load your trained SAC model in `load_artifacts.py`
2. Pass the loaded model to `RLPolicy` constructor
3. Replace mock methods with actual SAC agent inference

### 3. Preprocessing (`src/inference/preprocessing.py`)

**File to modify:** `src/inference/preprocessing.py`

**What to replace:**
- `encode_query()` - Replace with actual query encoding (e.g., sentence transformers)
- `extract_user_context_features()` - Replace with actual feature extraction
- `construct_state_vector()` - Replace with actual state construction logic

**Example Integration:**
```python
def encode_query(self, query_text: str) -> np.ndarray:
    """Encode query using sentence transformers."""
    from sentence_transformers import SentenceTransformer
    
    if not hasattr(self, '_encoder'):
        model_name = self.config.get("embedding_model", "all-MiniLM-L6-v2")
        self._encoder = SentenceTransformer(model_name)
    
    embedding = self._encoder.encode(query_text, convert_to_numpy=True)
    return embedding.astype(np.float32)
```

### 4. Similarity Search (`src/recommender/similarity_search.py`)

**File to modify:** `src/recommender/similarity_search.py`

**What to replace:**
- Initialize `SimilaritySearch` with actual embeddings from artifacts
- The search logic is already implemented, just needs real embeddings

**Integration Steps:**
1. Load embeddings in `load_artifacts.py`
2. Pass embeddings to `SimilaritySearch` constructor:
   ```python
   from src.recommender.similarity_search import SimilaritySearch
   from src.inference.load_artifacts import get_artifact_loader
   
   loader = get_artifact_loader()
   embeddings = loader.embeddings  # Loaded from artifacts
   similarity_search = SimilaritySearch(embeddings=embeddings)
   ```

## Step-by-Step Integration Process

### Step 1: Export Artifacts from Colab

1. **Export SAC Model:**
   ```python
   # In Colab
   torch.save(sac_agent.state_dict(), 'sac_agent.pth')
   # Or save entire model
   torch.save(sac_agent, 'sac_agent.pth')
   ```

2. **Export Embeddings:**
   ```python
   # In Colab
   import numpy as np
   np.save('model_embeddings.npy', embeddings_array)
   ```

3. **Export Preprocessing Config:**
   ```python
   # In Colab
   import json
   config = {
       "normalize": True,
       "embedding_model": "your-model-name",
       ...
   }
   with open('preprocessing_config.json', 'w') as f:
       json.dump(config, f, indent=2)
   ```

4. **Export Model Catalog:**
   ```python
   # In Colab
   import json
   with open('model_catalog.json', 'w') as f:
       json.dump(model_catalog_dict, f, indent=2)
   ```

### Step 2: Transfer Artifacts to API Project

1. Download artifacts from Colab (or use Google Drive sync)
2. Place them in the appropriate directories:
   ```
   artifacts/
   ├── models/
   │   └── sac_agent.pth
   ├── embeddings/
   │   └── model_embeddings.npy
   ├── preprocessing/
   │   └── preprocessing_config.json
   └── model_catalog.json
   ```

### Step 3: Update Code

1. **Update `load_artifacts.py`:**
   - Replace mock loading functions with actual loading code
   - Use the correct file paths and formats

2. **Update `rl_policy.py`:**
   - Replace `RLPolicy` with your SAC agent class
   - Ensure methods match the expected interface

3. **Update `preprocessing.py`:**
   - Replace mock encoding with actual embedding model
   - Use config from preprocessing config file

4. **Update `recommend.py` route:**
   - Replace mock recommendation generation with actual pipeline:
     ```python
     # Replace this:
     recommendations = _generate_mock_recommendations(...)
     
     # With this:
     state = preprocessor.preprocess_request(query_text, user_context)
     rl_scores = rl_policy.get_recommendation_scores([state], top_k=top_k)
     similarity_results = similarity_search.search(query_embedding, top_k=top_k)
     # ... combine and rank
     ```

### Step 4: Test Integration

1. Run tests to ensure nothing breaks:
   ```bash
   pytest
   ```

2. Test the API endpoints:
   ```bash
   # Start server
   uvicorn src.api.main:app --reload
   
   # Test recommendation endpoint
   curl -X POST "http://localhost:8000/recommend" \
     -H "Content-Type: application/json" \
     -d '{"query_text": "I need a text generation model"}'
   ```

3. Check logs for any errors or warnings

## Environment Variables

Update your `.env` file with artifact paths (if different from defaults):

```env
MODEL_NAME=sac_agent
EMBEDDING_MODEL_NAME=model_embeddings
PREPROCESSING_CONFIG_PATH=artifacts/preprocessing/config.json
```

## Troubleshooting

### Common Issues

1. **Model loading errors:**
   - Ensure PyTorch/TensorFlow versions match between Colab and API
   - Check model file format compatibility

2. **Embedding dimension mismatch:**
   - Verify embedding dimensions match between preprocessing and similarity search
   - Check config file has correct `feature_dim` or `embedding_dim`

3. **State vector shape errors:**
   - Ensure state vector construction matches what SAC agent expects
   - Check preprocessing config for correct dimensions

4. **Performance issues:**
   - Consider model quantization for faster inference
   - Use GPU if available (update device in config)

## Next Steps After Integration

1. Update `/health/ready` endpoint to check if artifacts are loaded
2. Add model versioning support
3. Implement caching for frequently accessed models
4. Add monitoring and logging for inference performance
5. Consider batch processing for multiple requests

## Support

If you encounter issues during integration:
1. Check the logs in `src/core/logger.py` output
2. Verify artifact file formats match expected formats
3. Ensure all dependencies are installed (check `requirements.txt`)
4. Review TODO comments in code for integration hints


