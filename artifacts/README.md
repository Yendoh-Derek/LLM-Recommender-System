# Artifacts Directory

This directory contains trained models, embeddings, and preprocessing configurations exported from the Colab training environment.

## Directory Structure

```
artifacts/
├── models/              # Trained SAC agent models
│   └── sac_agent.pth    # SAC agent model file (PyTorch)
├── embeddings/          # Model embeddings
│   └── model_embeddings.npy  # NumPy array of model embeddings
├── preprocessing/       # Preprocessing configurations
│   └── preprocessing_config.json  # Preprocessing config JSON
└── model_catalog.json  # Model metadata catalog
```

## Expected Artifact Formats

### 1. SAC Agent Model (`models/sac_agent.pth`)

**Format:** PyTorch model file (`.pth` or `.pt`)

**Expected Structure:**
- Can be either:
  - State dictionary: `torch.save(model.state_dict(), 'sac_agent.pth')`
  - Full model: `torch.save(model, 'sac_agent.pth')`

**Loading Example:**
```python
import torch
from src.inference.rl_policy import RLPolicy

# Load model
model = torch.load('artifacts/models/sac_agent.pth')
if isinstance(model, dict):
    # If it's a state dict, you'll need to reconstruct the model architecture
    # model = YourSACModel()
    # model.load_state_dict(torch.load('artifacts/models/sac_agent.pth'))
    pass
else:
    # Full model loaded
    model.eval()

# Create RL policy
rl_policy = RLPolicy(model=model)
```

**Alternative Formats:**
- TensorFlow SavedModel: `artifacts/models/sac_agent/`
- ONNX: `artifacts/models/sac_agent.onnx`

### 2. Model Embeddings (`embeddings/model_embeddings.npy`)

**Format:** NumPy array file (`.npy`)

**Expected Shape:** `(num_models, embedding_dim)`
- `num_models`: Number of models in the catalog
- `embedding_dim`: Embedding dimension (e.g., 128, 256, 384, 512)

**Example:**
```python
import numpy as np

# Save embeddings (in Colab)
embeddings = np.array([...])  # Shape: (100, 128)
np.save('model_embeddings.npy', embeddings)

# Load embeddings (in API)
embeddings = np.load('artifacts/embeddings/model_embeddings.npy')
print(embeddings.shape)  # (100, 128)
```

**Alternative Formats:**
- Pickle: `model_embeddings.pkl`
- HDF5: `model_embeddings.h5`

### 3. Preprocessing Config (`preprocessing/preprocessing_config.json`)

**Format:** JSON file

**Expected Structure:**
```json
{
  "normalize": true,
  "encode_features": true,
  "feature_dim": 128,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "normalization_params": {
    "mean": [0.0, 0.0, ...],
    "std": [1.0, 1.0, ...]
  },
  "query_encoding": {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "max_length": 512
  },
  "context_encoding": {
    "use_case_dim": 32,
    "constraints_dim": 16,
    "preferences_dim": 16
  }
}
```

**Fields:**
- `normalize`: Whether to normalize features
- `encode_features`: Whether to encode text features
- `feature_dim`: Dimension of feature vectors
- `embedding_model`: Model name for text embeddings
- `normalization_params`: Mean and std for normalization (if applicable)

### 4. Model Catalog (`model_catalog.json`)

**Format:** JSON file

**Expected Structure:**
```json
{
  "0": {
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
    "base_model": null,
    "model_type": "gpt2",
    "size_on_disk_gb": 0.5,
    "parameters": "124M",
    "quantization": null,
    "license": "mit",
    "huggingface_url": "https://huggingface.co/gpt2"
  },
  "1": {
    "model_id": "distilgpt2",
    ...
  }
}
```

**Key Points:**
- Keys are string indices matching embedding array indices
- Each model entry should have all fields from `src/core/constants.py` MODEL_METADATA_FIELDS
- `model_id` and `huggingface_url` are required
- Other fields can have default values

## Exporting from Colab

### Example Colab Export Script

```python
# In your Colab notebook

import torch
import numpy as np
import json
from pathlib import Path

# 1. Export SAC model
torch.save(sac_agent.state_dict(), 'sac_agent.pth')
# Or full model:
# torch.save(sac_agent, 'sac_agent.pth')

# 2. Export embeddings
np.save('model_embeddings.npy', model_embeddings)
print(f"Embeddings shape: {model_embeddings.shape}")

# 3. Export preprocessing config
preprocessing_config = {
    "normalize": True,
    "encode_features": True,
    "feature_dim": model_embeddings.shape[1],
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
}
with open('preprocessing_config.json', 'w') as f:
    json.dump(preprocessing_config, f, indent=2)

# 4. Export model catalog
with open('model_catalog.json', 'w') as f:
    json.dump(model_catalog, f, indent=2)

# 5. Download files
from google.colab import files
files.download('sac_agent.pth')
files.download('model_embeddings.npy')
files.download('preprocessing_config.json')
files.download('model_catalog.json')
```

## File Size Considerations

- **SAC Model:** Typically 10-500 MB depending on architecture
- **Embeddings:** Size = `num_models × embedding_dim × 4 bytes` (float32)
  - Example: 100 models × 128 dims = ~50 KB
  - Example: 1000 models × 384 dims = ~1.5 MB
- **Config/Catalog:** Usually < 1 MB (JSON files)

## Version Control

**Note:** This directory is in `.gitignore` by default. Artifacts are typically:
- Too large for Git
- Environment-specific
- Generated from training

To track artifacts:
1. Use Git LFS for large files
2. Store artifacts in cloud storage (S3, GCS, etc.)
3. Document artifact versions separately

## Validation

After placing artifacts, validate them:

```python
# validation_script.py
import torch
import numpy as np
import json
from pathlib import Path

artifacts_dir = Path("artifacts")

# Validate SAC model
model_path = artifacts_dir / "models" / "sac_agent.pth"
if model_path.exists():
    model = torch.load(model_path)
    print(f"✓ SAC model loaded: {type(model)}")
else:
    print("✗ SAC model not found")

# Validate embeddings
embeddings_path = artifacts_dir / "embeddings" / "model_embeddings.npy"
if embeddings_path.exists():
    embeddings = np.load(embeddings_path)
    print(f"✓ Embeddings loaded: shape {embeddings.shape}")
else:
    print("✗ Embeddings not found")

# Validate config
config_path = artifacts_dir / "preprocessing" / "preprocessing_config.json"
if config_path.exists():
    with open(config_path) as f:
        config = json.load(f)
    print(f"✓ Config loaded: {config}")
else:
    print("✗ Config not found")

# Validate catalog
catalog_path = artifacts_dir / "model_catalog.json"
if catalog_path.exists():
    with open(catalog_path) as f:
        catalog = json.load(f)
    print(f"✓ Catalog loaded: {len(catalog)} models")
else:
    print("✗ Catalog not found")
```

## Next Steps

Once artifacts are in place:
1. Follow the integration guide in `docs/INTEGRATION.md`
2. Update code to load real artifacts instead of mocks
3. Test the integration with `pytest`
4. Verify API endpoints return real recommendations


