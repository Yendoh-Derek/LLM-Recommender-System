# LLM Recommender System - Production Artifacts

**Export Date:** 2025-12-06  
**Version:** 1.0.0

## 📋 Overview

This package contains all trained models and data needed to deploy the LLM recommendation system.

**Performance:**
- Random Baseline: 4.79
- Trained Agent: 5.73 ± 0.18
- Improvement: **+19.69%**

## 📦 Package Contents

### Models
- `sac_best.pt` - Trained SAC agent (best checkpoint)
- `sac_config.json` - Model hyperparameters

### Data
- `features_with_scores.parquet` - Model features dataset (95 models)
- `feature_scaler.pkl` - Feature normalization scaler
- `training_metrics.csv` - Training history

### Indexes
- `model_embeddings.npy` - Model embeddings (95 × 384)
- `model_embeddings.faiss` - FAISS similarity index
- `model_id_mapping.json` - Index to model ID mapping

### Configuration
- `env_config.json` - RL environment configuration
- `deployment_config.json` - Deployment metadata

## 🚀 Quick Start (VS Code)

### 1. Extract Package
```bash
unzip production_artifacts.zip
cd production_artifacts
```

### 2. Install Dependencies
```bash
pip install torch numpy pandas faiss-cpu fastapi uvicorn pydantic
```

### 3. Run Inference Example
```python
import torch
from sac_agent import SACAgent
import pandas as pd

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
agent = SACAgent(state_dim=6, action_dim=95, device=device)
agent.load('sac_best.pt')

# Load features
features = pd.read_parquet('features_with_scores.parquet')

# Get recommendations
# (See deployment_guide.md for full implementation)
```

## 📊 Model Architecture

- **Algorithm:** Soft Actor-Critic (SAC)
- **State Dimension:** 6 (5 features + step progress)
- **Action Dimension:** 95 (number of models)
- **Hidden Layers:** 256 units
- **Total Parameters:** 198,785

## 🎯 Features

The model uses 5 core features for recommendations:

1. **Cost Score** (0.2 weight) - Model size & quantization
2. **Performance Score** (0.4 weight) - Benchmark results
3. **Popularity Score** (0.1 weight) - Downloads & likes
4. **Recency Score** (0.1 weight) - Model age
5. **Diversity Score** (0.2 weight) - Architecture variety

## 📚 Documentation

- `deployment_guide.md` - Complete deployment instructions
- `API_DESIGN.md` - API endpoint specifications
- `deployment_config.json` - Technical specifications

## 🔧 System Requirements

- **Python:** 3.8+
- **Memory:** 2GB RAM minimum
- **CPU:** Any modern processor
- **GPU:** Optional (for faster inference)

## 📞 Support

For issues or questions, refer to the deployment guide.

## 📄 License

All rights reserved.
