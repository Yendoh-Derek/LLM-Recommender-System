"""
inference.py
Inference pipeline for model recommendation.
"""
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..env.hf_recommender_env import HFRecommenderEnv
from ..rl.sac_agent import SACAgent
from .ranking import rank_candidates


def _build_dummy_feature_catalog(n_models: int = 10) -> pd.DataFrame:
    """Create a fallback model catalog for recommendation."""
    return pd.DataFrame(
        {
            "model_id": [f"demo-model-{i + 1}" for i in range(n_models)],
            "cost_score": np.linspace(0.2, 0.8, n_models),
            "popularity_score": np.linspace(0.3, 0.9, n_models),
            "recency_score": np.linspace(0.7, 0.1, n_models),
            "diversity_score": np.linspace(0.4, 0.6, n_models),
            "performance_score": np.linspace(0.5, 0.95, n_models),
        }
    )


def _load_feature_catalog() -> pd.DataFrame:
    """Load a model catalog from disk if available, otherwise use a dummy dataset."""
    possible_paths = [
        Path("data") / "processed" / "model_features.csv",
        Path("data") / "processed" / "features.csv",
        Path("artifacts") / "indexes" / "model_catalog.csv",
    ]

    for path in possible_paths:
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:
                continue

    return _build_dummy_feature_catalog()


def _build_embeddings(n_models: int, embedding_dim: int = 16) -> np.ndarray:
    """Create fallback embeddings for the recommendation catalog."""
    rnd = np.random.default_rng(42)
    embeddings = rnd.standard_normal((n_models, embedding_dim)).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def recommend(task_embedding: Any, top_k: int = 5) -> List[Dict[str, Any]]:
    """Return top-K recommended models for a given task embedding."""
    features_df = _load_feature_catalog()
    embeddings = _build_embeddings(len(features_df))

    env = HFRecommenderEnv(
        features_df=features_df,
        embeddings=embeddings,
        top_k=top_k,
        max_steps_per_episode=1,
    )

    observation, _ = env.reset()
    agent = SACAgent(state_dim=observation.shape[0], action_dim=env.n_models, device="cpu")
    action = agent.select_action(observation, evaluate=True)
    _, _, _, _, info = env.step(action)

    selected_indices = info.get("selected_models", [])
    recommendations = []

    for idx in selected_indices:
        row = features_df.iloc[idx]
        recommendations.append(
            {
                "model_id": row["model_id"],
                "score": float(row.get("performance_score", 0.0)),
                "metadata": {
                    "cost_score": float(row.get("cost_score", 0.0)),
                    "popularity_score": float(row.get("popularity_score", 0.0)),
                    "recency_score": float(row.get("recency_score", 0.0)),
                    "diversity_score": float(row.get("diversity_score", 0.0)),
                },
            }
        )

    return rank_candidates(recommendations)[:top_k]
