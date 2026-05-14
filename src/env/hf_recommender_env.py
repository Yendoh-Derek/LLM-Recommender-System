"""
hf_recommender_env.py
Wrapper for the root RL environment implementation.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class HFRecommenderEnv:
    """Wrapper for the root LLM recommendation environment."""

    def __init__(
        self,
        features_df: pd.DataFrame,
        embeddings: np.ndarray,
        reward_weights: Optional[Any] = None,
        top_k: int = 5,
        max_steps_per_episode: int = 10,
        state_feature_cols: Optional[List[str]] = None,
    ) -> None:
        root_env = importlib.import_module("rl_env")
        self.env = root_env.LLMRecommendationEnv(
            features_df=features_df,
            embeddings=embeddings,
            reward_weights=reward_weights,
            top_k=top_k,
            max_steps_per_episode=max_steps_per_episode,
            state_feature_cols=state_feature_cols,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self.env.reset(seed=seed, options=options)

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        return self.env.step(action)

    def render(self) -> None:
        return self.env.render()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)
