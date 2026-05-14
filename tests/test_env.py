import numpy as np
import pandas as pd

from src.env.hf_recommender_env import HFRecommenderEnv


def test_hf_recommender_env_step_reset():
    features_df = pd.DataFrame(
        {
            "model_id": ["m1", "m2"],
            "cost_score": [0.1, 0.2],
            "popularity_score": [0.3, 0.5],
            "recency_score": [0.6, 0.4],
            "diversity_score": [0.7, 0.2],
            "performance_score": [0.9, 0.8],
        }
    )
    embeddings = np.random.randn(2, 8)

    env = HFRecommenderEnv(features_df=features_df, embeddings=embeddings)
    obs, info = env.reset()

    assert obs.shape == (6,)
    assert isinstance(info, dict)

    action = np.ones(2, dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(reward, float)
    assert isinstance(next_obs, np.ndarray)
    assert terminated in (True, False)
    assert truncated in (True, False)
