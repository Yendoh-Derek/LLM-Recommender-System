"""
Reinforcement Learning Environment for LLM Recommendation System.
Compatible with OpenAI Gym interface for SAC training.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces


@dataclass
class RewardWeights:
    """Weights for different reward components."""
    performance: float = 0.4    # α: Quality of recommendations
    cost: float = 0.2           # β: Penalize expensive models
    diversity: float = 0.2      # γ: Encourage varied recommendations
    popularity: float = 0.1     # δ: Consider user preferences
    recency: float = 0.1        # ε: Favor newer models
    
    def validate(self):
        """Ensure weights sum to 1.0"""
        total = (self.performance + self.cost + self.diversity + 
                self.popularity + self.recency)
        assert abs(total - 1.0) < 0.01, f"Weights must sum to 1.0, got {total}"


class LLMRecommendationEnv(gym.Env):
    """
    Gym environment for training LLM recommendation agent.
    
    State Space:
        - Model features (cost, popularity, recency, diversity, performance)
        - User context (simulated preferences)
        - Recommendation history
        
    Action Space:
        - Continuous: Selection weights for top-K models
        
    Reward:
        Composite score balancing performance, cost, diversity, and feedback
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        features_df: pd.DataFrame,
        embeddings: np.ndarray,
        reward_weights: Optional[RewardWeights] = None,
        top_k: int = 5,
        max_steps_per_episode: int = 10,
        state_feature_cols: Optional[List[str]] = None
    ):
        """
        Initialize the RL environment.
        
        Args:
            features_df: DataFrame with model features
            embeddings: Model embeddings array (N x D)
            reward_weights: Reward function weights
            top_k: Number of recommendations per step
            max_steps_per_episode: Maximum steps before episode ends
            state_feature_cols: Columns to use as state features
        """
        super().__init__()
        
        self.features_df = features_df.reset_index(drop=True)
        self.embeddings = embeddings
        self.reward_weights = reward_weights or RewardWeights()
        self.reward_weights.validate()
        self.top_k = top_k
        self.max_steps = max_steps_per_episode
        
        # Default state feature columns
        if state_feature_cols is None:
            self.state_feature_cols = [
                'cost_score', 'popularity_score', 'recency_score',
                'diversity_score', 'performance_score'
            ]
        else:
            self.state_feature_cols = state_feature_cols
        
        # Validate features exist
        for col in self.state_feature_cols:
            if col not in self.features_df.columns:
                raise ValueError(f"Feature column '{col}' not found in features_df")
        
        # Environment state
        self.n_models = len(self.features_df)
        self.current_step = 0
        self.recommended_models = []  # Track recommended model indices
        self.episode_rewards = []
        
        # Define action and observation spaces
        # Action: weights for selecting top-K models from candidate set
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_models,),
            dtype=np.float32
        )
        
        # Observation: aggregated features + history
        state_dim = len(self.state_feature_cols) + 1  # +1 for step count
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Initialize user preferences (simulated)
        self._init_user_preferences()
    
    def _init_user_preferences(self):
        """Initialize simulated user preferences."""
        # Random preference vector in embedding space
        self.user_preference_embedding = np.random.randn(self.embeddings.shape[1])
        self.user_preference_embedding /= np.linalg.norm(self.user_preference_embedding)
        
        # Random preference weights for features
        self.user_feature_preferences = {
            'performance': np.random.uniform(0.5, 1.0),
            'cost': np.random.uniform(0.0, 0.5),  # Lower cost preference
            'recency': np.random.uniform(0.3, 0.8),
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        
        Returns:
            observation: Initial state
            info: Additional info dict
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.recommended_models = []
        self.episode_rewards = []
        
        # Optionally re-initialize user preferences
        if options and options.get('new_user', False):
            self._init_user_preferences()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Selection weights for models
            
        Returns:
            observation: Next state
            reward: Reward for this step
            terminated: Whether episode ended naturally
            truncated: Whether episode was cut off
            info: Additional info
        """
        # Normalize action to valid probability distribution
        action = np.clip(action, 0, 1)
        action_sum = action.sum()
        if action_sum > 0:
            action = action / action_sum
        else:
            action = np.ones(self.n_models) / self.n_models
        
        # Select top-K models based on action weights
        # Penalize already recommended models
        masked_action = action.copy()
        masked_action[self.recommended_models] = 0
        
        if masked_action.sum() > 0:
            masked_action = masked_action / masked_action.sum()
            selected_indices = np.random.choice(
                self.n_models,
                size=min(self.top_k, self.n_models - len(self.recommended_models)),
                p=masked_action,
                replace=False
            )
        else:
            # All models recommended, sample randomly
            remaining = list(set(range(self.n_models)) - set(self.recommended_models))
            selected_indices = np.random.choice(
                remaining,
                size=min(self.top_k, len(remaining)),
                replace=False
            ) if remaining else []
        
        # Calculate reward
        reward = self._calculate_reward(selected_indices)
        
        # Update state
        self.recommended_models.extend(selected_indices.tolist())
        self.episode_rewards.append(reward)
        self.current_step += 1
        
        # Check termination
        terminated = len(self.recommended_models) >= self.n_models
        truncated = self.current_step >= self.max_steps
        
        observation = self._get_observation()
        info = self._get_info()
        info['selected_models'] = selected_indices.tolist()
        info['reward_components'] = self._get_reward_components(selected_indices)
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct current state observation.
        
        Returns:
            State vector
        """
        # Aggregate features across all models (mean)
        feature_values = self.features_df[self.state_feature_cols].fillna(0.5).mean().values
        
        # Add step progress
        step_progress = self.current_step / self.max_steps
        
        observation = np.concatenate([
            feature_values,
            [step_progress]
        ]).astype(np.float32)
        
        return observation
    
    def _calculate_reward(self, selected_indices: np.ndarray) -> float:
        """
        Calculate composite reward for selected models.
        
        Args:
            selected_indices: Indices of selected models
            
        Returns:
            Composite reward
        """
        if len(selected_indices) == 0:
            return -1.0  # Penalty for no selection
        
        # Extract features for selected models
        selected_features = self.features_df.iloc[selected_indices]
        
        # 1. Performance reward (higher is better)
        # Handle NaN: use median of available scores as fallback
        perf_scores = selected_features['performance_score'].values
        valid_perf = perf_scores[~np.isnan(perf_scores)]
        
        if len(valid_perf) > 0:
            r_performance = float(np.mean(valid_perf))
        else:
            # No performance scores available - use neutral value
            r_performance = 0.5
        
        # 2. Cost penalty (lower cost is better, so we invert)
        cost_scores = selected_features['cost_score'].values
        valid_cost = cost_scores[~np.isnan(cost_scores)]
        
        if len(valid_cost) > 0:
            r_cost = 1.0 - float(np.mean(valid_cost))
        else:
            r_cost = 0.5
        
        # 3. Diversity reward (varied architectures)
        diversity_scores = selected_features['diversity_score'].values
        valid_diversity = diversity_scores[~np.isnan(diversity_scores)]
        
        if len(valid_diversity) > 0:
            r_diversity = float(np.mean(valid_diversity))
        else:
            r_diversity = 0.5
        
        # 4. Popularity reward (aligned with user trends)
        popularity_scores = selected_features['popularity_score'].values
        valid_popularity = popularity_scores[~np.isnan(popularity_scores)]
        
        if len(valid_popularity) > 0:
            r_popularity = float(np.mean(valid_popularity))
        else:
            r_popularity = 0.5
        
        # 5. Recency reward (newer models)
        recency_scores = selected_features['recency_score'].values
        valid_recency = recency_scores[~np.isnan(recency_scores)]
        
        if len(valid_recency) > 0:
            r_recency = float(np.mean(valid_recency))
        else:
            r_recency = 0.5
        
        # Composite reward
        reward = (
            self.reward_weights.performance * r_performance +
            self.reward_weights.cost * r_cost +
            self.reward_weights.diversity * r_diversity +
            self.reward_weights.popularity * r_popularity +
            self.reward_weights.recency * r_recency
        )
        
        return float(reward)
    
    def _get_reward_components(self, selected_indices: np.ndarray) -> Dict[str, float]:
        """Get individual reward components for analysis."""
        if len(selected_indices) == 0:
            return {
                'performance': 0.0,
                'cost': 0.0,
                'diversity': 0.0,
                'popularity': 0.0,
                'recency': 0.0
            }
        
        selected_features = self.features_df.iloc[selected_indices]
        
        # Handle NaN properly - use mean of valid values
        def safe_mean(scores):
            valid = scores[~np.isnan(scores)]
            return float(np.mean(valid)) if len(valid) > 0 else 0.5
        
        return {
            'performance': safe_mean(selected_features['performance_score'].values),
            'cost': 1.0 - safe_mean(selected_features['cost_score'].values),
            'diversity': safe_mean(selected_features['diversity_score'].values),
            'popularity': safe_mean(selected_features['popularity_score'].values),
            'recency': safe_mean(selected_features['recency_score'].values)
        }
    
    def _get_info(self) -> Dict:
        """Get additional info about current state."""
        return {
            'step': self.current_step,
            'models_recommended': len(self.recommended_models),
            'episode_return': sum(self.episode_rewards),
            'mean_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        }
    
    def render(self):
        """Render environment state (for debugging)."""
        print(f"\n{'='*60}")
        print(f"Step: {self.current_step}/{self.max_steps}")
        print(f"Models recommended: {len(self.recommended_models)}/{self.n_models}")
        print(f"Episode return: {sum(self.episode_rewards):.4f}")
        print(f"{'='*60}\n")
    
    def get_model_info(self, indices: List[int]) -> pd.DataFrame:
        """Get detailed info about specific models."""
        return self.features_df.iloc[indices][
            ['model_id', 'cost_score', 'popularity_score', 
             'recency_score', 'diversity_score', 'performance_score']
        ]