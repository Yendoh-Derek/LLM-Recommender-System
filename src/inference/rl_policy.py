"""
RL Policy implementation for the SAC (Soft Actor-Critic) agent.
Real implementation using trained model from Colab.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path

from src.core.logger import logger


class Actor(nn.Module):
    """Actor network for SAC agent."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize Actor network.
        
        Args:
            state_dim: Dimension of state vector
            action_dim: Dimension of action vector (number of models)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        """Forward pass through actor network."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mean, log_std
    
    def sample(self, state):
        """Sample action from policy."""
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
    """
    SAC (Soft Actor-Critic) agent for model recommendations.
    """
    
    def __init__(self, state_dim: int, action_dim: int, device: str = 'cpu'):
        """
        Initialize SAC agent.
        
        Args:
            state_dim: Dimension of state vector
            action_dim: Dimension of action vector (number of models)
            device: Device to run on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.is_loaded = False
    
    def load(self, path: str):
        """
        Load trained SAC agent from checkpoint.
        
        Args:
            path: Path to model checkpoint file
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.actor.eval()
            self.is_loaded = True
            logger.info(f"Loaded SAC agent from {path}")
        except Exception as e:
            logger.error(f"Error loading SAC agent: {e}")
            raise
    
    def select_action(self, state: np.ndarray, evaluate: bool = True) -> np.ndarray:
        """
        Select action (model scores) given a state.
        
        Args:
            state: State vector (preprocessed features)
            evaluate: If True, use deterministic action (mean), else sample
            
        Returns:
            Action vector (scores for each model)
        """
        if not self.is_loaded:
            raise RuntimeError("SAC agent not loaded. Call load() first.")
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if evaluate:
                mean, _ = self.actor(state_tensor)
                action = torch.sigmoid(mean)
            else:
                action, _ = self.actor.sample(state_tensor)
        
        return action.cpu().numpy()[0]


class RLPolicy:
    """
    RL Policy wrapper for SAC agent.
    Provides interface compatible with existing code.
    """
    
    def __init__(self, sac_agent: Optional[SACAgent] = None):
        """
        Initialize RL Policy.
        
        Args:
            sac_agent: SACAgent instance (optional, can be loaded later)
        """
        self.sac_agent = sac_agent
        self.is_loaded = sac_agent is not None and sac_agent.is_loaded
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select an action (model index) given a state.
        
        Args:
            state: State vector (preprocessed features)
            
        Returns:
            Selected action (model index with highest score)
        """
        if not self.is_loaded:
            raise RuntimeError("RL Policy not loaded. Load SAC agent first.")
        
        action_scores = self.sac_agent.select_action(state, evaluate=True)
        return int(np.argmax(action_scores))
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Predict action scores for a given state.
        
        Args:
            state: State vector (preprocessed features)
            
        Returns:
            Array of action scores (one per model)
        """
        if not self.is_loaded:
            raise RuntimeError("RL Policy not loaded. Load SAC agent first.")
        
        return self.sac_agent.select_action(state, evaluate=True)
    
    def get_recommendation_scores(
        self,
        states: List[np.ndarray],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Get recommendation scores for multiple states.
        
        Args:
            states: List of state vectors (preprocessed features for each model)
            top_k: Optional limit on number of results
            
        Returns:
            List of (model_index, score) tuples, sorted by score descending
        """
        if not self.is_loaded:
            raise RuntimeError("RL Policy not loaded. Load SAC agent first.")
        
        # For single state, get scores for all models
        if len(states) == 1:
            scores = self.predict(states[0])
            results = [(idx, float(score)) for idx, score in enumerate(scores)]
        else:
            # Multiple states - get score for each
            results = []
            for idx, state in enumerate(states):
                score = self.predict(state)[0]  # Get score for this model
                results.append((idx, float(score)))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            results = results[:top_k]
        
        return results
