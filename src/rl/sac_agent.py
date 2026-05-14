"""
sac_agent.py
Wrapper for the root SAC implementation.
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, Optional


class SACAgent:
    """Soft Actor-Critic agent wrapper around the root implementation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_entropy: bool = True,
        hidden_dim: int = 256,
        buffer_size: int = 100000,
        **kwargs: Any,
    ) -> None:
        root_sac = importlib.import_module("sac_agent")
        self.agent = root_sac.SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            lr=lr,
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            auto_entropy=auto_entropy,
            hidden_dim=hidden_dim,
            buffer_size=buffer_size,
        )

    def select_action(self, state: Any, evaluate: bool = False) -> Any:
        return self.agent.select_action(state, evaluate=evaluate)

    def update(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.agent.update(*args, **kwargs)

    def save(self, filepath: str) -> None:
        self.agent.save(filepath)

    def load(self, filepath: str) -> None:
        self.agent.load(filepath)
