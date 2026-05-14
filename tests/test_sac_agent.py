import numpy as np

from src.rl.sac_agent import SACAgent


def test_sac_agent_select_action():
    agent = SACAgent(state_dim=4, action_dim=2)
    action = agent.select_action(np.zeros(4))

    assert action.shape == (2,)
    assert all(0.0 <= float(a) <= 1.0 for a in action)
