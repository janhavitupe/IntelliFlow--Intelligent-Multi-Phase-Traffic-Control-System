"""
rl package

Reinforcement-learning agents and training loops for Phase 4.

Stage 1 (implemented):
    - TabularQAgent : Q-learning over a discretized state (810 buckets).
    - train_tabular : training loop that varies traffic profiles across
                      episodes and returns the per-episode reward curve.

Stage 2 (implemented):
    - DQNAgent      : Deep Q-Network (pure-numpy MLP, replay buffer, target
                      network) over the raw 23-dim observation.
    - train_dqn     : DQN training loop sharing the episode/profile/seed
                      structure of train_tabular for a fair comparison.
"""
from .agents import TabularQAgent
from .dqn import DQNAgent, MLP, ReplayBuffer
from .train import train_tabular, train_dqn

__all__ = [
    "TabularQAgent",
    "DQNAgent",
    "MLP",
    "ReplayBuffer",
    "train_tabular",
    "train_dqn",
]
