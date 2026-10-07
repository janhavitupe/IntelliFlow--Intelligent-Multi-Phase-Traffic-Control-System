"""
rl package

Reinforcement-learning agents and training loops for Phase 4.

    - TabularQAgent : Q-learning over a discretized state (2430 buckets).
    - DQNAgent      : Deep Q-Network (pure-numpy MLP + Adam, replay buffer,
                      target network, Double-DQN, Huber loss) over the raw
                      observation vector.
    - train_tabular / train_dqn : one shared training loop (rotating
                      profiles, held-out validation, best-checkpoint
                      selection); each returns (agent, history).
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
