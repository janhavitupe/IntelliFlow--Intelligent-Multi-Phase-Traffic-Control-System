"""
agents.py

Reinforcement-learning agents for the traffic RL environment.

Stage 1: TabularQAgent
    Q-learning over a discretized state (810 buckets = 3^4 queue levels x
    10 last-phase). epsilon-greedy action selection. Simple, fast, fully
    interpretable - the recommended sanity-check baseline before DQN.

The agent is usable in two modes:
    - training: Q-learning updates on (s, a, r, s') transitions.
    - inference: select_action returns argmax_a Q[s, a] (no updates), which
      is what RLStrategy uses at demo time.
"""
import random

import numpy as np

from config import rl as rl_config


class TabularQAgent:
    """
    Tabular Q-learning agent.

    Args:
        n_states (int): number of discrete states (810 by default).
        n_actions (int): number of actions (10 phases).
        gamma (float): discount factor.
        alpha (float): learning rate.
        epsilon (float): initial exploration probability.
        epsilon_end (float): minimum exploration probability.
        epsilon_decay (float): per-episode decay factor.
        seed (int|None): optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        n_states=810,
        n_actions=10,
        gamma=None,
        alpha=None,
        epsilon=None,
        epsilon_end=None,
        epsilon_decay=None,
        seed=None,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma if gamma is not None else rl_config.GAMMA
        self.alpha = alpha if alpha is not None else rl_config.LEARNING_RATE
        self.epsilon = epsilon if epsilon is not None else rl_config.EPSILON_START
        self.epsilon_end = (
            epsilon_end if epsilon_end is not None else rl_config.EPSILON_END
        )
        self.epsilon_decay = (
            epsilon_decay
            if epsilon_decay is not None
            else rl_config.EPSILON_DECAY
        )

        # Q-table: [state][action].
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)

        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def epsilon_greedy(self, state: int) -> int:
        """Pick an action using epsilon-greedy exploration."""
        if self._rng.random() < self.epsilon:
            return self._rng.randrange(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def select_action(self, state: int) -> int:
        """Greedy action (inference/demo time)."""
        return int(np.argmax(self.Q[state]))

    # ------------------------------------------------------------------
    # Q-learning update
    # ------------------------------------------------------------------

    def update(self, state, action, reward, next_state, done):
        """Apply one Q-learning update for a transition."""
        best_next = 0.0
        if not done:
            best_next = float(np.max(self.Q[next_state]))
        target = reward + self.gamma * best_next
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

    # ------------------------------------------------------------------
    # Exploration schedule
    # ------------------------------------------------------------------

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def __repr__(self):
        return (
            f"TabularQAgent(states={self.n_states}, "
            f"actions={self.n_actions}, eps={self.epsilon:.3f})"
        )