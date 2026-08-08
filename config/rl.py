"""
rl.py

Central configuration for the Phase 4 reinforcement-learning wrapper.

Reward stays the spec-default crude proxy: r = -sum(queue_lengths).
Emergency/ambulance handling stays entirely rule-based (the scheduler's
preemption state machine). The RL agent never sees or acts during an
emergency window.
"""
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.1

GREEN_DURATION = 12.0
EPISODE_LENGTH = 200
PROFILES = ("LIGHT_TRAFFIC", "NORMAL_TRAFFIC", "RUSH_HOUR", "NIGHT", "CUSTOM")

# Observation vector (23-dim).
QUEUE_NORM = 50.0
STARVATION_MAX = 4.0
MAX_GREEN_NORM = 40.0
OBS_DIM = 23

# Tabular Q discretization.
LOW_THRESHOLD = 5
HIGH_THRESHOLD = 20
NUM_PHASES = 10

# Tabular Q training.
TABULAR_EPISODES = 300

# ---------------------------------------------------------------------------
# DQN settings (Stage 2)
# ---------------------------------------------------------------------------
DQN_HIDDEN_LAYERS = (64, 64)
DQN_REPLAY_SIZE = 20000
DQN_BATCH_SIZE = 64
DQN_TARGET_UPDATE = 50
DQN_EPISODES = 300
DQN_LEARNING_RATE = 0.001

SEED = 42
NUMPY_SEED = 42
PLOT = False
