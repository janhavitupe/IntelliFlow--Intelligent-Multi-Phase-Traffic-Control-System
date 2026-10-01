"""
rl.py

Central configuration for the Phase 4 reinforcement-learning wrapper.

Reward: r = -(queueing delay during the step) / REWARD_SCALE, where queueing
delay = sum over the step's ticks of (total queue length * tick seconds),
i.e. vehicle-seconds spent waiting. Summed over the whole decision interval
so the objective is total delay even when a step is stretched by an
emergency preemption.

Emergency/ambulance handling stays entirely rule-based (the scheduler's
preemption state machine). The RL agent never sees or acts during an
emergency window.
"""
# Decisions are no longer evenly spaced (a 5 s extension vs a 12 s
# yellow + minimum green), so discounting is per simulated SECOND:
# a step lasting d seconds is discounted by GAMMA_PER_SECOND ** d.
# 0.9925 ** 14 = 0.90, i.e. the old per-decision gamma for a 14 s phase.
GAMMA_PER_SECOND = 0.9925
GAMMA = 0.9                  # per-step fallback when no duration is given
EPSILON_START = 1.0
EPSILON_END = 0.05
# Epsilon decays linearly from START to END over this fraction of the
# training episodes, then stays at END (exploit what was learned).
EPSILON_DECAY_FRACTION = 0.7
LEARNING_RATE = 0.1

# Green-time control: extend-or-switch, with the SAME bounds the Density
# controller uses (config/density.py), so the comparison is fair.
#   - A newly selected phase gets MIN_GREEN seconds.
#   - When its green runs out the agent decides again: choosing the
#     current phase extends it by GREEN_EXTENSION; choosing another phase
#     ends it (yellow) and starts that one.
#   - A phase cannot exceed MAX_GREEN; "extend" at the cap advances to the
#     next phase in order instead (the agent sees elapsed time, so this is
#     a deterministic, learnable consequence).
MIN_GREEN = 10.0
GREEN_EXTENSION = 5.0
MAX_GREEN = 40.0
EPISODE_LENGTH = 1200        # ticks (= 10 simulated minutes, ~40 decisions)
PROFILES = ("LIGHT_TRAFFIC", "NORMAL_TRAFFIC", "RUSH_HOUR", "NIGHT", "CUSTOM")

# Reward scale (vehicle-seconds). Moderate traffic gives |r| of order 0.1-1
# per decision; oversaturated rush hour gives |r| of order 10.
REWARD_SCALE = 1000.0

# Observation vector (23-dim).
QUEUE_NORM = 50.0
WAIT_NORM = 60.0          # seconds; longest-wait (starvation) feature scale
MAX_GREEN_NORM = 40.0      # elapsed-green feature scale (= MAX_GREEN)
OBS_DIM = 23

# Tabular Q discretization.
LOW_THRESHOLD = 5
HIGH_THRESHOLD = 20
NUM_PHASES = 10
# Elapsed-green buckets: [0, 15) just started, [15, 35] mid, > 35 at the
# cap (no further extension possible).
ELAPSED_BUCKET_EDGES = (15.0, 35.0)

# Tabular Q training.
TABULAR_EPISODES = 500

# ---------------------------------------------------------------------------
# DQN settings (Stage 2)
# ---------------------------------------------------------------------------
DQN_HIDDEN_LAYERS = (64, 64)
DQN_REPLAY_SIZE = 20000
DQN_BATCH_SIZE = 64
DQN_TARGET_UPDATE = 200      # gradient steps between target-network syncs
DQN_EPISODES = 500
DQN_LEARNING_RATE = 0.001    # Adam
DQN_HUBER_DELTA = 1.0        # TD errors beyond this get a linear (not squared) loss
DQN_GRAD_CLIP = 10.0         # max global gradient norm per update

# ---------------------------------------------------------------------------
# Validation during training (greedy policy on held-out seeds)
# ---------------------------------------------------------------------------
# Mirrors the final evaluation protocol (run_experiments: 5 profiles x 3
# seeds x 200 ticks) so "best on validation" means "best on the metric we
# report" - but on seeds disjoint from training (42+) and evaluation (1-3).
VALIDATION_EVERY = 25        # episodes between validation runs
VALIDATION_SEEDS = (1000, 1001, 1002)
VALIDATION_TICKS = 200

SEED = 42
NUMPY_SEED = 42
PLOT = False
