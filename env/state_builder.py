"""
state_builder.py

Builds the RL observation vector (and the tabular discretization) from a
perception IntersectionObservation: the per-lane summary a camera pipeline
can provide (vehicle counts and how long each vehicle has waited).

Observation vector (75-dim):
    For each of the 16 lanes, in perception.LANE_ORDER (4 values each):
        count / LANE_QUEUE_NORM
        mean wait / WAIT_NORM            (clipped at 3)
        longest wait / WAIT_NORM         (clipped at 3)
        long waiters (> LONG_WAIT_THRESHOLD s) / LANE_QUEUE_NORM
    [64:74]  active phase one-hot (10 normal phases)
    [74]     elapsed green of the active phase / MAX_GREEN_NORM

The long-waiter count makes "1 vehicle waiting 150 s" and "10 vehicles
waiting 150 s" different states, and lane resolution matches what a phase
actually serves (movements, not whole approaches).

All features are computed at the moment the scheduler asks for a decision,
by RLStrategy.observe(), in both training and inference - so the agent sees
exactly the same feature definitions in both modes.

Tabular discretization (for Q-learning):
    queue bucket per approach: LOW / MED / HIGH (3 levels, 4 approaches)
    x active phase (10) x elapsed-green bucket (3: just started / mid /
    at the MAX_GREEN cap) -> 2430 states.
"""
import numpy as np

from config.phases import all_phase_types
from config import perception as perception_config
from config import rl as rl_config
from perception.observation import APPROACH_ORDER, LANE_ORDER

FEATURES_PER_LANE = 4


class ObservationBuilder:
    """Builds the raw observation vector from a perception observation."""

    def __init__(self):
        self.phase_types = all_phase_types()
        self.phase_index = {pt.name: i for i, pt in enumerate(self.phase_types)}
        self.size = len(LANE_ORDER) * FEATURES_PER_LANE + len(self.phase_types) + 1
        assert self.size == rl_config.OBS_DIM

    def build(self, observation, active_phase, elapsed_in_phase):
        """
        Return the observation vector (float32, length OBS_DIM).

        Args:
            observation (IntersectionObservation): per-lane perception.
            active_phase (PhaseType|None): the currently active phase.
            elapsed_in_phase (float): seconds already spent in the phase.
        """
        obs = np.zeros(self.size, dtype=np.float32)
        long_wait = perception_config.LONG_WAIT_THRESHOLD
        for i, lane in enumerate(observation.lanes):
            base = i * FEATURES_PER_LANE
            obs[base] = lane.count / rl_config.LANE_QUEUE_NORM
            obs[base + 1] = min(lane.mean_wait / rl_config.WAIT_NORM, 3.0)
            obs[base + 2] = min(lane.max_wait / rl_config.WAIT_NORM, 3.0)
            obs[base + 3] = lane.long_waiters(long_wait) / rl_config.LANE_QUEUE_NORM
        n_lane = len(LANE_ORDER) * FEATURES_PER_LANE
        obs[n_lane:n_lane + len(self.phase_types)] = self._phase_one_hot(active_phase)
        obs[-1] = elapsed_in_phase / rl_config.MAX_GREEN_NORM
        return obs

    def _phase_one_hot(self, active_phase):
        vec = np.zeros(len(self.phase_types), dtype=np.float32)
        if active_phase is not None and active_phase.name in self.phase_index:
            vec[self.phase_index[active_phase.name]] = 1.0
        return vec


class Discretizer:
    """
    Collapses the state into a small integer bucket for tabular Q.

    State = ((queue_bucket(4 approaches) base-3) * 10 + phase_index) * 3
            + elapsed_bucket
    -> 3^4 * 10 * 3 = 2430 states. The elapsed bucket lets the table tell
    whether the active green can still be extended.
    """

    N_STATES = 3 ** len(APPROACH_ORDER) * 10 * (len(rl_config.ELAPSED_BUCKET_EDGES) + 1)

    def __init__(self):
        self.approach_order = APPROACH_ORDER
        self.phase_types = all_phase_types()
        self.phase_index = {pt.name: i for i, pt in enumerate(self.phase_types)}
        self.n_elapsed = len(rl_config.ELAPSED_BUCKET_EDGES) + 1     # 3
        self.n_states = (
            3 ** len(self.approach_order) * len(self.phase_types) * self.n_elapsed
        )
        assert self.n_states == self.N_STATES

    def queue_bucket(self, q: int) -> int:
        """0 = LOW, 1 = MED, 2 = HIGH."""
        if q < rl_config.LOW_THRESHOLD:
            return 0
        if q >= rl_config.HIGH_THRESHOLD:
            return 2
        return 1

    def elapsed_bucket(self, elapsed: float) -> int:
        """0 = just started, 1 = mid, 2 = at the cap (> upper edge)."""
        low, high = rl_config.ELAPSED_BUCKET_EDGES
        if elapsed < low:
            return 0
        return 1 if elapsed <= high else 2

    def discretize(self, counts: dict, last_phase, elapsed: float = 0.0) -> int:
        """
        Map (queue counts per approach, active phase, elapsed green) -> state.

        Args:
            counts (dict): approach -> queue length.
            last_phase (PhaseType|None): the currently active phase.
            elapsed (float): seconds since that phase's green started.
        """
        bucket = 0
        for name in self.approach_order:
            bucket = bucket * 3 + self.queue_bucket(counts.get(name, 0))

        phase_idx = 0
        if last_phase is not None and last_phase.name in self.phase_index:
            phase_idx = self.phase_index[last_phase.name]

        state = (bucket * len(self.phase_types) + phase_idx) * self.n_elapsed \
            + self.elapsed_bucket(elapsed)
        assert 0 <= state < self.n_states, f"State {state} out of range"
        return state
