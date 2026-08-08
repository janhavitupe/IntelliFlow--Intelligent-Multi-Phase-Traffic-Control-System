"""
state_builder.py

Builds the RL observation vector (and the tabular discretization) by
REUSING exactly what the Density strategy already computes. We do not
invent new features here - the state is the same signal the Phase 3
adaptive controller sees, so the learned policy is directly comparable
to the rule-based Density controller.

Observation vector (23-dim):
    [0:4]    queue length per approach (North/South/East/West), normalized
    [4:8]    percentile rank per approach (1..4 -> (rank-1)/3)
    [8:12]   starvation counters per approach, normalized
    [12:22]  active phase one-hot (10 normal phases)
    [22]     elapsed seconds in the current phase, normalized

Tabular discretization (for Q-learning):
    queue bucket per approach: LOW / MED / HIGH (3 levels, 4 approaches)
    folded with the last-active-phase (10 choices) -> 810 states.
    The queue thresholds reuse the Density strategy's concept of relative
    density, applied with simple LOW/MED/HIGH cutoffs so the tabular agent
    stays conceptually consistent with the rule-based controller.
"""
import numpy as np

from core.enums import PhaseType
from config.phases import all_phase_types
from config import rl as rl_config

APPROACH_ORDER = ("North", "South", "East", "West")


class ObservationBuilder:
    """
    Builds the raw 23-dim observation vector from an intersection + the
    density strategy's bookkeeping.
    """

    def __init__(self, density_strategy=None):
        self.approach_order = APPROACH_ORDER
        self.phase_types = all_phase_types()
        self.phase_index = {pt.name: i for i, pt in enumerate(self.phase_types)}
        self.density = density_strategy  # optional, for starvation counters

    def build(self, intersection, active_phase, elapsed_in_phase):
        """
        Return a 23-dim numpy float32 vector.

        Args:
            intersection (Intersection): the current intersection.
            active_phase (PhaseType|None): the currently active phase.
            elapsed_in_phase (float): seconds already spent in the phase.
        """
        # 1. Queue counts per approach (reuse DensityStrategy counting).
        counts = {
            name: intersection.get_approach(name).total_queue_length()
            for name in self.approach_order
        }

        # 2. Percentile ranking (relative to current state).
        ranks = self._rank(counts)           # approach -> rank (1..4)

        # 3. Starvation counters (from the density strategy, if available).
        starvation = self._starvation(counts)

        # 4. Active-phase one-hot.
        one_hot = self._phase_one_hot(active_phase)

        obs = np.zeros(23, dtype=np.float32)
        for i, name in enumerate(self.approach_order):
            obs[i] = counts[name] / rl_config.QUEUE_NORM            # queue
            obs[4 + i] = (ranks[name] - 1) / 3.0                    # rank
            obs[8 + i] = starvation[name] / rl_config.STARVATION_MAX
        obs[12:22] = one_hot
        obs[22] = elapsed_in_phase / rl_config.MAX_GREEN_NORM
        return obs

    # ---- helpers (mirror DensityStrategy semantics) ----

    def _rank(self, counts: dict) -> dict:
        """Rank approaches 1 (most loaded) .. 4, deterministic tie-break."""
        ordered = sorted(
            self.approach_order,
            key=lambda name: (-counts[name], self.approach_order.index(name)),
        )
        return {name: rank for rank, name in enumerate(ordered, start=1)}

    def _starvation(self, counts: dict) -> dict:
        """Return per-approach starvation counters (0 if no density strategy)."""
        if self.density is None:
            return {name: 0 for name in self.approach_order}
        return dict(self.density._starvation_cycles)

    def _phase_one_hot(self, active_phase):
        vec = np.zeros(len(self.phase_types), dtype=np.float32)
        if active_phase is not None and active_phase.name in self.phase_index:
            vec[self.phase_index[active_phase.name]] = 1.0
        return vec


class Discretizer:
    """
    Collapses the raw state into a small integer bucket for tabular Q.

    Bucket = (queue_bucket(4 approaches) base-3) * 10 + last_phase_index
    -> 3^4 * 10 = 810 states. A quick assert guards the table shape so a
    future refactor cannot silently break it.
    """

    def __init__(self):
        self.approach_order = APPROACH_ORDER
        self.phase_types = all_phase_types()
        self.phase_index = {pt.name: i for i, pt in enumerate(self.phase_types)}
        self.n_buckets = 3 ** len(self.approach_order)  # 81
        self.n_states = self.n_buckets * len(self.phase_types)  # 810
        assert self.n_states == 810, (
            f"Tabular state count mismatch: expected 810, got {self.n_states}"
        )

    def queue_bucket(self, q: int) -> int:
        """0 = LOW, 1 = MED, 2 = HIGH."""
        if q < rl_config.LOW_THRESHOLD:
            return 0
        if q >= rl_config.HIGH_THRESHOLD:
            return 2
        return 1

    def discretize(self, counts: dict, last_phase) -> int:
        """
        Map (per-approach queue counts, last active phase) -> integer state.

        Args:
            counts (dict): approach -> queue length.
            last_phase (PhaseType|None): the phase that was just active.
        """
        bucket = 0
        for name in self.approach_order:
            bucket = bucket * 3 + self.queue_bucket(counts.get(name, 0))

        phase_idx = 0
        if last_phase is not None and last_phase.name in self.phase_index:
            phase_idx = self.phase_index[last_phase.name]

        state = bucket * len(self.phase_types) + phase_idx
        assert 0 <= state < self.n_states, f"State {state} out of range"
        return state

    # The full 23-dim observation is also useful for the discretizer's
    # caller to know the one-hot phase; expose a convenience accessor.
    def phase_of(self, last_phase):
        if last_phase is None or last_phase.name not in self.phase_index:
            return 0
        return self.phase_index[last_phase.name]