"""RL observation vector and tabular discretization."""
import numpy as np
import pytest

from config import rl as rl_config
from config.phases import all_phase_types
from core.enums import MovementType, PhaseType
from env.state_builder import Discretizer, ObservationBuilder
from helpers import add, car


def test_observation_layout(intersection):
    for _ in range(5):
        add(intersection, "North", car(waiting=10.0))
    add(intersection, "East", car(waiting=30.0), MovementType.LEFT)
    obs = ObservationBuilder().build(intersection, PhaseType.PHASE_3, 20.0)

    assert obs.shape == (23,) and obs.dtype == np.float32
    # [0:4] queue per approach (N, S, E, W) / QUEUE_NORM
    assert obs[0:4] == pytest.approx(np.array([5, 0, 1, 0]) / rl_config.QUEUE_NORM)
    # [4:8] rank: North busiest (rank 1 -> 0.0), East 2nd, ties by N/S/E/W order
    assert obs[4:8] == pytest.approx([0.0, 2 / 3, 1 / 3, 1.0])
    # [8:12] longest current wait / WAIT_NORM
    assert obs[8:12] == pytest.approx(np.array([10, 0, 30, 0]) / rl_config.WAIT_NORM)
    # [12:22] active-phase one-hot
    assert obs[12:22].tolist() == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    # [22] elapsed green / MAX_GREEN_NORM
    assert obs[22] == pytest.approx(20.0 / rl_config.MAX_GREEN_NORM)


def test_longest_wait_feature_is_clipped(intersection):
    add(intersection, "West", car(waiting=10_000.0))
    obs = ObservationBuilder().build(intersection, None, 0.0)
    assert obs[11] == pytest.approx(3.0)
    assert obs[12:22].sum() == 0                    # no active phase


@pytest.mark.parametrize("elapsed, bucket", [
    (0.0, 0), (14.9, 0), (15.0, 1), (35.0, 1), (35.5, 2), (40.0, 2),
])
def test_elapsed_bucket_edges(elapsed, bucket):
    assert Discretizer().elapsed_bucket(elapsed) == bucket


def test_discretizer_is_a_bijection_onto_its_range():
    d = Discretizer()
    levels = {0: 0, 1: 10, 2: 30}                   # LOW / MED / HIGH queue counts
    seen = set()
    for n in levels.values():
        for s in levels.values():
            for e in levels.values():
                for w in levels.values():
                    counts = {"North": n, "South": s, "East": e, "West": w}
                    for phase in all_phase_types():
                        for elapsed in (0.0, 20.0, 40.0):
                            seen.add(d.discretize(counts, phase, elapsed))
    assert len(seen) == d.n_states == Discretizer.N_STATES == 2430
    assert min(seen) == 0 and max(seen) == d.n_states - 1
