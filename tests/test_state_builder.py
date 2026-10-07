"""RL observation vector and tabular discretization."""
import numpy as np
import pytest

from config import perception as perception_config
from config import rl as rl_config
from config.phases import all_phase_types
from core.enums import MovementType, PhaseType
from env.state_builder import FEATURES_PER_LANE, Discretizer, ObservationBuilder
from perception import LANE_ORDER, GroundTruthPerception
from helpers import add, car


def _lane_features(obs, movement_id):
    base = LANE_ORDER.index(movement_id) * FEATURES_PER_LANE
    return obs[base:base + FEATURES_PER_LANE]


def test_observation_layout(intersection):
    add(intersection, "North", car(waiting=90.0))
    add(intersection, "North", car(waiting=70.0))
    add(intersection, "North", car(waiting=20.0))
    add(intersection, "East", car(waiting=30.0), MovementType.LEFT)
    observation = GroundTruthPerception().observe(intersection)
    obs = ObservationBuilder().build(observation, PhaseType.PHASE_3, 20.0)

    assert obs.shape == (rl_config.OBS_DIM,) == (75,) and obs.dtype == np.float32
    n, w = rl_config.LANE_QUEUE_NORM, rl_config.WAIT_NORM
    # count, mean wait, longest wait, long waiters (> 60 s)
    assert _lane_features(obs, "North_STRAIGHT") == pytest.approx([3 / n, 60 / w, 90 / w, 2 / n])
    assert _lane_features(obs, "East_LEFT") == pytest.approx([1 / n, 30 / w, 30 / w, 0])
    assert _lane_features(obs, "West_UTURN") == pytest.approx([0, 0, 0, 0])
    one_hot = obs[64:74]
    assert one_hot.tolist() == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    assert obs[74] == pytest.approx(20.0 / rl_config.MAX_GREEN_NORM)


def test_long_waiters_distinguish_one_from_many(intersection):
    """1 vehicle waiting 150 s and 10 vehicles waiting 150 s must be different states."""
    def obs_with(n):
        from core.intersection import Intersection
        inter = Intersection()
        for _ in range(n):
            add(inter, "South", car(waiting=150.0))
        return ObservationBuilder().build(GroundTruthPerception().observe(inter), None, 0.0)
    one, ten = obs_with(1), obs_with(10)
    assert _lane_features(one, "South_STRAIGHT")[3] * rl_config.LANE_QUEUE_NORM == 1
    assert _lane_features(ten, "South_STRAIGHT")[3] * rl_config.LANE_QUEUE_NORM == 10
    assert perception_config.LONG_WAIT_THRESHOLD < 150


def test_wait_features_are_clipped(intersection):
    add(intersection, "West", car(waiting=10_000.0))
    obs = ObservationBuilder().build(GroundTruthPerception().observe(intersection), None, 0.0)
    assert _lane_features(obs, "West_STRAIGHT")[1:3] == pytest.approx([3.0, 3.0])
    assert obs[64:74].sum() == 0                    # no active phase


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
