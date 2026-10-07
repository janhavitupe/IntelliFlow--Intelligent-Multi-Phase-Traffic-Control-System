"""Perception sources, the wait-aware reward and the maximum-red safety rule."""
import pytest

from config import rl as rl_config
from core.enums import MovementType, PhaseType
from env.traffic_env import TrafficRLEnv
from perception import LANE_ORDER, GroundTruthPerception, NoisyPerception
from strategies.rl_strategy import RLStrategy
from helpers import add, car


def _fill(intersection, n=30, approach="North"):
    for i in range(n):
        add(intersection, approach, car(waiting=float(n - i)))   # head waits longest


def test_ground_truth_reports_every_vehicle_head_first(intersection):
    _fill(intersection, 5)
    add(intersection, "West", car(waiting=7.0), MovementType.UTURN)
    obs = GroundTruthPerception().observe(intersection)
    assert [l.movement_id for l in obs.lanes] == list(LANE_ORDER)
    assert obs.lane("North_STRAIGHT").waits == (5.0, 4.0, 3.0, 2.0, 1.0)
    assert obs.lane("West_UTURN").waits == (7.0,)
    assert obs.approach_count("North") == 5 and obs.approach_count("East") == 0


def test_noisy_perception_respects_view_limit_and_is_reproducible(intersection):
    _fill(intersection, 40)
    a = NoisyPerception(seed=3, view_limit=15)
    b = NoisyPerception(seed=3, view_limit=15)
    oa, ob = a.observe(intersection), b.observe(intersection)
    assert oa == ob                                            # same seed, same errors
    assert oa.lane("North_STRAIGHT").count <= 15 + 1           # view limit (+1 phantom)
    a.reset()
    assert a.observe(intersection) == oa                       # reset restarts the RNG


def test_noisy_perception_without_errors_equals_ground_truth(intersection):
    _fill(intersection, 10)
    clean = NoisyPerception(seed=1, miss_rate=0, false_positive_rate=0, view_limit=10**6,
                            wait_error=0, track_loss_rate=0)
    assert clean.observe(intersection) == GroundTruthPerception().observe(intersection)


def test_noisy_perception_misses_some_vehicles(intersection):
    _fill(intersection, 15)
    noisy = NoisyPerception(seed=0, miss_rate=0.5, false_positive_rate=0, wait_error=0,
                            track_loss_rate=0)
    counts = [noisy.observe(intersection).lane("North_STRAIGHT").count for _ in range(50)]
    assert 4 < sum(counts) / len(counts) < 11                  # ~7.5 of 15 seen


def test_wait_penalty_weights_vehicles_by_how_long_they_waited():
    env = TrafficRLEnv(profile_key="NORMAL_TRAFFIC", seed=1, wait_penalty_slope=30.0)
    env.reset()
    for lane in env.intersection.all_lanes():                  # start from a clean slate
        while not lane.is_empty:
            lane.remove_front_vehicle()
    for wait in (120.0, 120.0, 30.0):                          # two long waiters, one short
        add(env.intersection, "North", car(waiting=wait))
    # costs per second: 1 + (120-60)/30 = 3 (x2), and 1 for the 30 s vehicle
    expected = -(3 + 3 + 1) * env.tick_interval / rl_config.REWARD_SCALE
    assert env._tick_reward() == pytest.approx(expected)
    env.wait_penalty_slope = None                              # penalty off: 1 per vehicle
    assert env._tick_reward() == pytest.approx(-3 * env.tick_interval / rl_config.REWARD_SCALE)


class _AlwaysPhase1:
    Q = None

    def select_action(self, _state):
        return 0


def test_max_red_rule_overrides_the_agent_for_starved_lanes(intersection):
    from config.phases import build_phase_plan
    strategy = RLStrategy(agent=_AlwaysPhase1(), max_red=90.0, serve_waiting=False)
    phase1 = build_phase_plan(intersection)[PhaseType.PHASE_1]
    add(intersection, "East", car(waiting=100.0))              # East_STRAIGHT: not in PHASE_1

    assert strategy._get_action(intersection, phase1, time=50.0) == 0    # red 50 s: no override
    choice = strategy._get_action(intersection, phase1, time=120.0)      # red 120 s: override
    served = {m.movement_id for m in build_phase_plan(intersection)[strategy._phases[choice]].movements}
    assert "East_STRAIGHT" in served
    assert strategy.shield_overrides == 1


def test_no_empty_green_serves_the_most_accumulated_waiting(intersection):
    from config.phases import build_phase_plan
    strategy = RLStrategy(agent=_AlwaysPhase1(), max_red=None, serve_waiting=True)
    phase1 = build_phase_plan(intersection)[PhaseType.PHASE_1]
    add(intersection, "East", car(waiting=20.0))               # East_STRAIGHT: not in PHASE_1
    choice = strategy._get_action(intersection, phase1, time=30.0)
    served = {m.movement_id for m in build_phase_plan(intersection)[strategy._phases[choice]].movements}
    assert "East_STRAIGHT" in served and strategy.empty_green_overrides == 1
    add(intersection, "West", car(waiting=5.0))                # now PHASE_1 serves someone
    assert strategy._get_action(intersection, phase1, time=31.0) == 0


def test_sensor_outage_falls_back_to_fixed_rotation(intersection):
    from config.phases import build_phase_plan
    from perception import BlackoutPerception
    strategy = RLStrategy(agent=_AlwaysPhase1(), perception=BlackoutPerception(start=0.0))
    phase3 = build_phase_plan(intersection)[PhaseType.PHASE_3]
    assert strategy._get_action(intersection, phase3, time=10.0) == 3      # PHASE_4 next
    assert strategy.fallback_decisions == 1


def test_training_env_runs_without_the_safety_envelope():
    env = TrafficRLEnv(profile_key="NORMAL_TRAFFIC", seed=1)
    env.reset()
    assert env.strategy.max_red is None and env.strategy.serve_waiting is False
