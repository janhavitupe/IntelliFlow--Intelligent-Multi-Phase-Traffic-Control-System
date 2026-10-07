"""Edge-case scenarios, the invariant monitor, and the deployed safety envelope."""
import os

import pytest

from core.enums import SignalState
from evaluation.stress_test import (SCENARIO_BY_NAME, InvariantMonitor, controller_factory,
                                    run_scenario)
from simulation import Simulation
from strategies.fixed_timer_strategy import FixedTimerStrategy

MODELS_PRESENT = os.path.exists(os.path.join("models", "dqn_seed42.npz"))


def test_invariant_monitor_flags_a_wrong_green():
    """The monitor must catch a violation, or its 'zero violations' means nothing."""
    sim = Simulation(profile_key="NORMAL_TRAFFIC", seed=1, max_ticks=50, live=False,
                     strategy=FixedTimerStrategy())
    monitor = InvariantMonitor(sim)
    for _ in range(5):
        sim.step()
        monitor.check()
    assert monitor.violations == []
    outsider = next(m for m in sim.intersection.all_movements()
                    if m not in sim.scheduler.current_phase.movements)
    outsider.signal.state = SignalState.GREEN            # plant a conflicting green
    monitor.check()
    assert any(v.startswith("I1") for v in monitor.violations)


@pytest.mark.parametrize("scenario", ["lone_cars", "ambulance_behind_queue", "burst_drain"])
@pytest.mark.parametrize("controller", ["Fixed Timer", "Density"])
def test_rule_based_controllers_hold_all_invariants(scenario, controller):
    result = run_scenario(SCENARIO_BY_NAME[scenario], controller_factory(controller))
    assert result["violations"] == []


@pytest.mark.skipif(not MODELS_PRESENT, reason="trained models not present (run run_experiments.py)")
def test_deployed_dqn_serves_lone_cars_quickly():
    """Regression: the raw policy left a lone car waiting 332 s at an empty junction."""
    sc = SCENARIO_BY_NAME["lone_cars"]
    for seed in (42, 7, 123):
        result = run_scenario(sc, controller_factory("DQN", seed, sc))
        assert result["violations"] == []
        assert result["max_delay"] < 60


@pytest.mark.skipif(not MODELS_PRESENT, reason="trained models not present (run run_experiments.py)")
def test_deployed_dqn_falls_back_when_the_camera_fails():
    sc = SCENARIO_BY_NAME["camera_blackout"]
    strategy_holder = []
    factory = controller_factory("DQN", 42, sc)

    def make():
        strategy_holder.append(factory())
        return strategy_holder[-1]
    result = run_scenario(sc, make)
    assert result["violations"] == []
    assert strategy_holder[0].fallback_decisions > 0
    assert result["unserved"] < 30
