"""
Golden-value regression tests and end-to-end smoke tests.

The simulator is deterministic for a given seed, so the rule-based
controllers' KPIs are pinned exactly. If one of these fails, a change has
altered simulation behaviour: confirm that was intended, then update the
numbers (and expect every reported result to move).
"""
import math

import pytest

from config import rl as rl_config
from rl.train import train_dqn, train_tabular
from simulation import Simulation

GOLDEN = {
    # (strategy, profile): (avg_wait, avg_queue, served, spawned) after 200 ticks, seed 1
    ("fixed_timer", "NORMAL_TRAFFIC"): (484.0525, 25.965, 146, 163),
    ("fixed_timer", "RUSH_HOUR"): (999.1325, 49.72, 230, 323),
    ("density", "NORMAL_TRAFFIC"): (254.2125, 18.745, 147, 163),
    ("density", "RUSH_HOUR"): (905.765, 50.17, 236, 323),
}


@pytest.mark.parametrize("strategy, profile", list(GOLDEN))
def test_rule_based_controllers_are_unchanged(strategy, profile):
    sim = Simulation(profile_key=profile, seed=1, max_ticks=200, live=False,
                     strategy_key=strategy)
    for _ in range(200):
        sim.step()
    a = sim.analytics
    wait, queue, served, spawned = GOLDEN[(strategy, profile)]
    assert a.average_waiting_time == pytest.approx(wait, rel=1e-9)
    assert a.average_queue_length == pytest.approx(queue, rel=1e-9)
    assert a.total_vehicles_served == served
    assert a.total_vehicles_spawned == spawned


class _SpyAgent:
    """Records every learning update the training loop makes."""
    Q = None

    def __init__(self):
        self.epsilon = 1.0
        self.updates = []

    def epsilon_greedy(self, state):
        return len(self.updates) % rl_config.NUM_PHASES

    def select_action(self, state):
        return 0

    def update(self, state, action, reward, next_state, terminal, discount):
        self.updates.append((terminal, discount))

    def get_weights(self):
        return None

    def set_weights(self, weights):
        pass


def test_training_loop_uses_truncation_and_per_second_discount():
    """Regression (Steps 2-3): time limits are not terminal; discount = gamma ** seconds."""
    from rl.train import _train
    agent = _SpyAgent()
    _train(agent, use_discrete=True, n_episodes=1, profiles=("NORMAL_TRAFFIC",),
           episode_length=300, seed=1, verbose=False)
    terminals = [t for t, _ in agent.updates]
    discounts = {round(d, 12) for _, d in agent.updates}
    assert terminals and not any(terminals)
    g = rl_config.GAMMA_PER_SECOND
    # Every discount is gamma ** (the step's real duration in half-second ticks), and the
    # durations differ between steps (extensions, switches, gap-outs).
    ticks = [math.log(d) / math.log(g) / 0.5 for d in discounts]
    assert all(abs(t - round(t)) < 1e-6 and round(t) >= 1 for t in ticks)
    assert len(discounts) > 1


@pytest.mark.parametrize("train", [train_tabular, train_dqn])
def test_training_pipeline_runs_end_to_end(train):
    agent, history = train(n_episodes=2, episode_length=200, verbose=False)
    assert len(history["episode_rewards"]) == 2
    assert [ep for ep, _ in history["validation"]] == [0, 2]
    assert history["best_episode"] in (0, 2)
    assert all(r < 0 for r in history["episode_rewards"])
