"""
The RL environment contract, including regression tests for the Step 1
credit-assignment bug and train/inference feature skew.
"""
import numpy as np
import pytest

from config import rl as rl_config
from env.traffic_env import TrafficRLEnv
from simulation import Simulation
from strategies.rl_strategy import RLStrategy


def _env(**kw):
    kw.setdefault("profile_key", "NORMAL_TRAFFIC")
    kw.setdefault("seed", 1)
    return TrafficRLEnv(**kw)


def _record_ticks(env):
    """Wrap _finish_tick to log (phase, in_yellow, queue) for every tick."""
    log = []
    original = env._finish_tick

    def wrapped():
        reward = original()
        pt = env.scheduler.active_phase_type
        log.append((pt.name if pt else None, env.scheduler.in_yellow,
                    env.intersection.total_queue_length()))
        return reward

    env._finish_tick = wrapped
    return log


def test_reset_stops_at_first_decision_point():
    env = _env()
    obs = env.reset()
    assert obs.shape == (23,)
    assert env.strategy.awaiting_action
    assert env.tick == 0


def test_reward_is_credited_to_the_action_that_caused_it():
    """Regression (Step 1): reward used to cover the PREVIOUS action's phase."""
    env = _env()
    env.reset()
    log = _record_ticks(env)
    for action in [0, 0, 5, 9, 9, 2]:
        log.clear()
        env.step(action)
        chosen = f"PHASE_{action + 1}"
        greens = {name for name, yellow, _ in log if not yellow}
        assert greens == {chosen}, f"green ticks of step({action}) were {greens}"


def test_switch_costs_yellow_plus_min_green_and_extend_costs_extension():
    env = _env()
    env.reset()
    _, _, _, info = env.step(0)
    assert info["duration"] == rl_config.MIN_GREEN
    _, _, _, info = env.step(0)                       # extend
    assert info["duration"] == rl_config.GREEN_EXTENSION
    log = _record_ticks(env)
    _, _, _, info = env.step(4)                       # switch
    assert info["duration"] == 2.0 + rl_config.MIN_GREEN
    assert sum(1 for _, yellow, _ in log if yellow) == 4


def test_extend_at_max_green_advances_to_next_phase():
    env = _env()
    env.reset()
    env.step(0)                                       # 10 s
    for _ in range(6):                                # +30 s -> 40 s (the cap)
        env.step(0)
    assert env.strategy.last_obs[22] * rl_config.MAX_GREEN_NORM == pytest.approx(40.0)
    env.step(0)                                       # "extend" at the cap
    assert env.scheduler.active_phase_type.name == "PHASE_2"


def test_reward_equals_scaled_queueing_delay_over_the_step():
    env = _env(profile_key="RUSH_HOUR")
    env.reset()
    env.step(0)
    log = _record_ticks(env)
    _, reward, _, _ = env.step(3)
    expected = -sum(q for _, _, q in log) * env.tick_interval / rl_config.REWARD_SCALE
    assert reward == pytest.approx(expected)
    assert reward < 0


def test_episode_ends_at_time_limit_as_truncation():
    env = _env(episode_length=120)
    env.reset()
    done, steps = False, 0
    while not done:
        _, _, done, info = env.step(steps % 10)
        steps += 1
    assert env.tick == 120
    assert info["truncated"] is True


def test_step_rejects_invalid_use():
    env = _env()
    env.reset()
    with pytest.raises(ValueError):
        env.step(10)
    env.strategy.awaiting_action = False
    with pytest.raises(RuntimeError):
        env.step(0)


def test_same_seed_is_reproducible():
    rewards = []
    for _ in range(2):
        env = _env(profile_key="CUSTOM", seed=7, episode_length=400)
        env.reset()
        done, run = False, []
        while not done:
            _, r, done, _ = env.step(len(run) % 10)
            run.append(r)
        rewards.append(run)
    assert rewards[0] == rewards[1]


class _ReplayAgent:
    """Tabular-style agent that replays a fixed action sequence."""
    Q = None

    def __init__(self, actions):
        self.actions = list(actions)
        self.i = 0

    def select_action(self, _state):
        action = self.actions[self.i % len(self.actions)]
        self.i += 1
        return action


def test_training_and_inference_compute_identical_observations():
    """Regression (Step 1): elapsed-time feature differed between the two paths."""
    actions = [3, 3, 7, 1, 0, 0, 0, 9, 4, 2, 8, 6, 5] * 3
    ticks = 600

    env = _env(profile_key="RUSH_HOUR", seed=7, episode_length=ticks)
    env_obs = [env.reset()]
    for action in actions:
        obs, _, done, _ = env.step(action)
        if done:
            break
        env_obs.append(obs)

    strategy = RLStrategy(agent=_ReplayAgent(actions))
    sim_obs = []
    original = strategy.observe

    def capture(*args):
        result = original(*args)
        sim_obs.append(result[0].copy())
        return result

    strategy.observe = capture
    sim = Simulation(profile_key="RUSH_HOUR", seed=7, max_ticks=ticks,
                     live=False, strategy=strategy)
    for _ in range(ticks):
        sim.step()

    n = min(len(env_obs), len(sim_obs))
    assert n >= 15
    for i in range(n):
        np.testing.assert_array_equal(env_obs[i], sim_obs[i], err_msg=f"decision {i}")
