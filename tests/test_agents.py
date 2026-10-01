"""Learning algorithms: tabular Q update, epsilon schedule, numpy MLP/Adam/DQN."""
import numpy as np
import pytest

from config import rl as rl_config
from rl.agents import TabularQAgent
from rl.dqn import MLP, Adam, DQNAgent
from rl.train import epsilon_at


# ---------------------------------------------------------------- tabular

def test_q_update_bootstraps_with_given_discount():
    agent = TabularQAgent(n_states=2, n_actions=3, alpha=0.1)
    agent.Q[1] = [2.0, 4.0, 1.0]
    agent.update(0, 1, reward=-1.0, next_state=1, terminal=False, discount=0.5)
    # target = -1 + 0.5 * max(Q[1]) = 1  ->  Q += 0.1 * (1 - 0)
    assert agent.Q[0, 1] == pytest.approx(0.1)


def test_q_update_does_not_bootstrap_from_terminal_state():
    agent = TabularQAgent(n_states=2, n_actions=3, alpha=0.1)
    agent.Q[1] = [100.0, 100.0, 100.0]
    agent.update(0, 0, reward=-1.0, next_state=1, terminal=True, discount=0.9)
    assert agent.Q[0, 0] == pytest.approx(-0.1)


def test_tabular_save_load_roundtrip(tmp_path):
    agent = TabularQAgent(seed=0)
    agent.Q = np.random.default_rng(0).normal(size=agent.Q.shape)
    path = tmp_path / "q.npy"
    agent.save(path)
    loaded = TabularQAgent()
    loaded.load(path)
    np.testing.assert_array_equal(agent.Q, loaded.Q)


# ---------------------------------------------------------------- schedule

def test_epsilon_decays_linearly_then_holds_at_floor():
    n = 100
    decay_end = int(n * rl_config.EPSILON_DECAY_FRACTION)
    assert epsilon_at(0, n) == pytest.approx(rl_config.EPSILON_START)
    assert epsilon_at(decay_end // 2, n) == pytest.approx(
        (rl_config.EPSILON_START + rl_config.EPSILON_END) / 2)
    assert epsilon_at(decay_end, n) == pytest.approx(rl_config.EPSILON_END)
    assert epsilon_at(n - 1, n) == pytest.approx(rl_config.EPSILON_END)


# ---------------------------------------------------------------- MLP / Adam

def test_mlp_backprop_matches_finite_differences():
    """The hand-written backward pass must equal numerical gradients."""
    rng = np.random.default_rng(0)
    net = MLP(5, 3, hidden=(4, 4), seed=0, std=1.0)
    x = rng.normal(size=(6, 5))
    y = rng.normal(size=(6, 3))

    def loss():
        return 0.5 * float(((net.forward(x) - y) ** 2).sum())

    out = net.forward(x)
    net.zero_grad()
    net.backward(out - y)
    analytic = {i: {"W": g["dW"].copy(), "b": g["db"].copy()}
                for i, g in net.grads.items()}

    eps = 1e-6
    for i, layer in net.params.items():
        for key in ("W", "b"):
            param = layer[key]
            for idx in [(0, 0), (param.shape[0] - 1, param.shape[1] - 1)]:
                original = param[idx]
                param[idx] = original + eps
                up = loss()
                param[idx] = original - eps
                down = loss()
                param[idx] = original
                numeric = (up - down) / (2 * eps)
                assert analytic[i][key][idx] == pytest.approx(numeric, rel=1e-4, abs=1e-7), \
                    f"layer {i} {key}{idx}"


def test_adam_fits_a_small_regression():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(64, 3))
    y = x @ np.array([[1.0], [-2.0], [0.5]])
    net = MLP(3, 1, hidden=(16,), seed=1)
    opt = Adam(net, lr=1e-2)

    def mse():
        return float(((net.forward(x) - y) ** 2).mean())

    start = mse()
    for _ in range(300):
        out = net.forward(x)
        net.zero_grad()
        net.backward((out - y) / len(x))
        opt.step(max_grad_norm=10.0)
    assert mse() < 0.05 * start


# ---------------------------------------------------------------- DQN

def test_dqn_learns_a_one_step_bandit():
    """Action 3 pays 1, every other action pays 0: the greedy action must be 3."""
    agent = DQNAgent(obs_dim=4, n_actions=5, seed=0, batch_size=32)
    rng = np.random.default_rng(0)
    s = np.ones(4)
    for _ in range(1500):
        a = int(rng.integers(5))
        agent.store(s, a, 1.0 if a == 3 else 0.0, s, terminal=True)
    q = agent.policy_net.predict(s)[0]
    assert int(np.argmax(q)) == 3
    assert q[3] == pytest.approx(1.0, abs=0.15)


def test_dqn_save_load_and_weight_snapshots(tmp_path):
    agent = DQNAgent(obs_dim=23, n_actions=10, seed=3)
    obs = np.linspace(0, 1, 23)
    before = agent.policy_net.predict(obs).copy()

    path = tmp_path / "dqn.npz"
    agent.save(path)
    other = DQNAgent(obs_dim=23, n_actions=10, seed=99)
    other.load(path)
    np.testing.assert_allclose(other.policy_net.predict(obs), before)
    np.testing.assert_allclose(other.target_net.predict(obs), before)

    snapshot = agent.get_weights()
    agent.policy_net.params[0]["W"] += 1.0
    agent.set_weights(snapshot)
    np.testing.assert_allclose(agent.policy_net.predict(obs), before)
