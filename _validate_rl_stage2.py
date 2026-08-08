"""
_validate_rl_stage2.py

Validation for Phase 4 Stage 2: pure-numpy DQN + RLStrategy self-driving +
the three-way evaluation harness.

Tests:
  TEST 1  - DQN MLP produces finite, non-zero gradients (backprop ran).
  TEST 2  - DQN training loop returns a reward curve.
  TEST 3  - RLStrategy self-drives with a DQN agent (argmax, no env).
  TEST 4  - RLStrategy self-drives with a tabular agent.
  TEST 5  - three-way evaluation harness runs all strategies.
  TEST 6  - Simulation accepts an injected RLStrategy (integration).
  TEST 7  - regression: existing Phase 4 Stage 1 suite still passes.
"""
import numpy as np

from core.enums import PhaseType
from core.intersection import Intersection
from simulation import Simulation

from rl.dqn import MLP, DQNAgent
from rl.train import train_dqn
from strategies.fixed_timer_strategy import FixedTimerStrategy
from strategies.density_strategy import DensityStrategy
from strategies.rl_strategy import RLStrategy
from rl.agents import TabularQAgent
from rl.train import train_tabular
from evaluation.evaluate import evaluate_strategies


def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))
    return passed


results = []

# ---------------------------------------------------------------------------
# TEST 1 : DQN MLP gradient sanity
# ---------------------------------------------------------------------------
X = np.random.default_rng(0).normal(size=(5, 3))
D = np.random.default_rng(1).normal(size=(5, 2))
m = MLP(3, 2, hidden=(4, 4), seed=0, std=0.5)
m.forward(X)
m.zero_grad()
m.backward(D)
grad_norm = sum(
    float(np.sum(m.grads[i]["dW"] ** 2)) + float(np.sum(m.grads[i]["db"] ** 2))
    for i in range(len(m.params))
)
test1 = np.isfinite(grad_norm) and grad_norm > 0.0
results.append(check(
    "TEST 1 DQN MLP backprop finite, non-zero gradients",
    test1,
    f"grad_norm={grad_norm:.3e}",
))

# ---------------------------------------------------------------------------
# TEST 2 : DQN training loop returns a reward curve
# ---------------------------------------------------------------------------
dqn_agent, dqn_rewards = train_dqn(
    agent=DQNAgent(obs_dim=23, n_actions=10, hidden=(8, 8),
                   batch_size=8, replay_size=200, seed=7),
    n_episodes=4,
    episode_length=30,
    seed=7,
    verbose=False,
)
test2 = (
    len(dqn_rewards) == 4
    and all(isinstance(r, float) for r in dqn_rewards)
    and hasattr(dqn_agent, "policy_net")
)
results.append(check(
    "TEST 2 DQN training loop returns a reward curve",
    test2,
    f"rewards={[round(r, 1) for r in dqn_rewards]}",
))

# ---------------------------------------------------------------------------
# TEST 3 : RLStrategy self-drives with a DQN agent (no env injection)
# ---------------------------------------------------------------------------
dqn_rl = RLStrategy(agent=dqn_agent)
dqn_rl.reset()
intersection = Intersection()
phase, green = dqn_rl.decide_next_phase(intersection, None, 0.0)
test3 = (
    phase is not None
    and isinstance(green, float)
    and 0 <= list(PhaseType).index(phase) < 10
)
results.append(check(
    "TEST 3 RLStrategy self-drives with a DQN agent (argmax, no env)",
    test3,
    f"phase={phase.name if phase else None} green={green}",
))

# ---------------------------------------------------------------------------
# TEST 4 : RLStrategy self-drives with a tabular agent
# ---------------------------------------------------------------------------
tabular_agent, _ = train_tabular(n_episodes=5, episode_length=30,
                                 seed=42, verbose=False)
tabular_rl = RLStrategy(agent=tabular_agent)
tabular_rl.reset()
intersection4 = Intersection()
phase4, green4 = tabular_rl.decide_next_phase(intersection4, None, 0.0)
test4 = phase4 is not None and isinstance(green4, float)
results.append(check(
    "TEST 4 RLStrategy self-drives with a tabular agent",
    test4,
    f"phase={phase4.name if phase4 else None} green={green4}",
))

# ---------------------------------------------------------------------------
# TEST 5 : three-way evaluation harness runs all strategies
# ---------------------------------------------------------------------------
eval_strategies = {
    "fixed_timer": FixedTimerStrategy(),
    "density": DensityStrategy(),
    "rl_tabular": tabular_rl,
}
eval_res = evaluate_strategies(
    eval_strategies, profile_key="NORMAL_TRAFFIC", seed=42, max_ticks=60
)
test5 = (
    set(eval_res.keys()) == set(eval_strategies.keys())
    and all("average_waiting_time" in s for s in eval_res.values())
    and all("throughput" in s for s in eval_res.values())
    and all("congestion_ratio" in s for s in eval_res.values())
)
results.append(check(
    "TEST 5 three-way evaluation harness runs all strategies",
    test5,
    f"strategies={list(eval_res.keys())}",
))

# ---------------------------------------------------------------------------
# TEST 6 : Simulation accepts an injected RLStrategy (integration)
# ---------------------------------------------------------------------------
sim_rl = Simulation(
    profile_key="NORMAL_TRAFFIC",
    seed=42,
    max_ticks=30,
    live=False,
    strategy=tabular_rl,
)
for _ in range(30):
    sim_rl.step()
test6 = sim_rl.tick == 30 and sim_rl.analytics.total_vehicles_spawned >= 0
results.append(check(
    "TEST 6 Simulation runs with an injected RLStrategy",
    test6,
    f"ticks={sim_rl.tick} spawned={sim_rl.analytics.total_vehicles_spawned}",
))

# ---------------------------------------------------------------------------
# TEST 7 : regression - existing Phase 4 Stage 1 suite still passes
# ---------------------------------------------------------------------------
import subprocess, sys
proc = subprocess.run(
    [sys.executable, "_validate_rl.py"],
    capture_output=True,
    text=True,
    cwd=".",
)
test7 = proc.returncode == 0
results.append(check(
    "TEST 7 Existing Phase 4 Stage 1 suite still passes",
    test7,
    "returncode=0" if test7 else proc.stdout[-400:],
))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 4 STAGE 2 (DQN + EVAL) VALIDATION SUMMARY")
print("=" * 60)
passed = sum(results)
total = len(results)
for i, r in enumerate(results, 1):
    print(f"  Test {i}: {'PASS' if r else 'FAIL'}")
print(f"{passed}/{total} checks passed")
raise SystemExit(0 if passed == total else 1)
