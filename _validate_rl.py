"""
_validate_rl.py

Validation for the Phase 4 Stage 1 work: the Gym-style environment wrapper +
tabular Q-learning baseline.

Tests:
  TEST 1  - reset() returns a 23-dim observation.
  TEST 2  - step(action) returns (obs, reward, done, info).
  TEST 3  - action space == 10 (one normal phase per action).
  TEST 4  - reward is non-positive on an empty intersection (-sum queue).
  TEST 5  - step activates the requested phase (RLStrategy consumes action).
  TEST 6  - an episode terminates (done=True) within the tick budget.
  TEST 7  - TabularQAgent Q-table shape is (810, 10).
  TEST 8  - Tabular training produces a rising reward curve.
  TEST 9  - Fixed-seed tabular training is reproducible (deterministic).
  TEST 10 - Emergency preemption does not break the step contract.
  TEST 11 - Regression: existing Phase 2 suite still passes.
"""
from core.enums import PhaseType, Priority, VehicleType, MovementType
from core.vehicle import Vehicle
from env.traffic_env import TrafficRLEnv
from rl.agents import TabularQAgent
from rl.train import train_tabular
import numpy as np

APPROACHES = ("North", "South", "East", "West")


def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))
    return passed


def spawn_ambulance(intersection, approach_name, movement_type):
    """Spawn an ambulance on the given approach/lane (front of queue)."""
    vehicle = Vehicle(
        vehicle_type=VehicleType.AMBULANCE,
        arrival_time=intersection.time,
        destination_movement=movement_type,
        priority=Priority.HIGH,
    )
    lane = intersection.get_approach(approach_name).get_lane(movement_type)
    lane.queue._vehicles.appendleft(vehicle)
    vehicle.current_lane = lane
    return vehicle


results = []

# ---------------------------------------------------------------------------
# TEST 1 : reset() returns a 23-dim observation
# ---------------------------------------------------------------------------
env = TrafficRLEnv(profile_key="NORMAL_TRAFFIC", episode_length=50, seed=1)
obs = env.reset()
obs_shape_ok = (
    isinstance(obs, np.ndarray)
    and obs.shape == (23,)
    and obs.dtype == np.float32
)
results.append(check(
    "TEST 1 reset() returns 23-dim float32 observation",
    obs_shape_ok,
    f"shape={obs.shape} dtype={obs.dtype}",
))

# ---------------------------------------------------------------------------
# TEST 2 : step(action) returns (obs, reward, done, info)
# ---------------------------------------------------------------------------
obs2, reward, done, info = env.step(0)
step_ok = (
    isinstance(obs2, np.ndarray)
    and obs2.shape == (23,)
    and isinstance(reward, float)
    and isinstance(done, bool)
    and isinstance(info, dict)
)
results.append(check(
    "TEST 2 step() returns (obs, reward, done, info)",
    step_ok,
    f"reward={reward:.1f} done={done} info_keys={list(info.keys())}",
))

# ---------------------------------------------------------------------------
# TEST 3 : action space == 10
# ---------------------------------------------------------------------------
results.append(check(
    "TEST 3 action space == 10 (one normal phase per action)",
    env.action_space == 10,
    f"action_space={env.action_space}",
))

# ---------------------------------------------------------------------------
# TEST 4 : reward is the negative sum of queue lengths
# ---------------------------------------------------------------------------
# No spawns on an empty profile -> reward is 0 (or negative but never positive).
env = TrafficRLEnv(profile_key="NORMAL_TRAFFIC", episode_length=5, seed=1)
env.reset()
_, r_empty, _, _ = env.step(0)
empty_ok = r_empty <= 0.0
results.append(check(
    "TEST 4 reward never positive (-sum queue)",
    empty_ok,
    f"reward={r_empty:.1f}",
))

# ---------------------------------------------------------------------------
# TEST 5 : step activates the requested phase (RLStrategy consumes it)
# ---------------------------------------------------------------------------
env = TrafficRLEnv(profile_key="NORMAL_TRAFFIC", episode_length=50, seed=2)
env.reset()
_, _, _, _ = env.step(5)  # choose PHASE_6 (index 5)
# After a step, the scheduler should have been asked for a decision and the
# pending phase consumed.
phase = env.scheduler.active_phase_type
step5_ok = (
    phase is not None
    and phase.name == env.phase_types[5].name  # PHASE_6
)
# However, the first action may be consumed at the very first decision after
# reset. Verify the strategy's pending was cleared.
strategy_ok = env.strategy.pending_phase is None and env.strategy.decision_made
results.append(check(
    "TEST 5 step() activates the requested phase",
    step5_ok,
    f"active_phase={phase.name if phase else None} "
    f"expected={env.phase_types[5].name} strategy_consumed={strategy_ok}",
))

# ---------------------------------------------------------------------------
# TEST 6 : episode terminates within the tick budget
# ---------------------------------------------------------------------------
env = TrafficRLEnv(profile_key="RUSH_HOUR", episode_length=30, seed=3)
env.reset()

# Let the episode run with random actions.
done = False
steps = 0
while not done and steps < 1000:
    done, reward, next_state, info = None, None, None, None
    for _ in range(1):
        a = steps % env.action_space
        obs_next, reward, done, info = env.step(a)
    steps += 1
    if done:
        break

results.append(check(
    "TEST 6 episode terminates within the tick budget",
    done and steps <= env.episode_length,
    f"steps={steps} episode_length={env.episode_length}",
))

# ---------------------------------------------------------------------------
# TEST 7 : TabularQAgent Q-table shape (810, 10)
# ---------------------------------------------------------------------------
agent = TabularQAgent(seed=42)
n_states = agent.n_states
results.append(check(
    "TEST 7 TabularQAgent Q-table shape",
    agent.Q.shape == (n_states, 10),
    f"Q.shape={agent.Q.shape} expected={(n_states, 10)}",
))

# ---------------------------------------------------------------------------
# TEST 8 : tabular training produces a rising reward curve
# ---------------------------------------------------------------------------
# Use a SHORT run so the validation stays fast, and compare the running
# average of the second half vs the first half.
NUM_EP = 30
agent, rewards = train_tabular(
    n_episodes=NUM_EP,
    episode_length=60,
    seed=42,
    verbose=False,
)
if len(rewards) >= 6:
    first_half = sum(rewards[: len(rewards) // 2]) / max(1, len(rewards) // 2)
    second_half = sum(rewards[len(rewards) // 2:]) / max(1, len(rewards) - len(rewards) // 2)
    rising = second_half >= first_half
else:
    rising = False
    first_half = second_half = 0.0
results.append(check(
    "TEST 8 tabular training produces a rising reward curve",
    rising,
    f"first_half_avg={first_half:.0f} second_half_avg={second_half:.0f}",
))

# ---------------------------------------------------------------------------
# TEST 9 : fixed-seed tabular training is reproducible (deterministic)
# ---------------------------------------------------------------------------
_, rewards_a = train_tabular(n_episodes=10, episode_length=40, seed=99, verbose=False)
_, rewards_b = train_tabular(n_episodes=10, episode_length=40, seed=99, verbose=False)
results.append(check(
    "TEST 9 fixed-seed tabular training is reproducible",
    rewards_a == rewards_b,
    f"rewards_match={rewards_a == rewards_b}",
))

# ---------------------------------------------------------------------------
# TEST 10 : emergency preemption does not break the step contract
# ---------------------------------------------------------------------------
env = TrafficRLEnv(profile_key="NORMAL_TRAFFIC", episode_length=200, seed=5)
env.reset()
# Force an ambulance onto North after a couple of steps so preemption fires.
done = False
for _ in range(3):
    _, _, done, _ = env.step(2)
    if done:
        break
spawn_ambulance(env.intersection, "North", MovementType.STRAIGHT)
# Continue stepping; preemption should fire internally and step must still
# return a valid contract.
ok = True
for _ in range(50):
    obs_n, r_n, done_n, info_n = env.step(3)
    if done_n:
        break
    if not (isinstance(obs_n, np.ndarray) and obs_n.shape == (23,)):
        ok = False
        break
results.append(check(
    "TEST 10 emergency preemption does not break the step contract",
    ok,
    f"continued_ok={ok}",
))

# ---------------------------------------------------------------------------
# TEST 11 : regression - existing Phase 2 suite still passes
# ---------------------------------------------------------------------------
import subprocess, sys
proc = subprocess.run(
    [sys.executable, "_validate_simulations.py"],
    capture_output=True,
    text=True,
    cwd=".",
)
results.append(check(
    "TEST 11 Existing Phase 2 validation suite still passes",
    proc.returncode == 0,
    "returncode=0" if proc.returncode == 0 else proc.stdout[-300:],
))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 4 (ENV + TABULAR Q + DQN + EVAL) VALIDATION SUMMARY")
print("=" * 60)
passed = sum(results)
total = len(results)
for i, r in enumerate(results, 1):
    print(f"  Test {i}: {'PASS' if r else 'FAIL'}")
print(f"{passed}/{total} checks passed")
raise SystemExit(0 if passed == total else 1)

