"""
env package

A Gym-style wrapper around the traffic simulator (Phase 4). It exposes the
standard reset()/step() interface so any reinforcement-learning agent can
drive the simulator without touching the scheduler/controller code.

The environment:
    - observation : per-lane vector (count, mean/longest wait, long waiters
                    per lane; active-phase one-hot; elapsed green), built by
                    RLStrategy.observe() for training and inference alike.
    - action      : one of the 10 phases each time the active green runs
                    out (same phase = extend 5 s, other phase = switch).
    - reward      : -(wait-weighted queueing delay over the step) / REWARD_SCALE.
    - done        : episode time budget used up (a truncation:
                    info["truncated"] is True and learners still bootstrap).

Emergency preemption remains fully rule-based. The scheduler handles it
internally without consulting the strategy, so the agent never sees or acts
during an emergency window.
"""
from .traffic_env import TrafficRLEnv

__all__ = ["TrafficRLEnv"]
