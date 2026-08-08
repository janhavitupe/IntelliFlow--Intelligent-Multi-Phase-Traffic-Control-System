"""
env package

A Gym-style wrapper around the traffic simulator (Phase 4). It exposes the
standard reset()/step() interface so any reinforcement-learning agent can
drive the simulator without touching the scheduler/controller code.

The environment:
    - observation : reuses the Density strategy's features (queue length,
                    percentile rank, starvation) plus active-phase one-hot
                    and elapsed time -> a 23-dim vector.
    - action      : choose one of the 10 normal phases at each decision
                    point (not every tick).
    - reward      : r = -sum(queue_lengths) over the step (crude congestion
                    proxy, sufficient to get a learning signal).
    - done        : True once the episode's simulation-time budget is used.

Emergency preemption remains fully rule-based. The scheduler handles it
internally without consulting the strategy, so the agent never sees or acts
during an emergency window.
"""
from .traffic_env import TrafficRLEnv

__all__ = ["TrafficRLEnv"]
