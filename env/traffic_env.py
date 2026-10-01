"""
traffic_env.py

A Gym-style environment wrapping the existing simulator.

The environment drives the SAME scheduler/service/spawn machinery as the
regular Simulation class, but:
  - decisions are made by an RLStrategy that holds a `pending_phase`
    chosen by the agent (inference-time: argmax, training-time: epsilon-
    greedy), rather than by a rule-based strategy.
  - A decision point is the moment the scheduler asks for the next phase
    (first tick, and after each phase's green + yellow). The env pauses the
    tick right there, hands the observation to the agent, and resumes the
    same tick once step(action) supplies the phase - so no simulated time
    is spent waiting for the agent.
  - step(a) applies `a` immediately and returns the reward accumulated
    while `a`'s phase was in control, i.e. the reward is credited to the
    action that caused it.

Contract:
  reset()              -> observation (23-dim) at the first decision point
  step(action)         -> (next_obs, reward, done, info)
  discrete_state       -> tabular bucket for the current decision point
  action_space         -> n = 10 (choose one normal phase)
  observation_space    -> shape = (23,)

Reward: r = -(queueing delay during the step) / REWARD_SCALE, where the
delay is sum over ticks of (total queue * tick seconds) = vehicle-seconds
spent waiting. Minimizing it minimizes total delay.

Episodes end on a time limit only. Traffic never "finishes", so the final
step is a truncation, not a terminal state: info["truncated"] is True and
learners should still bootstrap from next_obs.

Emergency/ambulance handling is untouched: the scheduler's preemption state
machine runs internally and never consults the strategy, so the agent never
sees or acts during an emergency window.
"""
from config.phases import all_phase_types
from config import rl as rl_config
from config import simulation as sim_config
from core.intersection import Intersection
from scheduler.traffic_scheduler import TrafficScheduler
from traffic_source.profile_traffic_source import ProfileTrafficSource
from services.service_model import ServiceModel
from strategies.rl_strategy import RLStrategy


class TrafficRLEnv:
    """
    Gym-style RL wrapper around the traffic simulator.

    Args:
        profile_key (str): traffic profile for the step.
        episode_length (int): number of simulation ticks per episode.
        seed (int|None): base seed for the traffic source.
        tick_interval (float): seconds per simulated tick.
    """

    def __init__(
        self,
        profile_key="NORMAL_TRAFFIC",
        episode_length=None,
        seed=None,
        tick_interval=None,
    ):
        self.profile_key = profile_key
        self.episode_length = (
            episode_length
            if episode_length is not None
            else rl_config.EPISODE_LENGTH
        )
        self.seed = seed if seed is not None else rl_config.SEED
        self.tick_interval = (
            tick_interval
            if tick_interval is not None
            else sim_config.TICK_DURATION
        )

        self.phase_types = all_phase_types()
        self.action_space = len(self.phase_types)  # 10 discrete actions
        self.observation_space = (23,)

        # Runtime state (rebuilt each reset).
        self.intersection = None
        self.scheduler = None
        self.service_model = None
        self.traffic_source = None
        self.strategy = None
        self.tick = 0

    # ------------------------------------------------------------------
    # Gym-style interface
    # ------------------------------------------------------------------

    def reset(self, seed=None):
        """
        Reset the simulator, advance to the first decision point and return
        the initial observation.

        Any FIXED seed is reproducible (used for evaluation comparisons).
        """
        if seed is not None:
            self.seed = seed

        # Fresh world.
        self.intersection = Intersection()
        self.strategy = RLStrategy()
        self.scheduler = TrafficScheduler(
            self.intersection,
            self.strategy,
            yellow_duration=sim_config.YELLOW_TIME,
            emergency_yellow_duration=sim_config.EMERGENCY_YELLOW_TIME,
            emergency_max_timeout=sim_config.EMERGENCY_MAX_TIMEOUT,
        )
        self.service_model = ServiceModel()
        self.traffic_source = ProfileTrafficSource(
            profile_key=self.profile_key,
            seed=self.seed,
            tick_duration=self.tick_interval,
        )
        self.tick = 0

        # The scheduler asks for a phase on the very first tick.
        _, done = self._run_until_decision()
        assert not done, "episode ended before the first decision"
        return self.strategy.last_obs

    def step(self, action):
        """
        Apply `action` at the current decision point, run the simulator
        until the NEXT decision point, and return (next_obs, reward, done,
        info). The reward covers exactly the ticks during which this
        action was in control (an extension, or yellow + a new phase's
        minimum green). info["duration"] is that span in seconds.

        Args:
            action (int): index into all_phase_types() (0..9).
        """
        if not 0 <= action < self.action_space:
            raise ValueError(f"action {action} out of range [0, {self.action_space})")
        if not self.strategy.awaiting_action:
            raise RuntimeError("step() called while not at a decision point")

        # Resume the paused tick: the scheduler re-asks the strategy (with a
        # zero time delta), which consumes the pending action.
        start_tick = self.tick
        self.strategy.set_pending(action)
        self.scheduler.update(0.0)
        assert not self.strategy.awaiting_action

        reward = self._finish_tick()
        more_reward, done = self._run_until_decision()
        reward += more_reward

        if done:
            # Episode cut off mid-phase: report the state as of now.
            self.strategy.observe(
                self.intersection,
                self.scheduler.current_phase,
                self.intersection.time,
            )
        info = {
            "truncated": done,
            "duration": (self.tick - start_tick) * self.tick_interval,
        }
        return self.strategy.last_obs, reward, done, info

    # ------------------------------------------------------------------
    # Tick mechanics (mirrors Simulation.step's order)
    # ------------------------------------------------------------------

    def _run_until_decision(self):
        """
        Advance whole ticks until the scheduler asks for a decision (that
        tick is then paused after its scheduler update) or the episode
        budget is exhausted.

        Returns:
            (float, bool): (reward accumulated, done).
        """
        reward = 0.0
        while self.tick < self.episode_length:
            self._start_tick()
            if self.strategy.awaiting_action:
                return reward, False
            reward += self._finish_tick()
        return reward, True

    def _start_tick(self):
        """First half of a tick: spawn vehicles, advance the scheduler."""
        spawns = self.traffic_source.generate_spawns(self.intersection.time)
        self.intersection.spawn_batch(spawns)
        # May end in a decision request (strategy.awaiting_action).
        self.scheduler.update(self.tick_interval)

    def _finish_tick(self) -> float:
        """Second half of a tick: discharge, waits, clock. Returns reward."""
        active = self.scheduler.active_movements()
        self.service_model.accumulate(active, self.tick_interval)
        self.service_model.discharge(active)

        self.intersection.update_waiting_times(self.tick_interval)
        self.intersection.advance_time(self.tick_interval)
        self.tick += 1
        return self._tick_reward()

    def _tick_reward(self) -> float:
        """Per-tick reward: -(vehicle-seconds queued this tick), scaled."""
        delay = self.intersection.total_queue_length() * self.tick_interval
        return -delay / rl_config.REWARD_SCALE

    # ------------------------------------------------------------------
    # Discrete state (for tabular Q)
    # ------------------------------------------------------------------

    @property
    def discrete_state(self) -> int:
        """Tabular bucket for the current decision point."""
        return self.strategy.last_state

    def __repr__(self):
        return (
            f"TrafficRLEnv(profile={self.profile_key}, "
            f"tick={self.tick}/{self.episode_length}, "
            f"action_space={self.action_space})"
        )
