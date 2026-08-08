"""
traffic_env.py

A Gym-style environment wrapping the existing simulator.

The environment drives the SAME scheduler/service/spawn machinery as the
regular Simulation class, but:
  - decisions are made by an RLStrategy that holds a `pending_phase`
    chosen by the agent (inference-time: argmax, training-time: epsilon-
    greedy), rather than by a rule-based strategy.
  - step() advances the simulator until the NEXT decision point (the moment
    the scheduler requests a new phase at a minimum-green boundary), then
    returns (next_state, reward, done, info).

Contract:
  reset()              -> observation (23-dim)
  step(action)         -> (next_obs, reward, done, info)
  action_space         -> n = 10 (choose one normal phase)
  observation_space    -> shape = (23,)

Reward: r = -sum(queue_lengths) accumulated over the tick(s) of the step.
This is a crude "minimize congestion" proxy sufficient to get a signal.

Emergency/ambulance handling is untouched: the scheduler's preemption state
machine runs internally and never consults the strategy, so the agent never
sees or acts during an emergency window.
"""
import numpy as np

from config.phases import all_phase_types
from config import rl as rl_config
from config import simulation as sim_config
from core.enums import PhaseType
from core.intersection import Intersection
from scheduler.traffic_scheduler import TrafficScheduler
from traffic_source.profile_traffic_source import ProfileTrafficSource
from services.service_model import ServiceModel
from strategies.rl_strategy import RLStrategy
from .state_builder import ObservationBuilder, Discretizer


class TrafficRLEnv:
    """
    Gym-style RL wrapper around the traffic simulator.

    Args:
        profile_key (str): traffic profile for the step.
        episode_length (int): number of simulation ticks per episode.
        seed (int|None): base seed for the traffic source.
        green_duration (float): green seconds granted to the chosen phase.
        tick_interval (float): seconds per simulated tick.
    """

    def __init__(
        self,
        profile_key="NORMAL_TRAFFIC",
        episode_length=None,
        seed=None,
        green_duration=None,
        tick_interval=None,
    ):
        self.profile_key = profile_key
        self.episode_length = (
            episode_length
            if episode_length is not None
            else rl_config.EPISODE_LENGTH
        )
        self.seed = seed if seed is not None else rl_config.SEED
        self.green_duration = (
            green_duration
            if green_duration is not None
            else rl_config.GREEN_DURATION
        )
        self.tick_interval = (
            tick_interval
            if tick_interval is not None
            else sim_config.TICK_DURATION
        )

        self.phase_types = all_phase_types()
        self.action_space = len(self.phase_types)  # 10 discrete actions
        self.observation_space = (23,)

        # State builders (reuse the Density strategy's features).
        self.obs_builder = ObservationBuilder()
        self.discretizer = Discretizer()

        # Optional rule-based density reference for starvation counters.
        self.density_ref = None

        # Runtime state (rebuilt each reset).
        self.intersection = None
        self.scheduler = None
        self.service_model = None
        self.traffic_source = None
        self.strategy = None
        self.tick = 0
        self._pending_phase = None
        self._decision_ready = False
        self._last_phase = None
        self._elapsed_in_phase = 0.0

    # ------------------------------------------------------------------
    # Spatial / bookkeeping helpers
    # ------------------------------------------------------------------

    def _approach_counts(self) -> dict:
        return {
            name: self.intersection.get_approach(name).total_queue_length()
            for name in ("North", "South", "East", "West")
        }

    def _elapsed_in_current_phase(self) -> float:
        """Seconds already spent in the current phase (green+yellow)."""
        if self.scheduler is None or self.scheduler.current_phase is None:
            return 0.0
        # Reconstruct from the intersection clock minus phase start time.
        # We track it incrementally instead for simplicity.
        return self._elapsed_in_phase

    # ------------------------------------------------------------------
    # Gym-style interface
    # ------------------------------------------------------------------

    def reset(self, seed=None):
        """
        Reset the simulator and return the initial observation.

        The environment picks a fresh per-episode seed (SEED + episode
        counter) so each episode sees varied traffic, but any FIXED seed is
        reproducible (used for evaluation comparisons).
        """
        if seed is not None:
            self.seed = seed

        # Fresh world.
        self.intersection = Intersection()
        self.strategy = RLStrategy()  # holds pending_phase set by the agent
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
        self._pending_phase = None
        self._decision_ready = True
        self._last_phase = None
        self._elapsed_in_phase = 0.0
        self.episode_reward = 0.0

        # Let the strategy build its phase plan lazily (needed for the
        # scheduler to call decide_next_phase against a real intersection).
        self.strategy.reset(self.intersection)

        return self._observe()

    def step(self, action):
        """
        Activate the chosen phase, advance until the next decision point,
        and return (next_obs, reward, done, info).

        Args:
            action (int): index into all_phase_types() (0..9).
        """
        if not 0 <= action < self.action_space:
            raise ValueError(f"action {action} out of range [0, {self.action_space})")

        phase_type = self.phase_types[action]
        self._pending_phase = phase_type
        # Push the chosen phase to the strategy so the scheduler's next
        # decision request can consume it.
        self.strategy.set_pending(phase_type)
        self.strategy.decision_made = False
        self._decision_ready = False

        reward = 0.0
        done = False
        info = {}

        # Advance ticks until the scheduler requests the next decision,
        # or the episode budget is exhausted.
        while not self._decision_ready and not done:
            self._advance_tick()
            reward += self._tick_reward()
            if self.tick >= self.episode_length:
                done = True

        next_obs = self._observe()
        info["decision_point_reached"] = True
        return next_obs, reward, done, info

    def _advance_tick(self):
        """Run one simulator tick (mirrors Simulation.step's core order)."""
        t = self.intersection.time

        # 1. Spawn new vehicles.
        spawns = self.traffic_source.generate_spawns(t)
        self.intersection.spawn_batch(spawns)

        # 2. Advance scheduler (phase transitions + emergency preemption).
        #    The RLStrategy's decide_next_phase() will consume the pending
        #    phase and set _decision_ready when it is asked for a decision.
        self.scheduler.update(self.tick_interval)

        # 3. Discharge via the service model.
        active = self.scheduler.active_movements()
        self.service_model.accumulate(active, self.tick_interval)
        self.service_model.discharge(active)

        # 4. Advance waiting times + clock.
        self.intersection.update_waiting_times(self.tick_interval)
        self.intersection.advance_time(self.tick_interval)

        # 5. Track phase elapsed time.
        self._update_elapsed()

        # 6. Detect whether the scheduler requested (and the strategy made) a
        #    decision this tick. This marks the next decision point.
        self._decision_ready = self.strategy.decision_made
        self.strategy.decision_made = False

        self.tick += 1

    def _update_elapsed(self):
        """Increment elapsed time if a phase is active; reset on switch."""
        if self.scheduler is not None and self.scheduler.current_phase is not None:
            self._elapsed_in_phase += self.tick_interval
        else:
            self._elapsed_in_phase = 0.0

    def _tick_reward(self) -> float:
        """Per-tick reward: -total queue length."""
        return -float(self.intersection.total_queue_length())

    def _observe(self) -> np.ndarray:
        """Build the 23-dim observation vector."""
        active = self.scheduler.active_phase_type
        return self.obs_builder.build(
            self.intersection,
            active,
            self._elapsed_in_phase,
        )

    # ------------------------------------------------------------------
    # Discrete state (for tabular Q)
    # ------------------------------------------------------------------

    def discretize_state(self, last_phase=None) -> int:
        """Map current state to a tabular bucket."""
        counts = self._approach_counts()
        lp = last_phase if last_phase is not None else self._last_phase
        return self.discretizer.discretize(counts, lp)

    @property
    def last_phase(self):
        return self._last_phase

    @last_phase.setter
    def last_phase(self, value):
        self._last_phase = value

    def __repr__(self):
        return (
            f"TrafficRLEnv(profile={self.profile_key}, "
            f"tick={self.tick}/{self.episode_length}, "
            f"action_space={self.action_space})"
        )
