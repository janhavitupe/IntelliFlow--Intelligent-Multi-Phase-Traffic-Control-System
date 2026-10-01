"""
rl_strategy.py

RLStrategy - wraps a trained reinforcement-learning agent behind the standard
BaseStrategy interface so the scheduler/controller code is completely
untouched. At demo/inference time select_phase(state) = argmax_a Q(state, a)
is just a table lookup / forward pass - no training happens live.

Green-time control (extend-or-switch):
    The agent picks one of the 10 phases at every decision point:
      - at the start (and after an emergency): the phase to run,
      - each time the active green runs out: picking the CURRENT phase
        extends it by GREEN_EXTENSION seconds; picking another phase ends
        the current one (yellow) and starts the chosen one with MIN_GREEN.
    A phase never exceeds MAX_GREEN: "extend" at the cap advances to the
    next phase in order. These are the same bounds the Density controller
    uses, so the agent has the same control freedom.

This strategy works in two modes:

    1. Cooperative/training mode (used by TrafficRLEnv):
       When a decision is needed and no action is pending, the strategy
       records the observation, sets `awaiting_action` and tells the
       scheduler to HOLD. The env reads `last_obs`, lets the agent choose,
       calls set_pending(action) and re-runs the scheduler, which now
       consumes it. This keeps the training loop in full control of
       epsilon-greedy exploration.

    2. Self-driving/inference mode (used by the evaluation harness / demo):
       When an agent is attached, the strategy builds the observation from
       the intersection itself and returns argmax_a Q(obs, a). This lets
       RLStrategy run as a normal pluggable strategy inside the standard
       Simulation, so comparisons with FixedTimer / Density are
       apples-to-apples.

Both modes build the observation through the SAME observe() call, at the
SAME moment, so the features the agent is trained on are exactly the
features it acts on at inference. The strategy owns the one piece of
decision-time bookkeeping those features need: when the current phase's
green started (elapsed-in-phase).

NOTE ON IMPORTS: This module imports env.state_builder lazily (inside reset)
to avoid a circular-import cycle (env.traffic_env imports RLStrategy from
here, and importing env at module scope would recurse).

Emergency/ambulance handling is entirely rule-based in the scheduler and
never consults this strategy, so the agent never sees or acts during an
emergency window.
"""
from .base_strategy import BaseStrategy, HOLD
from config import rl as rl_config
from config.phases import all_phase_types

APPROACH_ORDER = ("North", "South", "East", "West")


class RLStrategy(BaseStrategy):
    """
    A strategy that wraps a trained RL agent (tabular Q or DQN).

    Attributes:
        name (str): strategy identifier.
        agent: trained agent exposing `select_action(obs/discrete_state)`.
        min_green / extension / max_green (float): green-time bounds.
        pending_action (int|None): action chosen by the env for the current
                                   decision point (consumed when asked).
        awaiting_action (bool): True while the scheduler is holding for an
                                action from the env (training mode).
        last_obs / last_state: observation (23-dim) and tabular bucket
                               computed at the most recent decision point.

    A strategy instance carries per-run bookkeeping, so use a fresh
    instance (or call reset()) for every simulation run.
    """

    def __init__(self, agent=None, min_green=None, extension=None, max_green=None):
        super().__init__(name="rl")
        self.agent = agent
        self.min_green = min_green if min_green is not None else rl_config.MIN_GREEN
        self.extension = extension if extension is not None else rl_config.GREEN_EXTENSION
        self.max_green = max_green if max_green is not None else rl_config.MAX_GREEN
        self._phases = all_phase_types()
        self.reset()

    # ------------------------------------------------------------------
    # Cooperative hooks used by the environment
    # ------------------------------------------------------------------

    def set_pending(self, action):
        """Record the agent's action (0..9) for the current decision point."""
        self.pending_action = action

    def reset(self, intersection=None):
        """Reset all per-run state (called by the env on reset)."""
        from env.state_builder import ObservationBuilder, Discretizer

        self.pending_action = None
        self.awaiting_action = False
        self.last_obs = None
        self.last_state = None
        self.obs_builder = ObservationBuilder()
        self.discretizer = Discretizer()
        self._phase_started_at = 0.0
        self._next_phase = None

    # ------------------------------------------------------------------
    # Decision-time features
    # ------------------------------------------------------------------

    def observe(self, intersection, current_phase, time):
        """
        Build (observation, tabular_state) for the current decision point.

        Also caches them in last_obs / last_state for the env to read.
        """
        active = current_phase.phase_type if current_phase is not None else None
        elapsed = self._elapsed(current_phase, time)
        self.last_obs = self.obs_builder.build(intersection, active, elapsed)
        self.last_state = self.discretizer.discretize(
            self._approach_counts(intersection), active, elapsed
        )
        return self.last_obs, self.last_state

    def _elapsed(self, current_phase, time):
        """Seconds since the current phase's green started (0 if none)."""
        return time - self._phase_started_at if current_phase is not None else 0.0

    @staticmethod
    def _approach_counts(intersection):
        return {
            name: intersection.get_approach(name).total_queue_length()
            for name in APPROACH_ORDER
        }

    def _get_action(self, intersection, current_phase, time):
        """
        Action for the current decision point: the env's pending action,
        else the attached agent's greedy choice, else HOLD (await the env).
        """
        if self.pending_action is not None:
            action = self.pending_action
            self.pending_action = None
            self.awaiting_action = False
            return action

        obs, state = self.observe(intersection, current_phase, time)
        if self.agent is not None:
            # DQN agents consume the raw observation, tabular ones the bucket.
            return self.agent.select_action(
                obs if hasattr(self.agent, "policy_net") else state
            )

        self.awaiting_action = True
        return HOLD

    # ------------------------------------------------------------------
    # BaseStrategy interface (called by the scheduler)
    # ------------------------------------------------------------------

    def decide_next_phase(self, intersection, current_phase, time):
        """
        Called when the scheduler needs a phase: at the start, at the end
        of a yellow, and after an emergency.

        Returns:
            (PhaseType|None, float|None): phase + its minimum green, or
            (None, None) to hold until the env supplies an action.
        """
        if self._next_phase is not None:
            # Already chosen when the previous green ended.
            phase, self._next_phase = self._next_phase, None
        else:
            action = self._get_action(intersection, current_phase, time)
            if action is HOLD:
                return None, None
            phase = self._phases[action]
        self._phase_started_at = time
        return phase, self.min_green

    def on_green_end(self, intersection, current_phase, time):
        """
        Called when the active green runs out: extend it or switch.

        Returns:
            float (extend by that many seconds), None (end the phase; the
            chosen next phase is started after the yellow) or HOLD.
        """
        action = self._get_action(intersection, current_phase, time)
        if action is HOLD:
            return HOLD

        chosen = self._phases[action]
        current = current_phase.phase_type
        if chosen == current:
            if self._elapsed(current_phase, time) + self.extension <= self.max_green:
                return self.extension
            # At the cap: advance to the next phase in order.
            chosen = self._phases[(self._phases.index(current) + 1) % len(self._phases)]
        self._next_phase = chosen
        return None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def uses_clean_table(self) -> bool:
        """True if this wraps a tabular agent (Q-table), False for DQN."""
        return hasattr(self.agent, "Q") if self.agent is not None else False

    def __repr__(self):
        kind = "tabular" if self.uses_clean_table else (
            "dqn" if self.agent is not None else "no-agent"
        )
        return f"RLStrategy({kind})"
