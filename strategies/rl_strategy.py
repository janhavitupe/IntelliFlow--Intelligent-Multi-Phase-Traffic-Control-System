"""
rl_strategy.py

RLStrategy - wraps a trained reinforcement-learning agent behind the standard
BaseStrategy interface so the scheduler/controller code is completely
untouched. At demo/inference time select_phase(state) = argmax_a Q(state, a)
is just a table lookup / forward pass - no training happens live.

This strategy works in two modes:

    1. Cooperative/training mode (used by TrafficRLEnv):
       The env calls set_pending(phase_type) with the agent's chosen action,
       and decide_next_phase() simply consumes it (returning None to HOLD the
       scheduler until an action is supplied). This keeps the training loop
       in full control of epsilon-greedy exploration.

    2. Self-driving/inference mode (used by the evaluation harness / demo):
       When an agent is attached AND no pending phase has been set, the
       strategy builds the observation from the intersection itself and
       returns argmax_a Q(obs, a). This lets RLStrategy run as a normal
       pluggable strategy inside the standard Simulation - no env injection
       needed - so three-way comparisons (FixedTimer / Density / RL) are
       apples-to-apples.

NOTE ON IMPORTS: This module imports env.state_builder lazily (inside reset
and _infer_phase) to avoid a circular-import cycle (env.traffic_env imports
RLStrategy from here, and importing env at module scope would recurse).

Emergency/ambulance handling is entirely rule-based in the scheduler and
never consults this strategy, so the agent never sees or acts during an
emergency window.
"""
from .base_strategy import BaseStrategy
from config import rl as rl_config
from config.phases import all_phase_types


class RLStrategy(BaseStrategy):
    """
    A strategy that wraps a trained RL agent (tabular Q or DQN).

    Attributes:
        name (str): strategy identifier.
        agent: trained agent exposing `select_action(obs/discrete_state)`.
        green_duration (float): green seconds granted to the chosen phase.
        pending_phase: the phase_type the agent chose for the next decision
                       (set by the env; consumed by decide_next_phase).
        decision_made (bool): True once decide_next_phase consumed a pending
                              phase (the env uses this to detect a decision
                              point was reached).
    """

    def __init__(self, agent=None, green_duration=None):
        super().__init__(name="rl")
        self.agent = agent
        self.green_duration = (
            green_duration if green_duration is not None else rl_config.GREEN_DURATION
        )
        self.pending_phase = None
        self.decision_made = False
        self._phase_plan = None
        self.obs_builder = None
        self.discretizer = None
        self._seen_count = 0

    # ------------------------------------------------------------------
    # Cooperative hooks used by the environment
    # ------------------------------------------------------------------

    def set_pending(self, phase_type):
        """Record the agent's chosen phase for the next decision point."""
        self.pending_phase = phase_type
        self.decision_made = False

    def reset(self, intersection=None):
        """Reset internal state (called by the env on reset)."""
        from env.state_builder import ObservationBuilder, Discretizer

        self.pending_phase = None
        self.decision_made = False
        self._phase_plan = None
        self.obs_builder = ObservationBuilder()
        self.discretizer = Discretizer()
        self._seen_count = 0

    # ------------------------------------------------------------------
    # BaseStrategy interface (called by the scheduler)
    # ------------------------------------------------------------------

    def decide_next_phase(self, intersection, current_phase, time):
        """
        Called by the scheduler each time it reaches a decision point.

        Order of precedence:
            1. If the env has set a pending phase, consume it and return it
               (the scheduler then activates it).
            2. Else if an agent is attached, self-drive: build the current
               observation and return argmax_a Q(obs, a).
            3. Otherwise return (None, None) so the scheduler HOLDS at the
               decision point until an action is supplied.

        Returns:
            (PhaseType|None, float|None): chosen phase + green duration.
        """
        # 1. Cooperative (env-injected) mode.
        if self.pending_phase is not None:
            phase = self.pending_phase
            self.pending_phase = None
            self.decision_made = True
            return phase, self.green_duration

        # 2. Self-driving inference mode (evaluation / demo).
        if self.agent is not None:
            phase = self._infer_phase(intersection, current_phase)
            if phase is not None:
                self.decision_made = True
                return phase, self.green_duration

        # 3. Hold.
        return None, None

    def _infer_phase(self, intersection, current_phase):
        """Compute argmax_a Q(obs, a) from the intersection's current state."""
        from env.state_builder import ObservationBuilder, Discretizer

        if self.obs_builder is None:
            self.obs_builder = ObservationBuilder()
            self.discretizer = Discretizer()

        # The observation's rank/elapsed use placeholder values for starvation
        # (0) and elapsed (0) since this standalone strategy has no density
        # reference; the learned policy mostly keys off queues/ranks + the
        # phase one-hot, so this is acceptable.
        active = current_phase.phase_type if current_phase is not None else None
        obs = self.obs_builder.build(intersection, active, 0.0)

        # DQN agents consume the raw observation vector.
        if hasattr(self.agent, "policy_net"):
            action = self.agent.select_action(obs)
            return self._phase_for_action(action)

        # Tabular agents consume a discretized bucket.
        if hasattr(self.agent, "Q"):
            counts = {
                name: intersection.get_approach(name).total_queue_length()
                for name in ("North", "South", "East", "West")
            }
            last_phase = current_phase.phase_type if current_phase is not None else None
            state = self.discretizer.discretize(counts, last_phase)
            action = self.agent.select_action(state)
            return self._phase_for_action(action)

        return None

    def _phase_for_action(self, action):
        """Map an action index to a PhaseType (cached phase list)."""
        if not hasattr(self, "_phases"):
            self._phases = all_phase_types()
        if 0 <= action < len(self._phases):
            return self._phases[action]
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
