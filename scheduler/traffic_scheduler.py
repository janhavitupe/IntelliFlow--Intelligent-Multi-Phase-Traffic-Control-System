"""
traffic_scheduler.py

The TrafficScheduler is responsible for scheduling traffic phases. It
never hardcodes movement logic and never manipulates lanes directly.
Instead it:

    - Holds the phase plan (dict of PhaseType -> Phase).
    - Uses a pluggable BaseStrategy to decide which Phase to activate next.
    - Manages the current phase and its green/yellow timing.
    - Discharges vehicles from the active phase's compatible movements.
    - Updates statistics hooks.

The scheduler depends only on the abstract strategy interface and the
Phase abstraction, honoring the Open/Closed and Dependency-Inversion
principles.
"""
from core.enums import PhaseType, SignalState
from config import phases as phase_config


class TrafficScheduler:
    """
    Phase scheduler for the intersection.

    Attributes:
        intersection (Intersection): the intersection being scheduled.
        strategy (BaseStrategy): pluggable scheduling strategy.
        phase_plan (dict): PhaseType -> Phase.
        current_phase (Phase|None): the currently active phase.
        green_remaining (float): remaining green time for current phase.
        yellow_duration (float): yellow transition time.
        _in_yellow (bool): whether we're in the yellow transition.
    """

    def __init__(self, intersection, strategy, yellow_duration: float = 2.0):
        self.intersection = intersection
        self.strategy = strategy
        self.yellow_duration = yellow_duration

        # Build the phase plan from configuration.
        self.phase_plan = phase_config.build_phase_plan(intersection)

        self.current_phase = None
        self.green_remaining = 0.0
        self._in_yellow = False

    # -------- Public scheduling API --------

    def select_next_phase(self):
        """
        Ask the strategy which phase to activate next.

        Returns:
            (PhaseType|None, float|None): the chosen phase type and its
            suggested green duration.
        """
        return self.strategy.decide_next_phase(
            self.intersection, self.current_phase, self.intersection.time
        )

    def switch_phase(self, phase_type: PhaseType, green_seconds: float = 12.0):
        """
        Deactivate the current phase and activate the requested one.

        This is the only place the scheduler touches signals - via the
        active Phase object, never individual lanes.
        """
        # Deactivate current phase (turn its movements red).
        if self.current_phase is not None:
            self.current_phase.deactivate()

        # Activate the new phase.
        self.current_phase = self.phase_plan[phase_type]
        self.current_phase.activate()
        self.green_remaining = green_seconds
        self._in_yellow = False

    def switch_to_emergency(self, approach_name: str, green_seconds: float = 15.0):
        """
        Activate an emergency override phase for the given approach.

        This is the ambulance-preemption hook. It builds (or reuses) an
        EMERGENCY_OVERRIDE phase dynamically for the ambulance's approach.
        """
        # Deactivate whatever is currently running.
        if self.current_phase is not None:
            self.current_phase.deactivate()

        emergency_phase = phase_config.build_emergency_phase(
            self.intersection, approach_name
        )
        self.phase_plan[PhaseType.EMERGENCY_OVERRIDE] = emergency_phase
        self.current_phase = emergency_phase
        self.current_phase.activate()
        self.green_remaining = green_seconds
        self._in_yellow = False

    def serve_current_phase(self, vehicles_per_tick: int = 1):
        """
        Discharge vehicles from the currently active phase.

        For each movement in the active phase that can serve, remove up to
        `vehicles_per_tick` vehicles from its lane queue. Vehicles cross
        the intersection and immediately leave the simulation.
        """
        if self.current_phase is None:
            return 0

        served = 0
        for movement in self.current_phase.movements:
            for _ in range(vehicles_per_tick):
                if not movement.can_serve():
                    break
                movement.lane.remove_front_vehicle()
                served += 1
        return served

    def update_statistics(self):
        """
        Hook for periodic statistics collection.

        Future analytics / DB logging will subscribe here. Currently a
        no-op placeholder that keeps the scheduler ready for integration.
        """
        # The Analytics layer can attach to this hook without modifying
        # the scheduler logic.
        pass

    # -------- Tick / timing --------

    def update(self, delta: float = 1.0):
        """
        Advance the scheduler by one tick. Handles the green->yellow->
        red->next-phase transition using the strategy's decision.
        """
        if self.current_phase is None:
            # No phase active yet - start one.
            next_type, green = self.select_next_phase()
            if next_type is not None:
                self.switch_phase(next_type, green or 12.0)
            return

        # Tick signal clocks for the active phase.
        for movement in self.current_phase.movements:
            movement.signal.update(delta)

        if self._in_yellow:
            # Yellow transition: after yellow_duration, switch to next phase.
            self.yellow_remaining -= delta
            if self.yellow_remaining <= 0:
                next_type, green = self.select_next_phase()
                if next_type is not None:
                    self.switch_phase(next_type, green or 12.0)
            return

        # In green: count down remaining green time.
        self.green_remaining -= delta
        if self.green_remaining <= 0:
            # Begin yellow transition.
            self.current_phase.start_yellow_transition()
            self._in_yellow = True
            self.yellow_remaining = self.yellow_duration

# -------- Introspection --------

    def active_movements(self):
        """
        Return the list of Movement objects currently active (green).

        Used by the ServiceModel to accumulate green time and discharge
        vehicles. Returns an empty list when no phase is active.
        """
        if self.current_phase is None:
            return []
        return self.current_phase.movements

    @property
    def active_phase_type(self) -> PhaseType:
        return self.current_phase.phase_type if self.current_phase else None

    @property
    def in_yellow(self) -> bool:
        """True while the scheduler is in the yellow transition period."""
        return self._in_yellow

    @property
    def phase_remaining(self) -> float:
        """
        Seconds remaining in the current phase.

        During green this is the remaining green time; during yellow it
        is the remaining yellow time. Returns 0.0 when no phase is active.
        """
        if self.current_phase is None:
            return 0.0
        if self._in_yellow:
            return getattr(self, "yellow_remaining", 0.0)
        return self.green_remaining

    def __repr__(self) -> str:
        return (
            f"TrafficScheduler(strategy={self.strategy.name}, "
            f"phase={self.active_phase_type}, green_left={self.green_remaining:.1f})"
        )
