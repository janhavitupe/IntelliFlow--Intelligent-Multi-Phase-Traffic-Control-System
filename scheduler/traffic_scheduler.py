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

Emergency preemption:
    The scheduler detects HIGH-priority (ambulance) vehicles on an approach
    and temporarily overrides normal scheduling with an EMERGENCY_OVERRIDE
    phase. The emergency controller only observes AMBULANCE PRESENCE and
    APPROACH - never a vehicle's intended movement. The required safe
    transition is:

        NORMAL GREEN -> YELLOW CLEARANCE -> RED -> EMERGENCY GREEN

    The emergency phase stays active until the ambulance has cleared the
    intersection (configurable fail-safe timeout as a guard). Afterward,
    normal 10-phase scheduling resumes unchanged.
"""
from core.enums import PhaseType, SignalState
from config import phases as phase_config
from config import simulation as sim_config

# Fixed, deterministic scan order for ambulance detection. The first approach
# in this order that holds an ambulance "wins" (first-detected-wins policy).
# This preserves a stable, deterministic outcome when multiple approaches have
# an ambulance simultaneously.
APPROACH_ORDER = ("North", "South", "East", "West")


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

    def __init__(
        self,
        intersection,
        strategy,
        yellow_duration: float = 2.0,
        emergency_yellow_duration: float = None,
        emergency_max_timeout: float = None,
    ):
        self.intersection = intersection
        self.strategy = strategy
        self.yellow_duration = yellow_duration

        # Emergency timing - resolved from config when not provided, so no
        # hardcoded values are scattered through the code.
        self.emergency_yellow_duration = (
            emergency_yellow_duration
            if emergency_yellow_duration is not None
            else sim_config.EMERGENCY_YELLOW_TIME
        )
        self.emergency_max_timeout = (
            emergency_max_timeout
            if emergency_max_timeout is not None
            else sim_config.EMERGENCY_MAX_TIMEOUT
        )

        # Build the phase plan from configuration.
        self.phase_plan = phase_config.build_phase_plan(intersection)

        self.current_phase = None
        self.green_remaining = 0.0
        self._in_yellow = False

        # Emergency preemption state.
        self._emergency_approach = None      # approach being served, or None
        self._emergency_clearance_remaining = 0.0
        self._emergency_active_remaining = 0.0
        self._emergency_callback = None      # optional on-activate hook

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

    # -------- Emergency detection (controller observation only) --------

    def _approach_has_emergency(self, approach_name: str) -> bool:
        """
        Return True if any lane of the given approach holds a HIGH-priority
        (ambulance) vehicle.

        The controller observes ONLY ambulance presence on the approach - it
        never inspects a vehicle's intended movement. This preserves the
        camera-based controller boundary.
        """
        approach = self.intersection.get_approach(approach_name)
        return any(
            lane.queue.has_emergency_vehicle() for lane in approach.lanes.values()
        )

    def _detect_emergency_approach(self):
        """
        Return the approach that should receive emergency priority, or None.

        Deterministic first-detected-wins policy: scan approaches in the fixed
        APPROACH_ORDER and return the first one holding an ambulance. When
        multiple approaches have ambulances simultaneously, only one wins, so
        conflicting approaches are never both green.
        """
        for approach_name in APPROACH_ORDER:
            if self._approach_has_emergency(approach_name):
                return approach_name
        return None

    def set_emergency_callback(self, callback):
        """
        Register an optional callable invoked each time an EMERGENCY_OVERRIDE
        phase becomes active. The callback receives the approach name. Used by
        the Simulation layer to record emergency activations in analytics.
        """
        self._emergency_callback = callback

    # -------- Emergency preemption state machine --------

    def _begin_emergency(self, approach_name: str):
        """
        Begin emergency preemption for the given approach.

        If a normal green phase is active, start a YELLOW CLEARANCE on its
        movements. If nothing is green (or the intersection is idle), proceed
        straight to the emergency override (already safe). This is the REQUIRED
        NORMAL GREEN -> YELLOW CLEARANCE -> RED -> EMERGENCY GREEN sequence.
        """
        self._emergency_approach = approach_name
        self._emergency_active_remaining = self.emergency_max_timeout

        if self.current_phase is not None and self.current_phase.phase_type != PhaseType.EMERGENCY_OVERRIDE:
            # Suspend normal scheduling and interrupt the active phase with a
            # short yellow clearance so vehicles can clear the intersection.
            self.current_phase.start_yellow_transition()
            self._in_yellow = True
            self.yellow_remaining = self.emergency_yellow_duration
            self._emergency_clearance_remaining = self.yellow_remaining
        else:
            # Nothing green / already transitioning: go straight to emergency.
            self._activate_emergency_green()

    def _activate_emergency_green(self):
        """
        Force all currently non-emergency movements to RED, then activate the
        EMERGENCY_OVERRIDE for the ambulance's approach. Reuses the existing,
        correct build_emergency_phase() which green-lights exactly the four
        movements of the approach. A callback fires so the Simulation layer
        can record the preemption in analytics.
        """
        # Red out any phase still active (guarantees no conflicting movement
        # survives into the emergency green).
        if self.current_phase is not None:
            self.current_phase.deactivate()

        emergency_phase = phase_config.build_emergency_phase(
            self.intersection, self._emergency_approach
        )
        self.phase_plan[PhaseType.EMERGENCY_OVERRIDE] = emergency_phase
        self.current_phase = emergency_phase
        self.current_phase.activate()
        self._in_yellow = False
        self.green_remaining = self.emergency_max_timeout
        self._emergency_active_remaining = self.emergency_max_timeout

        if self._emergency_callback is not None:
            self._emergency_callback(self._emergency_approach)

    def _advance_emergency(self, delta: float):
        """
        Advance the emergency preemption state machine by one tick.

        Two phases:
          1. Clearance: yellow movements count down; once elapsed, RED them and
             activate the emergency green.
          2. Active: hold the emergency green until the ambulance has cleared
             the approach (or the fail-safe timeout expires), then return to
             normal scheduling.
        """
        if self._in_yellow:
            # Clearance phase.
            self._emergency_clearance_remaining -= delta
            if self._emergency_clearance_remaining <= 0:
                # Clearance complete: all movements now RED, then emergency green.
                self._activate_emergency_green()
            return

        # Emergency green active.
        self.green_remaining -= delta

        # Exit condition: ambulance has cleared the approach (or fail-safe).
        if not self._approach_has_emergency(self._emergency_approach):
            self._end_emergency()
        elif self.green_remaining <= 0:
            # Fail-safe timeout reached although ambulance still present.
            self._end_emergency()

    def _end_emergency(self):
        """
        Terminate the emergency override and return to normal scheduling.

        The emergency phase is red out and current_phase is cleared. The next
        update() will resume the normal PHASE_1..PHASE_10 rotation via the
        strategy - the official 10-phase controller is never modified.
        """
        if self.current_phase is not None:
            self.current_phase.deactivate()
        self.current_phase = None
        self._emergency_approach = None
        self._in_yellow = False
        self.green_remaining = 0.0

    # -------- Tick / timing --------

    def update(self, delta: float = 1.0):
        """
        Advance the scheduler by one tick. Handles the green->yellow->
        red->next-phase transition using the strategy's decision.

        Emergency preemption takes precedence: if an emergency is active or an
        ambulance is detected, normal scheduling is suspended.
        """
        # 1. If an emergency is already in progress, advance its state machine.
        if self._emergency_approach is not None:
            self._advance_emergency(delta)
            return

        # 2. Otherwise, look for a new ambulance to preempt.
        emergency_approach = self._detect_emergency_approach()
        if emergency_approach is not None:
            self._begin_emergency(emergency_approach)
            return

        # 3. Normal scheduling (official PHASE_1..PHASE_10).
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
