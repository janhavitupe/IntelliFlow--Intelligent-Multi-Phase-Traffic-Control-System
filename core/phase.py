"""
phase.py

A Phase is a collection of compatible Movement objects that may be
served simultaneously. The TrafficScheduler only activates Phase objects;
each Phase internally knows which Movement objects are allowed and how
to turn them on/off (including yellow transitions).
"""
from .enums import PhaseType, SignalState


class Phase:
    """
    A set of compatible movements.

    Attributes:
        phase_type (PhaseType): enum identifying the phase.
        movements (list[Movement]): the compatible movements in this phase.
    """

    def __init__(self, phase_type: PhaseType, movements):
        self.phase_type = phase_type
        self.movements = list(movements)

    # -------- Query --------

    @property
    def name(self) -> str:
        return self.phase_type.name

    def contains(self, movement) -> bool:
        return movement in self.movements

    def total_queue_length(self) -> int:
        """Total queued vehicles across all movements in this phase."""
        return sum(m.queue_length for m in self.movements)

    def has_emergency_vehicle(self) -> bool:
        """True if any movement in the phase has a HIGH-priority vehicle."""
        return any(m.lane.queue.has_emergency_vehicle() for m in self.movements)

    # -------- Transitions --------

    def activate(self):
        """Turn all movements in this phase green."""
        for movement in self.movements:
            movement.activate()

    def start_yellow_transition(self):
        """Transition all active (green) movements to yellow."""
        for movement in self.movements:
            if movement.signal.is_green:
                movement.start_yellow()

    def deactivate(self):
        """Force all movements in this phase to red."""
        for movement in self.movements:
            movement.deactivate()

    def update_signals(self):
        """Tick each movement's signal clock (for analytics)."""
        for movement in self.movements:
            movement.signal.update()

    # -------- Serving --------

    def can_serve(self) -> bool:
        """Can at least one movement in this phase discharge a vehicle?"""
        return any(m.can_serve() for m in self.movements)

    def __repr__(self) -> str:
        ids = ", ".join(m.movement_id for m in self.movements)
        return f"Phase({self.phase_type.name}) [{ids}]"
