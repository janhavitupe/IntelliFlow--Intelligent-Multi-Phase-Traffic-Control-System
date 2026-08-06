"""
movement.py

Movement is a first-class object representing one possible movement at
the intersection (e.g. North.Left). A Movement ties together an approach
direction, a movement type, the underlying Lane, and a Signal.

The Movement does NOT own its activation state - that lives in the Signal,
so the TrafficScheduler can manage state transitions cleanly.
"""
from .enums import MovementType
from .signal import Signal


class Movement:
    """
    A single directional movement at an intersection.

    Attributes:
        movement_id (str): unique key, e.g. 'North_Left'.
        approach: the approach name (North/South/East/West).
        movement_type (MovementType): LEFT / STRAIGHT / RIGHT.
        lane (Lane): the physical lane serving this movement.
        signal (Signal): the signal controlling this movement.
    """

    def __init__(self, approach, movement_type: MovementType, lane):
        self.approach = approach
        self.movement_type = movement_type
        self.movement_id = f"{approach}_{movement_type.name}"
        self.lane = lane
        self.signal = Signal(self.movement_id)

    # -------- Signal convenience --------

    @property
    def is_active(self) -> bool:
        return self.signal.is_green

    def activate(self):
        """Turn this movement's signal green."""
        self.signal.turn_green()

    def deactivate(self):
        """Turn this movement's signal red."""
        self.signal.turn_red()

    def start_yellow(self):
        """Transition this movement's signal to yellow."""
        self.signal.turn_yellow()

    # -------- Serving --------

    @property
    def queue_length(self) -> int:
        return self.lane.queue_length

    def can_serve(self) -> bool:
        """Whether vehicles can be discharged from this movement right now."""
        return self.signal.is_green and not self.lane.is_empty

    def __repr__(self) -> str:
        return f"Movement({self.movement_id}, {self.signal.state.name})"
