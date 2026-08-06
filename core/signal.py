"""
signal.py

A dedicated Signal object manages the RED, YELLOW, GREEN state for a
single Movement. Separating the signal from the Movement keeps Movement
focused on geometry/identity and centralizes state-transition logic,
which simplifies yellow transitions and emergency (preemption) handling.
"""
from .enums import SignalState


class Signal:
    """
    Traffic signal state machine for one movement lane.

    Supported transitions:
        GREEN  -> YELLOW -> RED
        RED    -> GREEN
        Any    -> RED (emergency override)
    """

    def __init__(self, movement_id: str):
        self.movement_id = movement_id
        self.state = SignalState.RED
        self.time_in_state = 0.0

    # ---------- State query ----------

    @property
    def is_green(self) -> bool:
        return self.state == SignalState.GREEN

    @property
    def is_yellow(self) -> bool:
        return self.state == SignalState.YELLOW

    @property
    def is_red(self) -> bool:
        return self.state == SignalState.RED

    # ---------- Transitions ----------

    def turn_green(self):
        """Advance the signal to GREEN."""
        self.state = SignalState.GREEN
        self.time_in_state = 0.0

    def turn_yellow(self):
        """Advance the signal to YELLOW (from GREEN)."""
        self.state = SignalState.YELLOW
        self.time_in_state = 0.0

    def turn_red(self):
        """Force the signal to RED (from any state)."""
        self.state = SignalState.RED
        self.time_in_state = 0.0

    def update(self, delta=1.0):
        """Tick the signal clock (used for analytics/yellow timing)."""
        self.time_in_state += delta

    def can_serve(self) -> bool:
        """Whether vehicles may be discharged from this movement."""
        return self.is_green

    def __repr__(self) -> str:
        return f"Signal({self.movement_id}, {self.state.name})"

