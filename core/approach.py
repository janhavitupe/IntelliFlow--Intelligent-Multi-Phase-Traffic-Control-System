"""
approach.py

An Approach represents a single incoming carriageway direction
(North, South, East, West). Each Approach contains four Lanes
(Left, Straight, Right, UTurn) and exposes the four corresponding
Movement objects built from those lanes.
"""
from .enums import MovementType
from .lane import Lane
from .movement import Movement


class Approach:
    """
    One incoming approach at the intersection.

    Attributes:
        name (str): North / South / East / West.
        lanes (dict): movement -> Lane.
        movements (dict): movement -> Movement.
    """

    def __init__(self, name: str):
        self.name = name

        # Build the four lanes.
        self.lanes = {
            MovementType.LEFT: Lane(f"{name}-Left", MovementType.LEFT),
            MovementType.STRAIGHT: Lane(f"{name}-Straight", MovementType.STRAIGHT),
            MovementType.RIGHT: Lane(f"{name}-Right", MovementType.RIGHT),
            MovementType.UTURN: Lane(f"{name}-UTurn", MovementType.UTURN),
        }

        # Build the four movements on top of the lanes.
        self.movements = {
            mt: Movement(name, mt, lane)
            for mt, lane in self.lanes.items()
        }

    # -------- Query --------

    @property
    def left(self) -> Movement:
        return self.movements[MovementType.LEFT]

    @property
    def straight(self) -> Movement:
        return self.movements[MovementType.STRAIGHT]

    @property
    def right(self) -> Movement:
        return self.movements[MovementType.RIGHT]

    @property
    def uturn(self) -> Movement:
        return self.movements[MovementType.UTURN]

    def get_movement(self, movement_type: MovementType) -> Movement:
        return self.movements[movement_type]

    def get_lane(self, movement_type: MovementType) -> Lane:
        return self.lanes[movement_type]

    def total_queue_length(self) -> int:
        """Total vehicles queued across all four lanes of this approach."""
        return sum(lane.queue_length for lane in self.lanes.values())

    def __iter__(self):
        return iter(self.movements.values())

    def __repr__(self) -> str:
        return f"Approach({self.name}, total_q={self.total_queue_length()})"
