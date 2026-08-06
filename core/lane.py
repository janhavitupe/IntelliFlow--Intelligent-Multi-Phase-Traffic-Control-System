"""
lane.py

A Lane is a physical incoming carriageway lane that holds a Queue of
vehicles. Lane focuses on lane properties (name, movement, capacity);
all vehicle/queue management is delegated to the contained Queue object.
"""
from .enums import MovementType
from .queue import Queue


class Lane:
    """
    A single incoming lane associated with one approach and one movement.

    Attributes:
        lane_name (str): e.g. 'North-Straight'.
        movement (MovementType): LEFT / STRAIGHT / RIGHT.
        queue (Queue): the FIFO of Vehicle objects stopped at this lane.
    """

    def __init__(self, lane_name: str, movement: MovementType):
        self.lane_name = lane_name
        self.movement = movement
        self.queue = Queue()
        self.queue.bind_lane(self)

    # -------- Queue delegation (convenience) --------

    @property
    def queue_length(self) -> int:
        return self.queue.length

    @property
    def is_empty(self) -> bool:
        return self.queue.is_empty

    def add_vehicle(self, vehicle):
        """Enqueue a vehicle into this lane's queue."""
        self.queue.enqueue(vehicle)

    def remove_front_vehicle(self):
        """Dequeue and return the head vehicle (used when served)."""
        return self.queue.dequeue()

    def update_waiting_time(self, delta=1.0):
        """Advance waiting time for all queued vehicles."""
        self.queue.update_waiting_time(delta)

    def __repr__(self) -> str:
        return f"Lane({self.lane_name}, {self.movement.name}, q={self.queue.length})"
