"""
vehicle.py

Represents a single vehicle in the simulation as a first-class object.
Every vehicle stores its own identity, type, timing, queue reference,
destination movement, and priority.
"""
from .enums import MovementType, Priority, VehicleType


class Vehicle:
    """
    A single vehicle queued on an incoming approach.

    Attributes:
        vehicle_id (int): Unique identifier.
        vehicle_type (VehicleType): Category of the vehicle.
        arrival_time (float): Simulation time (tick) the vehicle arrived.
        current_lane: The Lane object the vehicle is currently queued in.
        waiting_time (float): Total time the vehicle has waited.
        destination_movement (MovementType): LEFT / STRAIGHT / RIGHT.
        priority (Priority): NORMAL or HIGH (ambulance/emergency).
    """

    _id_counter = 0

    def __init__(
        self,
        vehicle_type=VehicleType.CAR,
        arrival_time=0.0,
        destination_movement=MovementType.STRAIGHT,
        priority=Priority.NORMAL,
    ):
        Vehicle._id_counter += 1
        self.vehicle_id = Vehicle._id_counter
        self.vehicle_type = vehicle_type
        self.arrival_time = arrival_time
        self.current_lane = None
        self.waiting_time = 0.0
        self.destination_movement = destination_movement
        self.priority = priority

    @property
    def is_emergency(self) -> bool:
        """Convenience flag for ambulance preemption checks."""
        return self.priority == Priority.HIGH

    def update_waiting_time(self, delta=1.0):
        """Increment the waiting time by the given tick delta."""
        self.waiting_time += delta

    def __repr__(self) -> str:
        return (
            f"Vehicle(id={self.vehicle_id}, "
            f"type={self.vehicle_type.name}, "
            f"movement={self.destination_movement.name}, "
            f"priority={self.priority.name})"
        )
