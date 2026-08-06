"""
intersection.py

The Intersection is the central container that owns the four Approaches
(North/South/East/West), manages vehicle generation, vehicle movement
(discharge), and exposes statistics. It depends on the Phase abstraction
and never hardcodes phase logic.
"""
from .approach import Approach


class Intersection:
    """
    Models the physical intersection and its four incoming approaches.

    Attributes:
        approaches (dict): direction -> Approach.
        time (float): current simulation tick.
    """

    def __init__(self):
        self.approaches = {
            "North": Approach("North"),
            "South": Approach("South"),
            "East": Approach("East"),
            "West": Approach("West"),
        }
        self.time = 0.0

    # -------- Access --------

    def get_approach(self, name: str) -> Approach:
        return self.approaches[name]

    def all_movements(self):
        """Yield all 12 Movement objects across all approaches."""
        for approach in self.approaches.values():
            yield from approach.movements.values()

    def all_lanes(self):
        """Yield all 12 Lane objects across all approaches."""
        for approach in self.approaches.values():
            yield from approach.lanes.values()

    def total_queue_length(self) -> int:
        return sum(a.total_queue_length() for a in self.approaches.values())

    # -------- Vehicle generation --------

    def spawn_vehicle(self, approach_name: str, movement_type, vehicle):
        """
        Add an incoming vehicle to the appropriate approach lane.

        Args:
            approach_name (str): North/South/East/West.
            movement_type (MovementType): LEFT/STRAIGHT/RIGHT.
            vehicle (Vehicle): the vehicle to enqueue.
        """
        lane = self.get_approach(approach_name).get_lane(movement_type)
        lane.add_vehicle(vehicle)
        return vehicle

    def spawn_batch(self, spawns):
        """
        Spawn a batch of vehicles described as (approach, movement, vehicle)
        tuples. Returns the list of spawned vehicles.
        """
        spawned = []
        for approach_name, movement_type, vehicle in spawns:
            spawned.append(self.spawn_vehicle(approach_name, movement_type, vehicle))
        return spawned

    # -------- Simulation tick --------

    def update_waiting_times(self, delta=1.0):
        """Advance waiting time for all queued vehicles."""
        for lane in self.all_lanes():
            lane.update_waiting_time(delta)

    def advance_time(self, delta=1.0):
        self.time += delta

    def __repr__(self) -> str:
        return (
            f"Intersection(t={self.time}, total_q={self.total_queue_length()}, "
            f"approaches={list(self.approaches.keys())})"
        )
