"""
random_generator.py

A stochastic vehicle generator that randomly spawns vehicles across the
four approaches and three movement lanes. This is the default working
traffic source for the simulation.
"""
import random

from core.enums import MovementType, Priority, VehicleType
from core.vehicle import Vehicle
from .base_source import BaseTrafficSource


class RandomTrafficSource(BaseTrafficSource):
    """
    Generates random vehicle arrivals.

    Args:
        approaches (list[str]): approach names to spawn into.
        max_per_tick (int): max vehicles spawned per tick.
        emergency_probability (float): probability a spawned vehicle is HIGH.
    """

    def __init__(
        self,
        approaches=("North", "South", "East", "West"),
        max_per_tick=3,
        emergency_probability=0.02,
    ):
        self.approaches = list(approaches)
        self.max_per_tick = max_per_tick
        self.emergency_probability = emergency_probability
        self.movements = list(MovementType)

    def generate_spawns(self, time: float):
        """Return a list of (approach, movement, Vehicle) spawn tuples."""
        spawns = []
        count = random.randint(0, self.max_per_tick)
        for _ in range(count):
            approach = random.choice(self.approaches)
            movement = random.choice(self.movements)
            vehicle_type = self._random_vehicle_type()
            priority = self._random_priority()
            vehicle = Vehicle(
                vehicle_type=vehicle_type,
                arrival_time=time,
                destination_movement=movement,
                priority=priority,
            )
            spawns.append((approach, movement, vehicle))
        return spawns

    def _random_vehicle_type(self) -> VehicleType:
        # Weighted toward cars; weights align with the extended VehicleType
        # enum (CAR, BIKE, BUS, TRUCK, AMBULANCE).
        import config.simulation as sim_config

        mix = sim_config.VEHICLE_MIX
        weights = [mix.get(vt.name, 0) for vt in VehicleType]
        return random.choices(list(VehicleType), weights=weights, k=1)[0]

    def _random_priority(self) -> Priority:
        return (
            Priority.HIGH
            if random.random() < self.emergency_probability
            else Priority.NORMAL
        )

    def reset(self):
        # Stateless generator; nothing to reset.
        pass
