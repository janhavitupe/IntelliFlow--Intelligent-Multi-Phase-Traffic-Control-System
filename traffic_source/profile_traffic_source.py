"""
profile_traffic_source.py

A traffic source that generates realistic, asymmetric vehicle arrivals
using a configurable traffic profile (see config/traffic_profiles.py).

Each incoming movement has its own independent arrival rate, and each
spawned vehicle gets a type drawn from the profile's vehicle mix.

The source is fully deterministic when a random seed is provided, which
is essential for reproducible algorithm comparisons.
"""
import random

from core.enums import MovementType, Priority, VehicleType
from core.vehicle import Vehicle
from config import traffic_profiles as profiles
from config import simulation as sim_config
from .base_source import BaseTrafficSource


class ProfileTrafficSource(BaseTrafficSource):
    """
    Generates vehicle arrivals from a time-dependent traffic profile.

    Args:
        profile_key (str): one of the keys in config.traffic_profiles.PROFILES.
        seed (int|None): optional RNG seed for reproducibility.
        tick_duration (float): seconds per tick; used to convert per-second
            arrival rates into per-tick spawn probabilities.
    """

    def __init__(self, profile_key="NORMAL_TRAFFIC", seed=None,
                 tick_duration=0.5):
        self.profile_key = profile_key
        self.tick_duration = tick_duration
        self._rng = random.Random(seed)
        self._seed = seed

        # Precompute VehicleType enum list for weighted sampling.
        self._vehicle_types = list(VehicleType)

    # -------- RNG reset for reproducibility --------

    def reset(self):
        """Reset the internal RNG so a run reproduces identically."""
        self._rng = random.Random(self._seed)

    # -------- Vehicle type sampling --------

    def _mix_weights(self, mix: dict) -> list:
        """Convert a mix dict {name: weight} into a list aligned to VehicleType."""
        return [mix.get(vt.name, 0) for vt in self._vehicle_types]

    def _sample_vehicle_type(self, mix: dict) -> VehicleType:
        weights = self._mix_weights(mix)
        return self._rng.choices(self._vehicle_types, weights=weights, k=1)[0]

    # -------- Spawn generation --------

    def generate_spawns(self, time: float):
        """
        Produce spawns for the current tick based on the profile's active
        window and per-movement arrival rates.

        Returns:
            list of (approach, movement_type, Vehicle) tuples.
        """
        spec = profiles.profile_for_time(self.profile_key, time)
        rates = spec.get("rates", {})
        mix = spec.get("mix", profiles.MIX_DEFAULT)

        spawns = []
        for approach in ("North", "South", "East", "West"):
            for movement_type in MovementType:
                key = f"{approach}.{movement_type.name}"
                rate = rates.get(key, 0.0)
                # Convert vehicles/second to a per-tick Poisson-ish spawn.
                # Using a Bernoulli with p = rate * tick_duration gives a
                # simple, reproducible approximation.
                per_tick = rate * self.tick_duration
                if per_tick <= 0:
                    continue
                if self._rng.random() < per_tick:
                    vehicle = self._create_vehicle(time, movement_type, mix)
                    spawns.append((approach, movement_type, vehicle))
        return spawns

    def _create_vehicle(self, time: float, movement_type: MovementType,
                        mix: dict) -> Vehicle:
        """Build a Vehicle with a type sampled from the mix and priority."""
        vehicle_type = self._sample_vehicle_type(mix)
        priority = (
            Priority.HIGH
            if vehicle_type == VehicleType.AMBULANCE
            else Priority.NORMAL
        )
        return Vehicle(
            vehicle_type=vehicle_type,
            arrival_time=time,
            destination_movement=movement_type,
            priority=priority,
        )

    def __repr__(self) -> str:
        return f"ProfileTrafficSource(profile={self.profile_key}, seed={self._seed})"
