"""
traffic_source package

Abstraction over where vehicle data originates. The scheduler never knows
whether vehicles come from a random generator, a traffic profile, YOLO
detection, or SUMO. Each source exposes a uniform interface to produce
vehicle spawns.
"""
from .base_source import BaseTrafficSource
from .random_generator import RandomTrafficSource
from .profile_traffic_source import ProfileTrafficSource

__all__ = ["BaseTrafficSource", "RandomTrafficSource", "ProfileTrafficSource"]
