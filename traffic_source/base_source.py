"""
base_source.py

Abstract interface for all vehicle traffic sources.

The scheduler and simulation depend only on this interface, so swapping
random generation for YOLO or SUMO later requires no changes to the
scheduling logic.
"""
from abc import ABC, abstractmethod


class BaseTrafficSource(ABC):
    """
    Interface for generating vehicle arrivals.

    Each source produces a list of spawn tuples:
        (approach_name, movement_type, vehicle)
    """

    @abstractmethod
    def generate_spawns(self, time: float):
        """
        Produce vehicle spawns for the current simulation tick.

        Returns:
            list of (approach, movement, Vehicle) tuples.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Reset any internal state (e.g. counters) for a new run."""
        raise NotImplementedError
