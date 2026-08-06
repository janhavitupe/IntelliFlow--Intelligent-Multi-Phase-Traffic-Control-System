"""
density_strategy.py

PLACEHOLDER - Not yet implemented.

A density-based scheduling strategy that will read queue lengths from the
intersection and allocate green time proportional to demand. This is the
future home for the density-calculation feature (potentially fed by YOLO
vehicle detection).

The interface below lets it plug into the scheduler once implemented.
"""
from .base_strategy import BaseStrategy


class DensityStrategy(BaseStrategy):
    """
    Placeholder for a density/load-based phase scheduler.

    TODO: Implement queue-length weighted phase selection and dynamic
    green-time allocation.
    """

    def __init__(self):
        super().__init__(name="density")

    def decide_next_phase(self, intersection, current_phase, time):
        # Not implemented yet - would inspect intersection queues and
        # return the most "loaded" phase with a proportional duration.
        raise NotImplementedError("DensityStrategy is not implemented yet.")
