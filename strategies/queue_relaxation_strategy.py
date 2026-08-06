"""
queue_relaxation_strategy.py

PLACEHOLDER - Not yet implemented.

Home of the user's own Queue Relaxation scheduling algorithm. This
strategy will decide both WHICH phase to activate and for HOW LONG,
based on queue-pressure relaxation. Pluggable once implemented.
"""
from .base_strategy import BaseStrategy


class QueueRelaxationStrategy(BaseStrategy):
    """
    Placeholder for the Queue Relaxation Algorithm.

    TODO: Implement the queue-pressure relaxation scheduling logic.
    """

    def __init__(self):
        super().__init__(name="queue_relaxation")

    def decide_next_phase(self, intersection, current_phase, time):
        raise NotImplementedError("QueueRelaxationStrategy is not implemented yet.")
