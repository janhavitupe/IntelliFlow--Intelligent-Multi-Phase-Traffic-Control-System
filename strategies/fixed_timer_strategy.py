"""
fixed_timer_strategy.py

A simple round-robin strategy that gives every phase a fixed green
duration. This is the initial, working scheduling strategy. It
demonstrates the Strategy interface that more advanced algorithms
(Density, Queue Relaxation, Emergency) will also implement.
"""
from config.phases import all_phase_types
from .base_strategy import BaseStrategy


class FixedTimerStrategy(BaseStrategy):
    """
    Cycles through the standard phases, each with a fixed green duration.

    Attributes:
        green_duration (float): fixed green time (ticks) per phase.
        yellow_duration (float): yellow transition time (ticks).
    """

    def __init__(self, green_duration: float = 12.0, yellow_duration: float = 2.0):
        super().__init__(name="fixed_timer")
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration

    def decide_next_phase(self, intersection, current_phase, time):
        """
        Choose the next phase in a fixed rotation.

        Returns:
            (phase_type, green_duration): the next phase to activate and
            its suggested green duration.
        """
        order = all_phase_types()

        if current_phase is None or current_phase.phase_type not in order:
            next_type = order[0]
        else:
            idx = order.index(current_phase.phase_type)
            next_type = order[(idx + 1) % len(order)]

        return next_type, self.green_duration
