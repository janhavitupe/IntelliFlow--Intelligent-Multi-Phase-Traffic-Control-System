"""
emergency_strategy.py

PLACEHOLDER - Not yet implemented.

An ambulance-preemption strategy. When an emergency vehicle is detected,
this strategy will build an EMERGENCY_OVERRIDE phase for the ambulance's
approach and return it to the scheduler. The Phase abstraction already
supports this via config.phases.build_emergency_phase().
"""
from .base_strategy import BaseStrategy


class EmergencyStrategy(BaseStrategy):
    """
    Placeholder for ambulance preemption.

    TODO: Detect HIGH-priority vehicles and build an EMERGENCY_OVERRIDE
    phase for the approach they are arriving from.
    """

    def __init__(self):
        super().__init__(name="emergency")

    def decide_next_phase(self, intersection, current_phase, time):
        raise NotImplementedError("EmergencyStrategy is not implemented yet.")
