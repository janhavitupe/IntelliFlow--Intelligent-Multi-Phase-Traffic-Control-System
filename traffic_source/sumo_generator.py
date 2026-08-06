"""
sumo_generator.py

PLACEHOLDER - Not yet implemented.

Future traffic source backed by SUMO (Simulation of Urban MObility).
Will subscribe to the SUMO TraCI API and mirror real vehicle flows into
the simulation using the same BaseTrafficSource interface.
"""
from .base_source import BaseTrafficSource


class SumoTrafficSource(BaseTrafficSource):
    """
    Placeholder for SUMO integration via TraCI.

    TODO: Connect to SUMO, subscribe to vehicle data, map SUMO vehicles
    onto intersection approaches, and emit Vehicle spawns.
    """

    def __init__(self):
        self.connection = None

    def generate_spawns(self, time: float):
        raise NotImplementedError("SumoTrafficSource is not implemented yet.")

    def reset(self):
        pass
