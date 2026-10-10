"""
perception package

What the controller sees: a per-lane summary of queued vehicles and their
waits (see observation.py). Simulated sources live in simulated.py; a camera
pipeline will implement the same contract in the visual-input phase.
"""
from .observation import LANE_ORDER, IntersectionObservation, LaneObservation
from .simulated import (BlackoutPerception, GroundTruthPerception, HeadingIntentPerception,
                        NoisyPerception)

__all__ = ["LANE_ORDER", "IntersectionObservation", "LaneObservation",
           "BlackoutPerception", "GroundTruthPerception", "HeadingIntentPerception",
           "NoisyPerception"]
