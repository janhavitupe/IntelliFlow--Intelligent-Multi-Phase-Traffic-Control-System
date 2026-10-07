"""
simulated.py

Perception sources backed by the simulator.

    GroundTruthPerception : exactly what is in every queue.
    NoisyPerception       : the same, degraded by typical camera errors
                            (missed and phantom detections, limited view,
                            imprecise and lost tracks), so controllers can be
                            trained and tested for robustness before real
                            cameras exist.

A future camera pipeline implements the same `observe(intersection)` ->
IntersectionObservation contract (for real cameras the argument would be the
latest frames rather than the simulator).
"""
import random

from config import perception as cfg
from .observation import (APPROACH_ORDER, MOVEMENT_ORDER, IntersectionObservation,
                          LaneObservation)


def _lanes(intersection):
    """Yield (movement_id, approach_name, lane) in LANE_ORDER."""
    for approach in APPROACH_ORDER:
        a = intersection.get_approach(approach)
        for mt in MOVEMENT_ORDER:
            yield f"{approach}_{mt.name}", approach, a.get_lane(mt)


class GroundTruthPerception:
    """Exact per-lane waits, head of the queue first."""

    def reset(self):
        pass

    def observe(self, intersection) -> IntersectionObservation:
        lanes = tuple(
            LaneObservation(mid, approach, tuple(v.waiting_time for v in lane.queue))
            for mid, approach, lane in _lanes(intersection)
        )
        return IntersectionObservation(intersection.time, lanes)


class NoisyPerception:
    """
    Ground truth degraded by simulated camera errors (see config/perception.py).

    Errors are drawn independently at every observation from a seeded RNG,
    so runs stay reproducible.
    """

    def __init__(self, seed=0, miss_rate=None, false_positive_rate=None,
                 view_limit=None, wait_error=None, track_loss_rate=None):
        self.seed = seed
        self.miss_rate = cfg.NOISY_MISS_RATE if miss_rate is None else miss_rate
        self.false_positive_rate = (cfg.NOISY_FALSE_POSITIVE_RATE
                                    if false_positive_rate is None else false_positive_rate)
        self.view_limit = cfg.NOISY_VIEW_LIMIT if view_limit is None else view_limit
        self.wait_error = cfg.NOISY_WAIT_ERROR if wait_error is None else wait_error
        self.track_loss_rate = (cfg.NOISY_TRACK_LOSS_RATE
                                if track_loss_rate is None else track_loss_rate)
        self.reset()

    def reset(self):
        self._rng = random.Random(self.seed)

    def _measured_wait(self, true_wait):
        if self._rng.random() < self.track_loss_rate:
            return true_wait * self._rng.random()          # track lost and re-acquired
        return max(0.0, true_wait * (1.0 + self._rng.gauss(0.0, self.wait_error)))

    def observe(self, intersection) -> IntersectionObservation:
        lanes = []
        for mid, approach, lane in _lanes(intersection):
            waits = []
            for i, vehicle in enumerate(lane.queue):
                if i >= self.view_limit:                     # beyond the camera's view
                    break
                if self._rng.random() < self.miss_rate:      # occluded / missed
                    continue
                waits.append(self._measured_wait(vehicle.waiting_time))
            if self._rng.random() < self.false_positive_rate:
                waits.append(0.0)                            # phantom detection
            lanes.append(LaneObservation(mid, approach, tuple(waits)))
        return IntersectionObservation(intersection.time, tuple(lanes))
