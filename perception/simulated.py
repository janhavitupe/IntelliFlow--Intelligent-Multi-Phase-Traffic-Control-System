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


class BlackoutPerception:
    """
    Wraps another perception source and simulates a sensor outage between
    `start` and `end` seconds: no detections, available=False.
    """

    def __init__(self, base=None, start=0.0, end=float("inf")):
        self.base = base if base is not None else GroundTruthPerception()
        self.start, self.end = start, end

    def reset(self):
        self.base.reset()

    def observe(self, intersection) -> IntersectionObservation:
        if self.start <= intersection.time < self.end:
            lanes = tuple(LaneObservation(mid, approach, ())
                          for mid, approach, _ in _lanes(intersection))
            return IntersectionObservation(intersection.time, lanes, available=False)
        return self.base.observe(intersection)


class HeadingIntentPerception:
    """
    Turn intent read from each vehicle's heading, as a camera would.

    Drivers who will turn angle their vehicle toward the exit only near the
    stop line. So for each turning lane, the first `visible_depth` vehicles
    are reported in their true turning lane (except a `misread_rate` share,
    read as going straight), while turning vehicles further back still face
    straight and are reported as STRAIGHT demand on the same approach. Waits
    are kept; nothing is added or lost.

    Wraps any other source (ground truth by default, or NoisyPerception), so
    detection errors and intent errors can be combined.
    """

    def __init__(self, base=None, seed=0, visible_depth=None, misread_rate=None):
        self.base = base if base is not None else GroundTruthPerception()
        self.seed = seed
        self.visible_depth = cfg.INTENT_VISIBLE_DEPTH if visible_depth is None else visible_depth
        self.misread_rate = cfg.INTENT_MISREAD_RATE if misread_rate is None else misread_rate
        self.reset()

    def reset(self):
        self.base.reset()
        self._rng = random.Random(self.seed)

    def observe(self, intersection) -> IntersectionObservation:
        obs = self.base.observe(intersection)
        if not obs.available:
            return obs
        kept = {lane.movement_id: [] for lane in obs.lanes}
        moved = {}                                         # approach -> waits read as straight
        for lane in obs.lanes:
            turning = not lane.movement_id.endswith("_STRAIGHT")
            for position, wait in enumerate(lane.waits):
                read_as_turning = (not turning) or (
                    position < self.visible_depth and self._rng.random() >= self.misread_rate)
                if read_as_turning:
                    kept[lane.movement_id].append(wait)
                else:
                    moved.setdefault(lane.approach, []).append(wait)
        lanes = []
        for lane in obs.lanes:
            waits = kept[lane.movement_id]
            if lane.movement_id.endswith("_STRAIGHT") and lane.approach in moved:
                waits = sorted(waits + moved[lane.approach], reverse=True)   # longest wait first
            lanes.append(LaneObservation(lane.movement_id, lane.approach, tuple(waits)))
        return IntersectionObservation(obs.time, tuple(lanes), available=obs.available)


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
