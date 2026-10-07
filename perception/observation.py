"""
observation.py

The perception contract: a per-lane summary of what is waiting at the
intersection. Produced by the simulator today (perception.simulated) and by
a camera pipeline (detection + tracking) later; consumed by the controller.

Only quantities a camera can measure belong here: for each lane, the
vehicles it can see and how long each has been waiting in the queue.
"""
from dataclasses import dataclass

from core.enums import MovementType

APPROACH_ORDER = ("North", "South", "East", "West")
MOVEMENT_ORDER = (MovementType.LEFT, MovementType.STRAIGHT,
                  MovementType.RIGHT, MovementType.UTURN)
# Fixed lane order: North_LEFT, North_STRAIGHT, ..., West_UTURN (16 lanes).
LANE_ORDER = tuple(f"{a}_{m.name}" for a in APPROACH_ORDER for m in MOVEMENT_ORDER)


@dataclass(frozen=True)
class LaneObservation:
    """What is visible in one lane: the wait (s) of each queued vehicle."""
    movement_id: str
    approach: str
    waits: tuple = ()

    @property
    def count(self) -> int:
        return len(self.waits)

    @property
    def max_wait(self) -> float:
        return max(self.waits, default=0.0)

    @property
    def mean_wait(self) -> float:
        return sum(self.waits) / len(self.waits) if self.waits else 0.0

    def long_waiters(self, threshold: float) -> int:
        """How many vehicles have waited longer than `threshold` seconds."""
        return sum(1 for w in self.waits if w > threshold)


@dataclass(frozen=True)
class IntersectionObservation:
    """
    Per-lane observations in LANE_ORDER, at simulation time `time`.

    `available` is False when the sensor itself is down (e.g. a camera
    outage); the lanes then carry no information, which is NOT the same as
    "no vehicles".
    """
    time: float
    lanes: tuple
    available: bool = True

    def lane(self, movement_id: str) -> LaneObservation:
        return self.lanes[LANE_ORDER.index(movement_id)]

    def approach_count(self, approach: str) -> int:
        return sum(l.count for l in self.lanes if l.approach == approach)
