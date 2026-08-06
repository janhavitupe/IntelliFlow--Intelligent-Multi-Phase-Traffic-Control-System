"""
service_model.py

A pluggable ServiceModel that determines how many vehicles can leave each
active movement during a simulation tick, based on vehicle type, service
time, and tick duration.

Why this exists:
The scheduler's only responsibility is selecting and activating phases.
Vehicle discharge behavior belongs to the simulation/traffic-flow layer.
By keeping service logic here, the scheduler stays completely reusable
when DensityStrategy, QueueRelaxationStrategy, and EmergencyStrategy are
introduced later.

How it works:
Each lane accumulates "elapsed green time". A head vehicle departs only
once enough green time has built up to satisfy its type's service time
(e.g. a Bike needs 0.6s, a Truck needs 2.2s). This produces realistic,
non-uniform discharge rates instead of "one vehicle every tick".
"""
from core.enums import VehicleType
from config import simulation as sim_config


class ServiceModel:
    """
    Converts accumulated green time into discharged vehicles per lane.

    Attributes:
        service_times (dict): VehicleType -> seconds to clear.
        _elapsed (dict): lane -> accumulated green seconds.
    """

    def __init__(self, service_times=None):
        # Resolve service times from config unless overridden.
        self.service_times = service_times or {
            VehicleType[vkey]: t
            for vkey, t in sim_config.SERVICE_TIMES.items()
        }
        # Per-lane accumulated green time budget.
        self._elapsed = {}

    # -------- Tick budget management --------

    def reset(self):
        """Clear all per-lane green-time budgets (start of a new run)."""
        self._elapsed = {}

    def accumulate(self, active_movements, delta: float):
        """
        Add `delta` seconds of green time to every active movement's lane.

        Args:
            active_movements (iterable): Movement objects currently green.
            delta (float): tick duration in seconds.
        """
        for movement in active_movements:
            lane = movement.lane
            key = id(lane)
            self._elapsed[key] = self._elapsed.get(key, 0.0) + delta

    def _service_seconds(self, vehicle) -> float:
        """Return how many seconds the head vehicle needs to clear."""
        vtype = vehicle.vehicle_type
        return self.service_times.get(vtype, 1.0)

    def _can_serve(self, movement) -> bool:
        """A movement can serve if green, non-empty, and budget suffices."""
        if not movement.can_serve():
            return False
        lane = movement.lane
        key = id(lane)
        remaining = self._elapsed.get(key, 0.0)
        head = lane.queue.peek()
        return remaining >= self._service_seconds(head)

    def discharge(self, active_movements, max_per_lane=None) -> list:
        """
        Serve as many vehicles as the accumulated green budget allows.

        For each active movement/lane, repeatedly check the head vehicle;
        if the lane's green budget covers its service time, dequeue it and
        subtract the service time from the budget. Continue until the
        budget runs out or the queue empties.

        Args:
            active_movements (iterable): movements currently green.
            max_per_lane (int|None): optional cap on discharges per lane.

        Returns:
            list of (movement, vehicle) tuples discharged this tick.
            The vehicle objects are returned so the analytics layer can
            record per-movement and per-vehicle-type statistics.
        """
        served = []
        for movement in active_movements:
            lane = movement.lane
            key = id(lane)
            lane_served = 0
            while lane_served != max_per_lane:
                if not self._can_serve(movement):
                    break
                head = lane.queue.peek()
                self._elapsed[key] -= self._service_seconds(head)
                lane.remove_front_vehicle()
                served.append((movement, head))
                lane_served += 1
        return served

    def __repr__(self) -> str:
        return f"ServiceModel(types={sorted(k.name for k in self.service_times)})"
