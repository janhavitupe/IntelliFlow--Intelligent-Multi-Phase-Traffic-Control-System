"""
statistics.py

The Analytics framework collects and aggregates traffic statistics.

It is designed to be extensible:
    - Later it can persist to a database.
    - Later it can feed a React dashboard.
    - Later it can capture ambulance-delay and congestion metrics.

It captures the core KPIs:
    - average waiting time
    - average queue length
    - vehicles passed (served)
    - throughput
    - congestion statistics
    - maximum queue observed per movement
    - total green time per phase
    - vehicles served per movement
    - vehicles served per vehicle type
    - queue growth rate
    - queue reduction rate
"""
from core.enums import Priority, VehicleType


class Statistics:
    """
    Aggregates intersection-level traffic statistics.

    Attributes:
        intersection (Intersection): the intersection being measured.
        interval (float): seconds per tick (for throughput conversion).
    """

    def __init__(self, intersection, interval: float = 1.0):
        self.intersection = intersection
        self.interval = interval

        # Cumulative counters.
        self.total_vehicles_spawned = 0
        self.total_vehicles_served = 0
        self.total_waiting_time = 0.0
        self._queue_samples = 0
        self._queue_length_sum = 0
        self._wait_sample_count = 0
        self._wait_sum = 0.0

        # Emergency tracking.
        self.total_emergency_vehicles = 0
        self.total_emergency_delay = 0.0  # future
        self.total_emergency_preemptions = 0
        self.emergency_preemptions_by_approach = {}

        # Congestion.
        self.congestion_ticks = 0

        # ---- Approved additions (Phase 2 realism) ----

        # Maximum queue observed per movement (movement_id -> int).
        self.max_queue_by_movement = {}

        # Total green time accumulated per phase (PhaseType.name -> seconds).
        self.green_time_by_phase = {}

        # Vehicles served per movement (movement_id -> int).
        self.served_by_movement = {}

        # Vehicles served per vehicle type (VehicleType.name -> int).
        self.served_by_type = {}

        # Queue growth / reduction rates (vehicles per tick, moving window).
        self._queue_history = []
        self._growth_sum = 0.0
        self._reduction_sum = 0.0
        self._growth_samples = 0
        self._reduction_samples = 0

    # -------- Sampling --------

    def sample(self):
        """
        Sample the current state of the intersection. Called once per tick.
        """
        total_q = self.intersection.total_queue_length()
        self._queue_samples += 1
        self._queue_length_sum += total_q

        # Waiting time sum across all lanes.
        wait_sum = 0.0
        for lane in self.intersection.all_lanes():
            wait_sum += lane.queue.total_waiting_time
        self._wait_sample_count += 1
        self._wait_sum += wait_sum

        # Congestion: total queue above a threshold.
        if total_q >= 10:
            self.congestion_ticks += 1

        # Maximum queue per lane/movement.
        for lane in self.intersection.all_lanes():
            # Track max queue keyed by movement_id (reuse lane->movement link).
            movement = self._movement_for_lane(lane)
            key = movement.movement_id
            prior = self.max_queue_by_movement.get(key, 0)
            self.max_queue_by_movement[key] = max(prior, lane.queue_length)

        # Queue growth / reduction rates between consecutive samples.
        if self._queue_history:
            prev_q = self._queue_history[-1]
            diff = total_q - prev_q
            if diff > 0:
                self._growth_sum += diff
                self._growth_samples += 1
            elif diff < 0:
                self._reduction_sum += abs(diff)
                self._reduction_samples += 1
        self._queue_history.append(total_q)

    def _movement_for_lane(self, lane):
        """Return the Movement object owning the given Lane."""
        for movement in self.intersection.all_movements():
            if movement.lane is lane:
                return movement
        # Fallback: build a synthetic id from the lane name.
        return type("Movement", (), {"movement_id": lane.lane_name.replace("-", ".")})()

    def record_spawn(self, count: int = 1):
        """Record newly spawned vehicles."""
        self.total_vehicles_spawned += count

    def record_served(self, served_records):
        """
        Record vehicles that crossed the intersection.

        Args:
            served_records (list): list of (movement, vehicle) tuples.
                Each record must be a (Movement, Vehicle) tuple.
        """
        for movement, vehicle in served_records:
            self.total_vehicles_served += 1
            mid = movement.movement_id
            self.served_by_movement[mid] = self.served_by_movement.get(mid, 0) + 1
            vkey = vehicle.vehicle_type.name
            self.served_by_type[vkey] = self.served_by_type.get(vkey, 0) + 1

    def record_green_time(self, phase_type, delta: float):
        """Accumulate green time for a phase."""
        if phase_type is None:
            return
        key = phase_type.name
        self.green_time_by_phase[key] = self.green_time_by_phase.get(key, 0.0) + delta

    def record_emergency(self, count: int = 1):
        """Record emergency (HIGH-priority) vehicles encountered."""
        self.total_emergency_vehicles += count

    def record_emergency_preemption(self, approach_name: str):
        """
        Record that an EMERGENCY_OVERRIDE preemption was activated for an
        approach.
        """
        self.total_emergency_preemptions += 1
        self.emergency_preemptions_by_approach[approach_name] = (
            self.emergency_preemptions_by_approach.get(approach_name, 0) + 1
        )

    # -------- Computed KPIs --------

    @property
    def average_queue_length(self) -> float:
        if self._queue_samples == 0:
            return 0.0
        return self._queue_length_sum / self._queue_samples

    @property
    def average_waiting_time(self) -> float:
        if self._wait_sample_count == 0:
            return 0.0
        return self._wait_sum / self._wait_sample_count

    @property
    def throughput(self) -> float:
        """Vehicles served per simulation second."""
        if self.intersection.time == 0:
            return 0.0
        return self.total_vehicles_served / self.intersection.time

    @property
    def congestion_ratio(self) -> float:
        """Fraction of ticks the intersection was congested."""
        if self._queue_samples == 0:
            return 0.0
        return self.congestion_ticks / self._queue_samples

    @property
    def queue_growth_rate(self) -> float:
        """Average vehicles added per tick (from positive queue deltas)."""
        if self._growth_samples == 0:
            return 0.0
        return self._growth_sum / self._growth_samples

    @property
    def queue_reduction_rate(self) -> float:
        """Average vehicles cleared per tick (from negative queue deltas)."""
        if self._reduction_samples == 0:
            return 0.0
        return self._reduction_sum / self._reduction_samples

    # -------- Summary --------

    def summary(self) -> dict:
        """Return a dict snapshot of all KPIs."""
        return {
            "time": self.intersection.time,
            "vehicles_spawned": self.total_vehicles_spawned,
            "vehicles_served": self.total_vehicles_served,
            "average_waiting_time": round(self.average_waiting_time, 2),
            "average_queue_length": round(self.average_queue_length, 2),
            "throughput": round(self.throughput, 2),
            "congestion_ratio": round(self.congestion_ratio, 3),
            "emergency_vehicles": self.total_emergency_vehicles,
            "emergency_preemptions": self.total_emergency_preemptions,
            "emergency_preemptions_by_approach": self.emergency_preemptions_by_approach,
            "queue_growth_rate": round(self.queue_growth_rate, 3),
            "queue_reduction_rate": round(self.queue_reduction_rate, 3),
            "max_queue_by_movement": self.max_queue_by_movement,
            "green_time_by_phase": self.green_time_by_phase,
            "served_by_movement": self.served_by_movement,
            "served_by_type": self.served_by_type,
        }

    def __repr__(self) -> str:
        return f"Statistics({self.summary()})"
