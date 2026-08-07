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

        # ---- Phase 3 adaptive density metrics ----
        #
        # The adaptive controller (DensityStrategy) publishes a decision
        # snapshot each scheduling cycle. Statistics records the latest
        # snapshot plus cumulative counters derived from it.
        self._last_adaptive_decision_id = None
        self.adaptive_decision_count = 0
        self.adaptive_rankings = {}        # rank -> approach (latest)
        self.adaptive_densities = {}       # approach -> HIGH/MEDIUM/LOW (latest)
        self.adaptive_selected_phase = None   # PhaseType.name (latest)
        self.adaptive_green_duration = 0.0    # seconds (latest)
        self.adaptive_green_by_phase = {}     # PhaseType.name -> seconds (latest)
        self.fairness_activations = 0
        self.priority_selections_by_approach = {}  # approach -> count

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

    # -------- Phase 3 adaptive density recording --------

    def record_adaptive_decision(self, decision: dict):
        """
        Record an adaptive scheduling decision produced by DensityStrategy.

        The decision dict is the strategy's `last_decision` snapshot:
            decision_id, time, approach_counts, rankings, densities, weights,
            fairness_active, selected_phase, green_duration, scores.

        Cumulative counters (fairness activations, priority selections per
        approach) are incremented ONLY ONCE per new decision id, so repeated
        calls within the same scheduling cycle do not double-count.
        """
        decision_id = decision.get("decision_id")
        if decision_id is None:
            return
        if decision_id == self._last_adaptive_decision_id:
            return
        self._last_adaptive_decision_id = decision_id
        self.adaptive_decision_count += 1

        # Latest snapshots (for CSV rows / summary).
        self.adaptive_rankings = dict(decision.get("rankings", {}))
        self.adaptive_densities = dict(decision.get("densities", {}))
        self.adaptive_selected_phase = decision.get("selected_phase")
        self.adaptive_green_duration = float(
            decision.get("green_duration", 0.0)
        )
        # Track the latest adaptive green per selected phase.
        if self.adaptive_selected_phase is not None:
            self.adaptive_green_by_phase[self.adaptive_selected_phase] = (
                self.adaptive_green_duration
            )

        # Fairness activations: count once per decision where fairness was
        # applied (a starved approach received a boost).
        if decision.get("fairness_active"):
            self.fairness_activations += 1

        # Priority selections per approach: increment the approaches served
        # by the selected phase. The strategy exposes served approaches for
        # the selected phase so the analytics layer stays decoupled from
        # phase/movement internals.
        served = decision.get("served_approaches", [])
        for name in served:
            self.priority_selections_by_approach[name] = (
                self.priority_selections_by_approach.get(name, 0) + 1
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
            # Phase 3 adaptive density metrics.
            "adaptive_decision_count": self.adaptive_decision_count,
            "adaptive_rankings": self.adaptive_rankings,
            "adaptive_densities": self.adaptive_densities,
            "adaptive_selected_phase": self.adaptive_selected_phase,
            "adaptive_green_duration": self.adaptive_green_duration,
            "adaptive_green_by_phase": self.adaptive_green_by_phase,
            "fairness_activations": self.fairness_activations,
            "priority_selections_by_approach": self.priority_selections_by_approach,
        }

    def __repr__(self) -> str:
        return f"Statistics({self.summary()})"
