"""
simulation.py

The Simulation class orchestrates the entire system. It wires together
an Intersection, a TrafficScheduler, a TrafficSource, a ServiceModel, and
the Analytics framework, then runs the main loop. main.py simply creates
a Simulation and calls run().

Design notes:
- The scheduler is responsible only for selecting and activating phases.
- Vehicle discharge (how many leave) is handled by the ServiceModel, which
  is aware of vehicle-type service times. This keeps scheduling logic
  clean and reusable when smarter strategies are added.
- Traffic sources are pluggable (random, profile, YOLO, SUMO) through the
  BaseTrafficSource interface.
- A CsvLogger optionally records per-tick snapshots for later analysis,
  a database, or a React dashboard.
"""
import json
import time as _time

from core.enums import Priority
from core.intersection import Intersection
from scheduler.traffic_scheduler import TrafficScheduler
from strategies.fixed_timer_strategy import FixedTimerStrategy
from strategies.density_strategy import DensityStrategy
from traffic_source.profile_traffic_source import ProfileTrafficSource
from services.service_model import ServiceModel
from analytics.statistics import Statistics
from analytics.logger import CsvLogger
from config import simulation as sim_config


class Simulation:
    """
    Top-level orchestrator for the traffic simulation.

    Args:
        yellow_duration (float): yellow transition seconds.
        green_duration (float): default green seconds per phase.
        tick_interval (float): seconds per simulated tick.
        max_ticks (int|None): optional run-length limit.
        live (bool): if True, print progress and sleep per tick.
        profile_key (str): traffic profile for the ProfileTrafficSource.
        seed (int|None): RNG seed for reproducible runs.
        log_to_csv (bool): record per-tick snapshots via CsvLogger.
        strategy_key (str): "fixed_timer" (default) or "density" (Phase 3
            percentile-based adaptive density controller).
    """

    def __init__(
        self,
        yellow_duration: float = None,
        green_duration: float = None,
        tick_interval: float = None,
        max_ticks: int = None,
        live: bool = True,
        profile_key: str = None,
        seed: int = None,
        log_to_csv: bool = False,
        strategy_key: str = None,
    ):
        # Resolve defaults from central config module.
        self.tick_interval = tick_interval if tick_interval is not None else sim_config.TICK_DURATION
        self.max_ticks = max_ticks if max_ticks is not None else sim_config.SIMULATION_DURATION
        self.live = live
        seed = seed if seed is not None else sim_config.SEED
        profile_key = profile_key if profile_key is not None else sim_config.TRAFFIC_PROFILE
        strategy_key = strategy_key if strategy_key is not None else sim_config.STRATEGY

        # Core objects.
        self.intersection = Intersection()

        # Pluggable strategy. "fixed_timer" preserves the existing round-robin
        # controller; "density" opts into the adaptive Phase 3 scheduler.
        if strategy_key == "density":
            strategy = DensityStrategy()
        else:
            strategy = FixedTimerStrategy(
                green_duration=green_duration if green_duration is not None else sim_config.GREEN_TIME,
                yellow_duration=yellow_duration if yellow_duration is not None else sim_config.YELLOW_TIME,
            )
        self.strategy = strategy
        self.scheduler = TrafficScheduler(
            self.intersection, strategy,
            yellow_duration=yellow_duration if yellow_duration is not None else sim_config.YELLOW_TIME,
            emergency_yellow_duration=sim_config.EMERGENCY_YELLOW_TIME,
            emergency_max_timeout=sim_config.EMERGENCY_MAX_TIMEOUT,
        )

        # Physical discharge model.
        self.service_model = ServiceModel()

        # Pluggable traffic source.
        self.traffic_source = ProfileTrafficSource(
            profile_key=profile_key,
            seed=seed,
            tick_duration=self.tick_interval,
        )

        # Analytics + optional CSV logging.
        self.analytics = Statistics(self.intersection, interval=self.tick_interval)

        # Record each emergency preemption activation in analytics.
        self.scheduler.set_emergency_callback(self.analytics.record_emergency_preemption)

        self.logger = CsvLogger() if log_to_csv else None

        self.tick = 0

    # -------- Main loop --------

    def run(self):
        """Execute the main simulation loop."""
        try:
            while self.max_ticks is None or self.tick < self.max_ticks:
                self.step()
                if self.live:
                    self.render()
                    _time.sleep(self.tick_interval)
        except KeyboardInterrupt:
            print("\nSimulation stopped by user.")
        finally:
            self._finalize()

    def step(self):
        """Advance the simulation by a single tick."""
        t = self.intersection.time

        # 1. Generate new vehicles.
        spawns = self.traffic_source.generate_spawns(t)
        self.intersection.spawn_batch(spawns)
        self.analytics.record_spawn(len(spawns))

        # Emergency (HIGH-priority) hook for preemption.
        emergency = sum(1 for _, _, v in spawns if v.priority == Priority.HIGH)
        self.analytics.record_emergency(emergency)

        # 2. Advance scheduler (phase transitions + emergency preemption).
        self.scheduler.update(self.tick_interval)

        # 2b. Record the adaptive decision snapshot (Phase 3 analytics).
        #     The DensityStrategy publishes a `last_decision` dict each time
        #     it selects a phase. Statistics deduplicates by decision_id, so
        #     repeated calls within a cycle never double-count.
        if isinstance(self.strategy, DensityStrategy) and self.strategy.last_decision is not None:
            self.analytics.record_adaptive_decision(self.strategy.last_decision)

        # 3. Accumulate green time for active movements, then discharge.
        active = self.scheduler.active_movements()
        self.service_model.accumulate(active, self.tick_interval)
        served = self.service_model.discharge(active)
        self.analytics.record_served(served)

        # 3b. Accumulate green time per phase (for phase-level statistics).
        self.analytics.record_green_time(
            self.scheduler.active_phase_type, self.tick_interval
        )

        # 4. Advance waiting times for remaining queued vehicles.
        self.intersection.update_waiting_times(self.tick_interval)

        # 5. Advance total simulation time.
        self.intersection.advance_time(self.tick_interval)

        # 6. Sample analytics and optionally log.
        self.analytics.sample()
        if self.logger is not None:
            self.logger.write(self._current_row())

        self.tick += 1

    def _current_row(self) -> dict:
        """Build a per-tick snapshot row for the CSV logger."""
        phase = self.scheduler.active_phase_type
        active = self.scheduler.active_movements()

        # Lane-level detail: queue length and aggregate waiting time,
        # keyed by movement_id (e.g. "North.STRAIGHT") as compact JSON.
        lane_queues = {}
        lane_waits = {}
        for movement in self.intersection.all_movements():
            mid = movement.movement_id
            lane_queues[mid] = movement.lane.queue_length
            lane_waits[mid] = movement.lane.queue.total_waiting_time

        row = {
            "simulation_time": round(self.intersection.time, 2),
            "tick": self.tick,
            "active_phase": phase.name if phase else "None",
            "phase_remaining": round(self.scheduler.phase_remaining, 2),
            "vehicles_spawned": self.analytics.total_vehicles_spawned,
            "vehicles_served": self.analytics.total_vehicles_served,
            "total_queue": self.intersection.total_queue_length(),
            "average_wait": round(self.analytics.average_waiting_time, 2),
            "throughput": round(self.analytics.throughput, 3),
            "congestion_ratio": round(self.analytics.congestion_ratio, 3),
            "lane_queues_json": json.dumps(lane_queues),
            "lane_waits_json": json.dumps(
                {k: round(v, 2) for k, v in lane_waits.items()}
            ),
        }

        # Phase 3 adaptive metrics (populated when the density strategy is
        # active; empty/zero otherwise so fixed-timer rows remain valid).
        row["approach_rankings_json"] = json.dumps(
            self.analytics.adaptive_rankings
        )
        row["density_classifications_json"] = json.dumps(
            self.analytics.adaptive_densities
        )
        row["adaptive_selected_phase"] = self.analytics.adaptive_selected_phase or ""
        row["adaptive_green_duration"] = round(
            self.analytics.adaptive_green_duration, 2
        )
        row["fairness_activations"] = self.analytics.fairness_activations
        row["priority_selections_json"] = json.dumps(
            self.analytics.priority_selections_by_approach
        )
        return row

    # -------- Rendering --------

    def render(self):
        """Print the current simulation state to the console."""
        print("\n" + "=" * 50)
        print(f"SMART TRAFFIC SIMULATOR - Tick {self.tick}")
        print(f"Time: {self.intersection.time:.1f}s")
        print("=" * 50)

        for name, approach in self.intersection.approaches.items():
            parts = []
            for movement_type, movement in approach.movements.items():
                signal_char = {
                    "GREEN": "🟢",
                    "YELLOW": "🟡",
                    "RED": "🔴",
                }.get(movement.signal.state.name, "⚪")
                parts.append(
                    f"{movement.movement_type.name:<8} "
                    f"{movement.lane.queue_length:>2} "
                    f"{signal_char}"
                )
            print(f"{name:<6}| " + " | ".join(parts))

        phase = self.scheduler.active_phase_type
        print(f"\nActive Phase: {phase.name if phase else 'None'}")
        print(f"Green Remaining: {self.scheduler.green_remaining:.1f}s")
        print(f"Total Queued: {self.intersection.total_queue_length()}")

    def _finalize(self):
        """Close logging and print the final summary."""
        if self.logger is not None:
            self.logger.close()
        self.print_final_summary()

    def print_final_summary(self):
        """Print the analytics summary at the end of the run."""
        print("\n" + "=" * 50)
        print("SIMULATION COMPLETE")
        print("=" * 50)
        for key, value in self.analytics.summary().items():
            print(f"  {key:<24}: {value}")

    def __repr__(self) -> str:
        return (
            f"Simulation(tick={self.tick}, "
            f"phase={self.scheduler.active_phase_type}, "
            f"queued={self.intersection.total_queue_length()})"
        )

