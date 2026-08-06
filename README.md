# Smart Traffic Management System

An AI-powered adaptive traffic management system simulator built with a
scalable, object-oriented architecture. The current foundation models a
real intersection with incoming approaches, movement lanes, compatible
signal phases, and a pluggable scheduling strategy - ready to be extended
with YOLO detection, OpenCV, ambulance preemption, the Queue Relaxation
algorithm, SUMO, a React dashboard, and database logging.

## Architecture

```
d:/traffic/
├── main.py                        # Entry point (creates Simulation, calls run())
├── simulation.py                  # Simulation orchestrator (main loop)
├── requirements.txt
├── README.md
├── TODO.md
│
├── core/                          # Fundamental domain objects
│   ├── enums.py                   # PhaseType, MovementType, VehicleType, Priority, SignalState
│   ├── vehicle.py                 # Vehicle object (id, type, lane, movement, priority)
│   ├── queue.py                   # Queue of vehicles (FIFO + waiting stats)
│   ├── signal.py                  # Signal state machine (RED/YELLOW/GREEN)
│   ├── lane.py                    # Lane (holds a Queue)
│   ├── movement.py                # Movement - first-class object
│   ├── approach.py                # Approach (North/South/East/West = 3 lanes + 3 movements)
│   ├── phase.py                   # Phase = collection of compatible Movements
│   └── intersection.py            # Intersection (4 approaches, generation, movement, stats)
│
├── scheduler/
│   └── traffic_scheduler.py       # Schedules phases; activates Phase objects only
│
├── strategies/                    # Strategy Design Pattern
│   ├── base_strategy.py           # Abstract strategy interface
│   ├── fixed_timer_strategy.py    # Working round-robin strategy
│   ├── density_strategy.py        # Placeholder (future)
│   ├── queue_relaxation_strategy.py  # Placeholder (future - your algorithm)
│   └── emergency_strategy.py      # Placeholder (future ambulance preemption)
│
├── traffic_source/                # Vehicle source abstraction
│   ├── base_source.py             # Abstract traffic source interface
│   ├── random_generator.py        # Working random generator
│   ├── yolo_generator.py          # Placeholder (future YOLO/OpenCV)
│   └── sumo_generator.py          # Placeholder (future SUMO)
│
├── analytics/
│   ├── __init__.py
│   └── statistics.py              # KPI collection (waiting, queue, throughput, congestion)
│
├── config/
│   ├── __init__.py
│   └── phases.py                  # Phase plan definition (compatible movements)
│
└── images/                        # Reference intersection diagrams
```

## Core Concepts

### Movement (first-class object)
Each of the 12 intersection movements (North.Left, North.Straight,
South.Left, ..., West.Right) is an independent object. A Movement ties
together an approach, a movement type, its Lane, and its Signal.

### Phase (collection of compatible movements)
A Phase is simply a set of compatible Movement objects that may be served
simultaneously. The scheduler never hardcodes movement logic - it only
activates a Phase, and the Phase knows which movements are allowed.

```python
# The four normal phases map one-to-one to the official movement diagrams
# (images/3-4 .. images/6-4). Each wraps exactly the compatible movements
# shown in its diagram. The scheduler only ever activates these predefined
# phases; it never invents new movement combinations.
PhaseType.PHASE_1  # images/3-4 : North/South straight + left
PhaseType.PHASE_2  # images/4-4 : North/South straight + left, East left
PhaseType.PHASE_3  # images/5-4 : East/West straight + left, South left
PhaseType.PHASE_4  # images/6-4 : North all, South right, East left
PhaseType.EMERGENCY_OVERRIDE = {dynamically built from ambulance approach}
```

The exact movements for each phase are defined in `config/phases.py` via
`build_phase_plan()`. An emergency `EMERGENCY_OVERRIDE` phase is built at
runtime by `build_emergency_phase()` and does not modify the approved
normal phase definitions.

### TrafficScheduler
The scheduler schedules phases. It depends only on the abstract
`BaseStrategy` interface and the `Phase` abstraction. It never touches
individual lanes directly.

### Strategy Design Pattern
Different scheduling algorithms (FixedTimer, Density, Queue Relaxation,
Emergency) implement the same `BaseStrategy` interface. The scheduler can
swap strategies without modification.

## Running the Simulation

```bash
python main.py
```

The simulation runs for 100 ticks (or indefinitely with `max_ticks=None`).
Press Ctrl+C to stop an indefinite run.

## Phase 2 - Simulation Realism

The simulator now models realistic traffic behavior:

- **Pluggable ServiceModel** (`services/service_model.py`): vehicle discharge
  is governed by per-type service times (Bike 0.6s, Car 1.0s, Bus 1.8s,
  Truck 2.2s). Each lane accumulates green time; a vehicle leaves only once
  enough green time has built up to satisfy its service time. The scheduler
  remains completely unaware of service rates.
- **Time-dependent traffic profiles** (`config/traffic_profiles.py`): each
  profile is a schedule of windows, each with independent per-movement
  arrival rates and a vehicle mix. `CUSTOM` alternates Morning -> Rush ->
  Normal -> Evening. New schedules (e.g. `[(0,30,LIGHT),(30,60,RUSH_HOUR)]`)
  can be added without simulator code changes.
- **Reproducible seeds**: `Simulation(seed=42)` reproduces identical arrivals,
  vehicle types, and emergency generation across runs.
- **Extended statistics** (`analytics/statistics.py`): adds max queue per
  movement, total green time per phase, vehicles served per movement and per
  vehicle type, and queue growth/reduction rates.
- **Per-tick CSV logging** (`analytics/logger.py`): one row per tick with
  `simulation_time, tick, active_phase, phase_remaining, vehicles_spawned,
  vehicles_served, total_queue, average_wait, throughput, congestion_ratio,
  lane_queues_json, lane_waits_json`. Lane detail is compact JSON for
  pandas-based analysis.

### Validation

A validation suite confirms realistic behavior:

```bash
python _validate_simulations.py
```

It checks 8 scenarios (all pass): light traffic queues near zero, rush hour
queues grow, night spawns far fewer vehicles, 100% trucks lower throughput
than 100% bikes, no arrivals leaves the intersection empty, a burst drains
fully, only-North traffic grows only North queues, and a fixed seed is
reproducible.

> **Architecture note**: The preserved 4-phase plan (images/3-4..6-4) never
> serves the East.RIGHT and West.RIGHT movements. Under the FixedTimer those
> two lanes can never discharge. This is an inherent property of the approved
> phase architecture (not a Phase 2 defect) and is isolated in validation by
> using profiles that spawn only onto served movements.

## Extending for Future Features

| Feature                    | Where it plugs in                                  |
|----------------------------|-----------------------------------------------------|
| YOLO / OpenCV detection    | `traffic_source/yolo_generator.py`                 |
| Ambulance preemption       | `strategies/emergency_strategy.py` + `EMERGENCY_OVERRIDE` phase |
| Density scheduling         | `strategies/density_strategy.py`                   |
| Queue Relaxation Algorithm | `strategies/queue_relaxation_strategy.py`          |
| SUMO integration           | `traffic_source/sumo_generator.py`                 |
| React dashboard            | `analytics/statistics.py` (export KPIs)            |
| Database logging           | `analytics/statistics.py` (persist snapshot)       |
</content>

