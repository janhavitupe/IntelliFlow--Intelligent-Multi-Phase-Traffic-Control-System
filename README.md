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
│   ├── density_strategy.py        # Percentile-based adaptive density strategy (Phase 3)
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
│   ├── phases.py                  # Phase plan definition (compatible movements)
│   ├── simulation.py              # Central timing / simulation parameters
│   ├── density.py                 # Phase 3 adaptive density configuration
│   └── traffic_profiles.py        # Time-dependent arrival-rate profiles
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

## Phase 3 - Percentile-Based Adaptive Density Control

The adaptive layer decides *which* phase to activate and *for how long*
based only on **approach-level density** (number of vehicles waiting on
North / South / East / West). It never reads a vehicle's intended
movement — the controller observation boundary is preserved.

### Configuration (`config/density.py`)

All adaptive parameters live in one module (no magic numbers):

- `MIN_GREEN_TIME` / `MAX_GREEN_TIME` — hard limits on any adaptive green
  duration.
- `DENSITY_LEVELS` — HIGH / MEDIUM / LOW classification bands (by rank).
- `FAIRNESS_*` — starvation protection (minimum green for LOW approaches,
  maximum consecutive HIGH services, starvation age threshold).
- `GREEN_EXTENSION_*` — continuous-discharge extension model (discharge
  rate, extension per remaining vehicle, service-time estimate).

### Percentile-Based Ranking

At the start of every scheduling decision the strategy:

1. Counts queued vehicles on each of the four approaches.
2. Ranks the approaches (deterministic tie-break: fixed scan order
   `North, South, East, West`).
3. Classifies density **relative to the current state** (not absolute
   thresholds): Rank 1 → HIGH, Rank 2 → MEDIUM, Rank 3 → MEDIUM,
   Rank 4 → LOW.

Because density is percentile-based, the same queue length can be HIGH at
night and LOW during rush hour — the classification adapts to the current
intersection state.

### Continuous Green-Time Allocation (interval merging)

Green time is **never** computed as `vehicles × seconds_per_vehicle`.
Instead it follows the interval-extension idea from LeetCode 495
(*Teemo Attacking*):

- A green phase is a **continuous discharge interval**.
- It starts from `MIN_GREEN_TIME`.
- While significant discharge continues (vehicles remain queued and
  arrive during the phase), the interval is **extended** — like merging
  overlapping poison intervals — not restarted or multiplied.
- The extension is computed from the **approach discharge rate** (how fast
  the selected phase's lanes can physically clear vehicles) plus a
  configured service-time estimate for remaining queued vehicles.
- The result is clamped to `[MIN_GREEN_TIME, MAX_GREEN_TIME]`.

This models vehicles moving simultaneously while the signal stays green,
instead of independent per-vehicle service slots.

### Fairness

A LOW-density approach can never starve:

- Every approach accumulates a **starvation age** (time since last served).
- If a LOW approach has waited longer than `FAIRNESS_STARVATION_AGE`, the
  strategy boosts its priority and guarantees it a minimum green
  (`FAIRNESS_MIN_GREEN`).
- A configurable `FAIRNESS_MAX_CONSECUTIVE_HIGH` limits how many times the
  highest-density approach can be served back-to-back before another
  approach gets a turn.

### DensityStrategy (`strategies/density_strategy.py`)

Responsibilities (single responsibility):

- observe approach counts
- compute percentile ranking
- classify density
- choose next phase (scores the 10 normal phases by how much density they
  clear)
- compute adaptive green duration
- enforce fairness

It must **not** (and does not):

- move / discharge vehicles
- manipulate queues
- control signals directly

The scheduler remains generic — it simply asks the strategy *which phase?*
and *how long?* and never learns how the decision was made.

### Selecting the strategy

```python
from simulation import Simulation

# FixedTimer (original behaviour)
sim = Simulation(strategy_key="fixed_timer")

# Adaptive density control
sim = Simulation(strategy_key="density", profile_key="RUSH_HOUR")
```

Emergency preemption is unchanged and always has higher priority than
adaptive scheduling:

```
NORMAL GREEN → YELLOW CLEARANCE → EMERGENCY GREEN → RESUME ADAPTIVE CONTROL
```

### Analytics & CSV

New adaptive metrics are recorded and logged:

- `approach_rankings` (per decision: rank 1..4 → approach)
- `density_classifications` (approach → HIGH/MEDIUM/LOW)
- `selected_phase` (the phase chosen by the adaptive layer)
- `adaptive_green_duration` (seconds assigned)
- `adaptive_green_by_phase` (accumulated adaptive green per phase)
- `fairness_activations` (count of fairness boosts)
- `priority_selections_by_approach` (how often each approach drove the
  decision)

### Validation

```bash
python _validate_density_strategy.py
```

12/12 checks pass:

1. Highest-ranked approach is selected first.
2. Ranking updates every scheduling cycle.
3. Equal densities behave deterministically.
4. LOW-density approaches never starve.
5. Green duration increases with sustained traffic flow.
6. Green duration stays within configured limits.
7. Emergency override interrupts adaptive control.
8. Adaptive control resumes after emergency.
9. Fixed seed produces deterministic results.
10. Existing Phase 2 validation suite still passes.
11. Structural check (no `destination_movement`, no queue mutation, no
    signal control).
12. Every normal phase reachable under at least one traffic scenario.

## Phase 4 - Reinforcement Learning Wrapper

Phase 4 wraps the existing simulator in a standard Gym-style interface
(`reset()` / `step(action)`) and adds reinforcement-learning agents that
learn which phase to grant green next. The scheduler/controller code is
completely untouched — the RL agent is just another pluggable `BaseStrategy`.

### Gym-style environment (`env/`)

`TrafficRLEnv` exposes the standard contract:

```
reset()              -> observation (23-dim float32)
step(action)         -> (next_obs, reward, done, info)
action_space         -> 10 (choose one of the 10 normal phases)
observation_space    -> (23,)
```

**Observation** reuses exactly what the Density strategy already computes
(no new features invented):

| indices | feature                                   |
|---------|-------------------------------------------|
| 0–3     | queue length per approach (normalized)    |
| 4–7     | percentile rank per approach (1..4)       |
| 8–11    | starvation counters per approach          |
| 12–21   | active phase one-hot (10 normal phases)   |
| 22      | elapsed seconds in the current phase      |

**Action** is "pick one of the 10 phases" at each **decision point**
(a minimum-green boundary), not every tick. The policy's chosen phase runs
through the same yellow/red transition machinery as any other strategy.
Green duration stays handled by the existing extension logic.

**Reward** (deliberately crude): `r = -sum(queue_lengths)` over the step —
a "minimize congestion" proxy sufficient to get a learning signal.

**Emergency** is fully rule-based and outside the learning loop. The
scheduler's preemption state machine runs internally and never consults the
RL strategy, so the agent never sees or acts during an emergency window.

### Agents (`rl/`)

- **Tabular Q-learning** (`rl/agents.py`): discretizes the state into
  queue LOW/MED/HIGH per approach folded with the last-active-phase
  (3⁴ × 10 = 810 states). Fast, interpretable, trains in seconds — the
  sanity baseline.
- **DQN** (`rl/dqn.py`): a hand-rolled pure-numpy MLP (2 hidden layers of
  64) over the raw 23-dim state, with a replay buffer and target network.
  No PyTorch dependency — the gradient updates are implemented manually.
  The backprop is validated independently on a tiny XOR problem
  (`_validate_mlp.py`) before it is wired into RL, per the phase-4 spec.

### Training (`rl/train.py`)

```
python plot_rewards.py [n_episodes]   # train tabular Q + plot reward curve
```

- Episode = a fixed-length window (`EPISODE_LENGTH` ticks) of a traffic
  profile.
- Profiles cycle across episodes (LIGHT / NORMAL / RUSH / NIGHT / CUSTOM)
  so the agent doesn't overfit to one pattern.
- A fixed per-episode seed makes evaluation reproducible.
- The per-episode cumulative reward curve is the key evidence that learning
  is working (rising curve = RL is improving).

### RLStrategy (`strategies/rl_strategy.py`)

At inference/demo time, `RLStrategy` wraps a trained agent and selects
`argmax_a Q(state, a)` — a table lookup (tabular) or one forward pass (DQN).
No training happens live. It runs as a normal pluggable strategy:

```python
from strategies.rl_strategy import RLStrategy
from rl.train import train_tabular

agent, rewards = train_tabular(n_episodes=100)
sim = Simulation(strategy=RLStrategy(agent=agent), profile_key="RUSH_HOUR")
```

### Evaluation (`evaluation/evaluate.py`)

Run FixedTimer, Density, and RL through the **same** profile + seed and
compare using the existing analytics KPIs (avg wait, throughput, max queue,
congestion ratio). This three-way comparison is the demo narrative:
*naïve baseline → engineered rule-based → learned*.

```
python -c "from evaluation.evaluate import *; from strategies.fixed_timer_strategy import *; from strategies.density_strategy import *; from strategies.rl_strategy import *; from rl.train import train_tabular; a,_=train_tabular(n_episodes=30,verbose=False); print_comparison(evaluate_strategies({'fixed_timer':FixedTimerStrategy(),'density':DensityStrategy(),'rl':RLStrategy(agent=a)}))"
```

> Honest result note: on a small single-intersection problem a well-tuned
> rule-based system (Density) often beats a thinly-trained RL agent, and
> that is a fine, honest narrative. The RL value is the learned policy +
> the framework, not necessarily guaranteeing RL > Density.

### Validation

```bash
python _validate_rl.py            # Stage 1: env + tabular Q (11 tests)
python _validate_rl_stage2.py     # Stage 2: DQN + eval harness (7 tests)
python _validate_mlp.py           # MLP backprop gradient check + XOR (3 tests)
```

All suites pass, including regression against the Phase 2/3 suites.

## Extending for Future Features

| Feature                    | Where it plugs in                                  |
|----------------------------|-----------------------------------------------------|
| YOLO / OpenCV detection    | `traffic_source/yolo_generator.py`                 |
| Ambulance preemption       | `strategies/emergency_strategy.py` + `EMERGENCY_OVERRIDE` phase |
| Density scheduling         | `strategies/density_strategy.py`                   |
| Queue Relaxation Algorithm | `strategies/queue_relaxation_strategy.py`          |
| RL phase selection         | `strategies/rl_strategy.py` + `env/` + `rl/`       |
| SUMO integration           | `traffic_source/sumo_generator.py`                 |
| React dashboard            | `analytics/statistics.py` (export KPIs)            |
| Database logging           | `analytics/statistics.py` (persist snapshot)       |
</content>

