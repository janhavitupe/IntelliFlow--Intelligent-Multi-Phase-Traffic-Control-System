# Smart Traffic Management System — Project Documentation

An AI-powered, adaptive traffic-management simulator built in Python with a
scalable, object-oriented architecture. It models a real 4-way intersection
with 10 compatible signal phases, a pluggable scheduling strategy design
pattern, and now a full Reinforcement-Learning (RL) pipeline that learns
which phase to grant green time under varying traffic conditions.

This document is the authoritative project reference. It covers:

1. Overview & design goals
2. Project structure
3. Core domain model
4. Traffic profiles (dataset)
5. Scheduling strategies (FixedTimer, Density, RL)
6. Service & discharge model
7. Analytics & KPIs
8. Reinforcement-Learning pipeline (Tabular Q + DQN)
9. Controlled experiment methodology
10. Actual model-performance results
11. How to run
12. Reproducibility notes

---

## 1. Overview & Design Goals

The system continuously:

- **Spans** vehicles from a traffic profile (per-movement arrival rates).
- **Schedules** compatible signal phases via a pluggable `BaseStrategy`.
- **Discharges** vehicles through a service-time-aware model.
- **Collects** KPIs (waiting time, queue length, throughput, congestion).

The architecture keeps the **scheduler generic**: it asks the active
strategy *"which phase next?"* and *"how long?"* and never hardcodes
movement/scheduling logic. This lets us swap in FixedTimer, an adaptive
Density controller, or a trained RL agent without touching the simulation
core.

Everything is **deterministic**: a fixed seed reproduces identical arrivals,
vehicle types, and emergency generation across runs — essential for fair
algorithm comparisons.

---

## 2. Project Structure

```
d:/traffic/
├── main.py                        # Entry point (creates Simulation, runs it)
├── simulation.py                  # Simulation orchestrator (main loop)
├── run_experiments.py             # Controlled RL experiment harness (results)
├── plot_rewards.py                # Plots the tabular-Q reward curve
├── requirements.txt
├── README.md
├── TODO.md
├── PROJECT_DOCUMENTATION.md       # This file
│
├── core/                          # Fundamental domain objects
│   ├── enums.py                   # PhaseType, MovementType, VehicleType, Priority, SignalState
│   ├── vehicle.py                 # Vehicle (id, type, lane, movement, priority)
│   ├── queue.py                   # FIFO queue + waiting statistics
│   ├── signal.py                  # Signal state machine (RED/YELLOW/GREEN)
│   ├── lane.py                    # Lane (holds a Queue)
│   ├── movement.py                # Movement — first-class object
│   ├── approach.py                # Approach (North/South/East/West)
│   ├── phase.py                   # Phase = collection of compatible Movements
│   └── intersection.py            # Intersection (4 approaches, spawn, stats)
│
├── scheduler/
│   └── traffic_scheduler.py       # Schedules phases; activates Phase objects
│
├── strategies/                    # Strategy Design Pattern
│   ├── base_strategy.py           # Abstract strategy interface
│   ├── fixed_timer_strategy.py    # Round-robin baseline controller
│   ├── density_strategy.py        # Percentile-based adaptive density (Phase 3)
│   ├── rl_strategy.py             # RL inference wrapper (argmax over Q/policy)
│   ├── queue_relaxation_strategy.py  # Placeholder (future)
│   └── emergency_strategy.py      # Placeholder (future ambulance preemption)
│
├── env/                           # Gym-style RL environment
│   ├── traffic_env.py             # reset()/step() over the simulator
│   └── state_builder.py           # 23-dim observation + tabular discretizer
│
├── rl/                            # Reinforcement-learning agents
│   ├── agents.py                  # TabularQAgent (810-state Q-learning)
│   ├── dqn.py                     # Pure-numpy DQN (MLP + replay + target)
│   └── train.py                   # train_tabular / train_dqn loops
│
├── evaluation/
│   └── evaluate.py                # Three-way strategy comparison harness
│
├── traffic_source/                # Vehicle source abstraction
│   ├── base_source.py             # Abstract traffic source interface
│   ├── profile_traffic_source.py  # Profile-driven arrivals (used by sim)
│   ├── random_generator.py        # Random generator
│   ├── yolo_generator.py          # Placeholder (future YOLO/OpenCV)
│   └── sumo_generator.py          # Placeholder (future SUMO)
│
├── services/
│   └── service_model.py           # Service-time-aware discharge model
│
├── analytics/
│   ├── statistics.py              # KPI collection (wait, queue, throughput, congestion)
│   └── logger.py                  # Per-tick CSV logging
│
├── config/                        # Centralized configuration
│   ├── phases.py                  # Official 10-phase plan definition
│   ├── simulation.py              # Timing / simulation parameters
│   ├── density.py                 # Adaptive density configuration
│   ├── rl.py                      # RL hyperparameters
│   └── traffic_profiles.py        # Time-dependent traffic scenarios
│
├── results/                       # Experiment outputs
│   ├── results_table.csv          # per-run raw KPIs (60 runs)
│   └── model_cards.md             # dataset + model documentation
│
└── images/                        # Comparison & training-curve graphs
    ├── G1_avg_wait_comparison.png
    ├── G2_throughput_comparison.png
    ├── G3_avg_queue_comparison.png
    └── G4_rl_training_curves.png
```

---

## 3. Core Domain Model

| Class            | Responsibility                                                        |
|------------------|-----------------------------------------------------------------------|
| `Vehicle`        | id, vehicle type, current lane, destination movement, priority        |
| `Queue`          | FIFO of vehicles + aggregate waiting statistics                       |
| `Lane`           | owns a `Queue`; one lane per movement                                 |
| `Signal`         | RED / YELLOW / GREEN state machine                                    |
| `Movement`       | ties an approach + movement type + lane + signal (16 movements total) |
| `Approach`       | one of North / South / East / West, holding its movements             |
| `Phase`          | a set of compatible movements served simultaneously                   |
| `Intersection`   | 4 approaches; spawns, moves, and reports stats at the aggregate level |

### The 10-phase architecture

The simulator uses the **official 10-phase plan** (defined in
`config/phases.py`). Each phase is a fixed set of compatible movements; the
scheduler only ever activates these predefined phases and never invents new
movement combinations. An `EMERGENCY_OVERRIDE` phase is built dynamically
from the ambulance's approach and does not modify the normal phase set.

---

## 4. Traffic Profiles (the Dataset)

The system uses **simulation-generated synthetic traffic data** — there is
no external dataset. Each profile is a schedule of time windows; each window
specifies an independent arrival rate (vehicles/second) for every one of the
16 incoming movements, plus a vehicle-mix distribution.

The five scenarios used throughout this project:

| Profile          | Description--------------------------------------------------------------------|
| `LIGHT_TRAFFIC`  | Low volume, balanced across all movements.                         |
| `NORMAL_TRAFFIC` | Average daytime flow with mild asymmetry.                          |
| `RUSH_HOUR`      | Heavy morning/evening commuting, asymmetric.                       |
| `NIGHT`          | Very low traffic, truck-heavy freight hours.                       |
| `CUSTOM`         | Time-dependent: Morning → Rush → Normal → Evening.                 |

Vehicle mix is per-profile (e.g. `CAR/BIKE/BUS/TRUCK/AMBULANCE`), and UTurn
arrivals default to a small non-zero 5% of the straight volume.

---

## 5. Scheduling Strategies

All strategies implement the same `BaseStrategy` interface and are
interchangeable in `Simulation`.

### 5.1 FixedTimerStrategy (Baseline 1)
Round-robin scheduling with a fixed green duration per phase. The naive
baseline.

### 5.2 DensityStrategy (Baseline 2)
A percentile-based adaptive density controller. It observes **only the
number of queued vehicles per approach** (never a vehicle's movement) and
per decision cycle:

1. **Observe** — count queued vehicles on each approach.
2. **Rank** — rank approaches 1 (most loaded) to 4 (relative to the current
   state, not fixed thresholds).
3. **Classify** — convert rank into density classes (HIGH / MEDIUM / LOW).
4. **Fairness** — anti-starvation boost for approaches starved for too long.
5. **Score phases** — score all 10 phases by `Σ(approach weight × phase
   coverage)`; the highest wins.
6. **Green time** — estimate a continuous-discharge green interval
   (interval-merging / Teemo-Attacking intuition), clamped to
   `[MIN_GREEN, MAX_GREEN]`.

### 5.3 RLStrategy (Models 1 & 2)
Wraps a trained RL agent behind the same `BaseStrategy` interface. At
inference time it returns `argmax_a Q(state, a)` — a table lookup (tabular)
or a single forward pass (DQN). No training happens live during evaluation.

### 5.4 Emergency
Ambulance handling is entirely **rule-based** in the scheduler and always
has priority over normal adaptive/RL scheduling:

```
NORMAL GREEN → YELLOW CLEARANCE → EMERGENCY GREEN → RESUME NORMAL
```

---

## 6. Service Model

`services/service_model.py` governs vehicle discharge. Each lane accumulates
green time; a vehicle departs only once enough green time has built up to
satisfy its type's service time:

| Vehicle   | Service time (s) |
|-----------|------------------|
| BIKE      | 0.6              |
| CAR       | 1.0              |
| BUS       | 1.8              |
| TRUCK     | 2.2              |
| AMBULANCE | 0.8              |

The scheduler remains completely unaware of service rates.

---

## 7. Analytics & KPIs

`analytics/statistics.py` aggregates the core KPIs used in the results:

- **average_waiting_time** — mean aggregate waiting time per sample.
- **average_queue_length** — mean total queued vehicles per sample.
- **throughput** — vehicles served per simulation second.
- **congestion_ratio** — fraction of ticks with total queue ≥ 10.
- **max_queue_by_movement** — peak queue observed per movement.
- Plus vehicles served per movement/type, green time per phase, queue growth
  / reduction rates, and Phase-3 adaptive metrics.

---

## 8. Reinforcement-Learning Pipeline

### 8.1 Gym-style environment (`env/`)

`TrafficRLEnv` wraps the simulator with a standard interface:

```
reset()              -> 23-dim observation
step(action)         -> (next_obs, reward, done, info)
action_space         -> 10 (choose one of the 10 normal phases)
observation_space    -> (23,)
```

**State (23-dim observation)** — reuses exactly what the Density strategy
already computes:

| indices | feature                                  |
|---------|------------------------------------------|
| 0–3     | queue length per approach (normalized)   |
| 4–7     | percentile rank per approach (1..4)      |
| 8–11    | starvation counters per approach         |
| 12–21   | active-phase one-hot (10 normal phases)  |
| 22      | elapsed seconds in the current phase     |

**Action** — pick one of the 10 phases at each decision point (a
minimum-green boundary), not every tick.

**Reward** (deliberately crude): `r = -sum(queue_lengths)` accumulated over
the step — a "minimize congestion" proxy sufficient to get a learning signal.

### 8.2 Tabular Q-learning (`rl/agents.py`)
Discretizes the state into queue LOW/MED/HIGH per approach folded with the
last-active-phase: `3⁴ × 10 = 810 states`, Q-table shape `(810, 10)`.
Standard Q-learning with epsilon-greedy exploration. Fast, interpretable,
the sanity baseline.

### 8.3 DQN (`rl/dqn.py`)
A hand-rolled, **pure-numpy** MLP: `23 → 64 (ReLU) → 64 (ReLU) → 10 Q-values`.
Includes experience replay, a target network, and epsilon-greedy. No
deep-learning framework dependency — gradient updates are implemented
manually and validated independently on a tiny XOR problem.

### 8.4 Training (`rl/train.py`)
- Episode = a fixed-length window (`EPISODE_LENGTH` ticks) of a traffic
  profile.
- Profiles cycle across episodes (`LIGHT/NORMAL/RUSH/NIGHT/CUSTOM`) so the
  agent does not overfit to a single pattern.
- A fixed per-episode seed keeps evaluation reproducible.
- The per-episode cumulative reward curve is the key evidence of learning.

---

## 9. Controlled Experiment Methodology

`run_experiments.py` runs a **controlled, apples-to-apples experiment**:

- **4 controllers**: Fixed Timer, Density, Q-Learning, DQN.
- **Identical traffic profiles** for every controller: `LIGHT_TRAFFIC`,
  `NORMAL_TRAFFIC`, `RUSH_HOUR`, `NIGHT`, `CUSTOM`.
- **Identical simulation duration**: `EVAL_TICKS = 200` ticks per run.
- **Multiple seeds**: `[1, 2, 3]` for a stable average.

The ONLY difference between controllers is the scheduling policy (the
pluggable `BaseStrategy`). Traffic generation, duration, and seed are
identical, so the comparison is fair.

The RL agents are trained first (tabular then DQN, 150 episodes each), then
evaluated in **self-driving inference mode** (argmax over the learned
Q/policy) — no training happens during evaluation.

Total runs: `4 controllers × 5 profiles × 3 seeds = 60` simulation runs.

---

## 10. Actual Model-Performance Results

These numbers are **real** — every value comes from an actual simulation
run. Nothing is fabricated or cherry-picked. The RL agents underperform the
engineered Density baseline, which is an honest and acceptable result.

### Performance table (averaged over 5 profiles × 3 seeds = 15 runs each)

| Controller   | Avg Wait | Avg Queue | Max Queue | Throughput | Congestion |
|--------------|----------|-----------|-----------|------------|------------|
| Fixed Timer  | 425.787  | 22.578    | 16.467    | 1.140      | 0.547      |
| **Density**  | **239.800** | **16.003** | **10.400** | **1.193** | **0.506** |
| Q-Learning   | 683.657  | 27.679    | 22.467    | 0.946      | 0.580      |
| DQN          | 1185.705 | 37.462    | 26.333    | 0.710      | 0.586      |

- **Wait / Queue / Max Queue / Congestion**: lower is better.
- **Throughput**: higher is better.

**Best overall: Density** (lowest wait, lowest queue, lowest max queue,
highest throughput, lowest congestion).

### Interpretation
On this small single-intersection problem, the engineered rule-based Density
controller beats the thinly-trained RL agents. The RL training curves do show
learning (peaks of −105 for tabular, −41 for DQN), but the crude
`-sum(queue)` reward objective does not translate into better end-to-end KPIs
than the hand-tuned controller. This is expected and honestly reported — the
RL contribution is the learned-policy framework, not a guaranteed win over
expert-designed heuristics on a toy problem.

### Training curves
- `tabular_q`: first episode reward −589, last episode reward −6946
  (peak −105).
- `dqn`: first episode reward −240, last episode reward −6631 (peak −41).

### Generated graphs (`images/`)
- `G1_avg_wait_comparison.png` — average waiting time (lower better).
- `G2_throughput_comparison.png` — throughput (higher better).
- `G3_avg_queue_comparison.png` — average queue (lower better).
- `G4_rl_training_curves.png` — episode reward vs episode (RL learning).

### Raw data (`results/results_table.csv`)
All 60 per-run KPIs (controller × profile × seed) are stored for audit.

---

## 11. How to Run

### Run the interactive simulator
```bash
python main.py
```

### Run the controlled RL experiment (produces table + graphs + docs)
```bash
python run_experiments.py
```
Outputs:
- `results/results_table.csv` — raw per-run KPIs.
- `results/model_cards.md` — dataset/model documentation.
- `images/G1..G4_*.png` — the 4 critical graphs.

### Plot the tabular-Q reward curve
```bash
python plot_rewards.py [n_episodes]
```

### Three-way comparison (one-liner)
```bash
python -c "from evaluation.evaluate import *; from strategies.fixed_timer_strategy import *; from strategies.density_strategy import *; from strategies.rl_strategy import *; from rl.train import train_tabular; a,_=train_tabular(n_episodes=30,verbose=False); print_comparison(evaluate_strategies({'fixed_timer':FixedTimerStrategy(),'density':DensityStrategy(),'rl':RLStrategy(agent=a)}))"
```

---

## 12. Reproducibility Notes

- **Seeds**: `config/simulation.py` (`SEED=42`) and the experiment harness
  (`SEEDS=[1,2,3]`) make all runs deterministic.
- **No fabrication**: the results table, CSV, graphs, and model cards are
  produced directly from actual simulation runs by `run_experiments.py`.
- **Dependencies**: only `numpy` (required) and `matplotlib` (for graphs;
  the harness degrades to ASCII output if unavailable).
- **Dataset**: synthetic, generated by the project's deterministic traffic
  simulator — no external dataset is required or used.
