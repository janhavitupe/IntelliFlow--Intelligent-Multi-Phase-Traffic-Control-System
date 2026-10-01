# Dataset & Model Documentation

## Dataset Details

**Source**: Synthetic traffic data generated using the 
project's deterministic traffic simulator (no external dataset).

**Traffic scenarios**: LIGHT_TRAFFIC, NORMAL_TRAFFIC, RUSH_HOUR, NIGHT, CUSTOM.

**Simulation**: each run is a fixed window of ticks; multiple seeds 
([1, 2, 3]) are used per scenario for a stable average.

### State features (23-dimensional observation)

- 4  queue lengths (per approach)
- 4  percentile ranks (per approach)
- 4  longest current wait (per approach; starvation signal)
- 10 active-phase one-hot values
- 1  elapsed green time of the active phase
- **23 total**, computed by one function in both training and inference

### Action space (10 discrete actions, extend-or-switch)

- PHASE_1 ... PHASE_10, chosen each time the active green runs out.
- Choosing the active phase extends it by 5 s; choosing another starts it (after yellow) with 10 s green.
- Green is capped at 40 s. These are the same bounds the Density controller uses.

### Reward

```
reward = -(sum over the step's ticks of total_queue * tick_seconds) / 1000
```

Queueing delay in vehicle-seconds. Discount is per simulated second (gamma = 0.9925 ** step_seconds); episode time limits are treated as truncation, not termination.

## Models

### 1. Tabular Q-Learning

- State discretization: queue LOW/MED/HIGH per approach (3^4 = 81) x active phase (10) x elapsed-green level (3) => **2430 states**.
- Q-table shape: (2430, 10).
- Updates: Q-learning with linearly decaying epsilon-greedy exploration.

### 2. DQN (Deep Q-Network)

```
23 input features
       |
       64   (ReLU)
       |
       64   (ReLU)
       |
       10 Q-values
```

- Pure-numpy MLP and Adam optimizer (no deep-learning framework).
- Experience replay, target network, Double-DQN targets, Huber loss, gradient-norm clipping.
- Training: 500 episodes of 1200 ticks; best checkpoint on held-out validation seeds is kept.

## Model Performance Metrics

Averaged over all profiles (LIGHT_TRAFFIC, NORMAL_TRAFFIC, RUSH_HOUR, NIGHT, CUSTOM) and seeds (1, 2, 3).

| Controller | Avg Wait | Avg Queue | Max Lane Queue | Throughput | Congestion |
|------------|----------|-----------|-----------|------------|------------|
| Fixed Timer | 425.787 | 22.578 | 16.467 | 1.140 | 0.547 |
| Density | 239.800 | 16.003 | 10.400 | 1.193 | 0.506 |
| Q-Learning | 494.901 | 22.779 | 17.467 | 1.084 | 0.547 |
| DQN | 215.041 | 13.738 | 8.733 | 1.256 | 0.502 |

_Wait/Queue/Congestion: lower is better. Throughput: higher is better._
_Avg Queue is the whole-intersection total (all 16 lanes); Max Lane Queue is the longest single lane seen during the run._
_RL rows come from a single training run (seed 42 / 43); variance across training seeds is not yet included in this table._

## Training curves

Greedy policy scored every 25 episodes on held-out seeds [1000, 1001, 1002] (200 ticks per profile), average waiting time:

- Q-Learning: 1156 untrained -> 561 after 500 episodes; best 437 at episode 450 (this checkpoint is the one evaluated).
- DQN: 1026 untrained -> 306 after 500 episodes; best 242 at episode 375 (this checkpoint is the one evaluated).
- Fixed Timer on the same validation set: 667.
- Density on the same validation set: 265.
