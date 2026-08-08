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
- 4  starvation counters (per approach)
- 10 active-phase one-hot values
- 1  elapsed phase time
- **23 total**

### Action space (10 discrete actions)

- PHASE_1 ... PHASE_10

### Reward

```
reward = -sum(queue_lengths)
```

## Models

### 1. Tabular Q-Learning

- State discretization: queue LOW/MED/HIGH per approach (3^4 = 81) folded with last-active-phase (10) => **810 states**.
- Q-table shape: (810, 10).
- Updates: standard Q-learning with epsilon-greedy exploration.

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

- Pure-numpy MLP (no deep-learning framework).
- Experience replay + target network + epsilon-greedy.

## Model Performance Metrics

Averaged over all profiles (LIGHT_TRAFFIC, NORMAL_TRAFFIC, RUSH_HOUR, NIGHT, CUSTOM) and seeds (1, 2, 3).

| Controller | Avg Wait | Avg Queue | Max Queue | Throughput | Congestion |
|------------|----------|-----------|-----------|------------|------------|
| Fixed Timer | 425.787 | 22.578 | 16.467 | 1.140 | 0.547 |
| Density | 239.800 | 16.003 | 10.400 | 1.193 | 0.506 |
| Q-Learning | 683.657 | 27.679 | 22.467 | 0.946 | 0.580 |
| DQN | 1185.705 | 37.462 | 26.333 | 0.710 | 0.586 |

_Wait/Queue/Congestion: lower is better. Throughput: higher is better._

## Training curves

- tabular_q: first episode reward -589, last episode reward -6946 (peak -105).
- dqn: first episode reward -240, last episode reward -6631 (peak -41).
