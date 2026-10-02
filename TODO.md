# Roadmap

## Done

### Simulator and controllers
- [x] Domain model: 16 movements (4 approaches × left/straight/right/U-turn), official 10-phase plan
- [x] Generic scheduler + Strategy pattern; Fixed Timer controller
- [x] Simulation realism: per-vehicle-type discharge, time-dependent traffic profiles, seeded runs, CSV logging
- [x] Approach-level ambulance preemption (yellow clearance → emergency green → resume), with a fail-safe timeout and cooldown
- [x] Density controller: rank-based density classes, 10-phase scoring, phase-recency fairness, adaptive green (10–40 s)

### Reinforcement learning
- [x] Gym-style environment over the same simulator; reward credited to the action that caused it; identical features in training and inference
- [x] Extend-or-switch green control with Density's bounds; per-second (semi-MDP) discount
- [x] Tabular Q-learning (2,430 states) and numpy DQN (Adam, Huber loss, gradient clipping, Double DQN)
- [x] Training with a linear epsilon schedule, truncation-aware updates, held-out validation and best-checkpoint selection

### Evaluation and quality
- [x] Per-vehicle delay metrics (avg / P95 / max / ambulance)
- [x] Experiment harness: 10-min runs × 5 profiles × 5 test seeds; RL trained from 3 seeds in parallel; per-profile results; bootstrap CI
- [x] pytest suite (69 tests) with golden KPIs, gradient check and harness smoke test; CI on GitHub Actions
- [x] Documentation refresh (README, PROJECT_DOCUMENTATION, TODO, model card, docstrings)

## Next

- [x] **RL in light/night traffic.** Fixed by rebalancing the reward scale against the
      Huber loss (validated on held-out seeds). DQN now beats or ties Density in every scenario.
- [ ] **DQN worst-case delay.** Its maximum delay is still longer than Density's.
      Candidate: a longest-wait penalty in the reward.
- [ ] **Phase-conflict test.** Needs an authoritative movement conflict matrix for the
      10-phase plan (left-hand traffic).
- [ ] **Density approach-level starvation boost** never fires with this phase plan (every
      phase serves every approach). Redefine it per movement, or remove it.
- [ ] **RUSH_HOUR calibration.** Demand exceeds capacity and ~1% of vehicles are
      ambulances. Decide whether to keep it as a stress test only, or add a realistic
      near-capacity rush profile.

## Future integrations
- [ ] YOLO / OpenCV vehicle detection as a traffic source (`traffic_source/yolo_generator.py`)
- [ ] SUMO integration (`traffic_source/sumo_generator.py`)
- [ ] Web dashboard
- [ ] Database logging
- [ ] Queue-relaxation strategy (`strategies/queue_relaxation_strategy.py`)
