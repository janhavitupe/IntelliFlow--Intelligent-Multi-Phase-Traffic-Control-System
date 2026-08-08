"""
evaluation package

Phase 4 (Stage 2) evaluation harness: an apples-to-apples three-way
comparison of the scheduling strategies on identical traffic scenarios.

    FixedTimerStrategy  -> "naive baseline" (round-robin)
    DensityStrategy     -> "engineered rule-based" (Phase 3 percentile density)
    RLStrategy          -> "learned" (trained tabular Q or DQN)

Each strategy runs the SAME profile + seed through the standard Simulation
class (with an injected strategy), and the existing Statistics.summary()
metrics (avg wait, throughput, max queue, congestion ratio) are compared.
Using a fixed seed per evaluation keeps the comparison reproducible, so the
difference between strategies is due to the policy, not RNG noise.
"""
from .evaluate import evaluate_strategies, run_strategy, print_comparison

__all__ = ["evaluate_strategies", "run_strategy", "print_comparison"]
