"""
evaluate.py

Three-way strategy comparison harness for Phase 4 (Stage 2).

Runs FixedTimerStrategy, DensityStrategy, and an RLStrategy (wrapping a
trained tabular or DQN agent) through the SAME Simulation on identical
profile + seed, and compares the existing analytics KPIs.

Design:
    - Each run constructs a fresh Simulation with an injected strategy.
    - The traffic profile and seed are identical across strategies (the
      ONLY difference is the scheduling policy), so the comparison is
      apples-to-apples.
    - We reuse the existing Statistics.summary() so no new metrics are
      invented; the phase-3/phase-2 analytics are authoritative.
    - Emergency preemption runs identically for all strategies (rule-based
      in the scheduler), so it does not bias the comparison.

The demo narrative:
    "naive baseline -> engineered rule-based -> learned"
    (FixedTimer -> Density -> RL). Whether RL beats Density is an honest,
    informative result either way.
"""
from simulation import Simulation
from strategies.fixed_timer_strategy import FixedTimerStrategy
from strategies.density_strategy import DensityStrategy
from strategies.rl_strategy import RLStrategy


def run_strategy(
    strategy,
    profile_key="NORMAL_TRAFFIC",
    seed=42,
    max_ticks=200,
):
    """
    Run a single strategy through the Simulation and return its analytics.

    Args:
        strategy (BaseStrategy): the scheduling strategy to test.
        profile_key (str): traffic profile.
        seed (int): RNG seed (shared with other strategies).
        max_ticks (int): number of simulation ticks.

    Returns:
        dict: Statistics.summary() snapshot for the run.
    """
    sim = Simulation(
        profile_key=profile_key,
        seed=seed,
        max_ticks=max_ticks,
        live=False,
        log_to_csv=False,
        strategy=strategy,
    )
    for _ in range(max_ticks):
        sim.step()
    return sim.analytics.summary()


def _metric_table(results, keys):
    """Build a {metric: {strategy: value}} comparison table."""
    table = {}
    for key in keys:
        table[key] = {
            name: summary.get(key) for name, summary in results.items()
        }
    return table


COMPARE_KEYS = [
    "average_waiting_time",
    "average_queue_length",
    "throughput",
    "congestion_ratio",
    "vehicles_served",
    "max_queue_by_movement",
]


def evaluate_strategies(
    strategies,
    profile_key="NORMAL_TRAFFIC",
    seed=42,
    max_ticks=200,
):
    """
    Run multiple strategies on the same scenario and collect results.

    Args:
        strategies (dict): {name: strategy} to evaluate.
        profile_key (str): traffic profile.
        seed (int): shared seed.
        max_ticks (int): simulation ticks per run.

    Returns:
        dict: {name: summary} for each strategy.
    """
    results = {}
    for name, strategy in strategies.items():
        results[name] = run_strategy(
            strategy, profile_key=profile_key, seed=seed, max_ticks=max_ticks
        )
    return results


def _max_queue_value(summary):
    """Extract the max queue metric as a scalar (max over movements)."""
    mq = summary.get("max_queue_by_movement", {})
    if isinstance(mq, dict) and mq:
        return max(mq.values())
    return 0


def print_comparison(results, keys=None):
    """
    Print a readable three-way comparison table.

    Args:
        results (dict): {name: summary} from evaluate_strategies.
        keys (list|None): scalar metrics to display. Defaults to the core
            KPI set (throughput, congestion, avg wait, avg queue).
    """
    if keys is None:
        keys = [
            "vehicles_served",
            "throughput",
            "average_waiting_time",
            "average_queue_length",
            "congestion_ratio",
        ]

    names = list(results.keys())
    print("\n" + "=" * 70)
    print("THREE-WAY STRATEGY COMPARISON")
    print("=" * 70)
    header = f"{'metric':<28}" + "".join(f"{n:>14}" for n in names)
    print(header)
    print("-" * 70)

    for key in keys:
        row = f"{key:<28}"
        for name in names:
            val = results[name].get(key, 0)
            if isinstance(val, float):
                row += f"{val:>14.3f}"
            else:
                row += f"{val:>14}"
        print(row)

    # Max queue (scalar) is important for the congestion story.
    row = f"{'max_queue(movement)':<28}"
    for name in names:
        row += f"{_max_queue_value(results[name]):>14}"
    print(row)

    # Emerging winner summary (low wait/low queue/high throughput).
    print("-" * 70)
    print("Best per metric:")
    for key in keys:
        vals = {name: results[name].get(key, 0) for name in names}
        # For throughput/vehicles served higher is better; for wait/queue/
        # congestion lower is better.
        lower_better = key in (
            "average_waiting_time",
            "average_queue_length",
            "congestion_ratio",
        )
        best = min(vals, key=vals.get) if lower_better else max(vals, key=vals.get)
        print(f"  {key:<28} -> {best}")
