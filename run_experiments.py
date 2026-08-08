"""
run_experiments.py

Controlled experiment harness that produces the Model Performance Metrics
featured in the submission.

It trains the RL agents (tabular Q-learning and DQN) and then evaluates all
four controllers through the SAME traffic simulator under IDENTICAL:
    - traffic profiles (LIGHT / NORMAL / RUSH_HOUR / NIGHT / CUSTOM)
    - simulation duration (max_ticks per run)
    - seeds (multiple, for a stable average)

The ONLY thing that differs between controllers is the scheduling policy
(a pluggable BaseStrategy), so the comparison is apples-to-apples.

Outputs:
    1. A console performance table (Avg Wait / Avg Queue / Max Queue /
       Throughput / Congestion) - lower is better for Wait/Queue/Max/
       Congestion, higher is better for Throughput.
    2. 4 critical graphs (saved to images/):
       G1 avg_wait_comparison.png   - average waiting time (lower better)
       G2 throughput_comparison.png - throughput (higher better)
       G3 avg_queue_comparison.png  - average queue (lower better)
       G4 rl_training_curves.png    - episode reward vs episode (ML learned)
    3. results/results_table.csv   - the raw numbers
    4. results/model_cards.md      - the dataset/model documentation

No results are fabricated or cherry-picked: every number comes from an
actual simulation run.
"""
import os
import csv
import json
import time

import numpy as np

from config import rl as rl_config
from simulation import Simulation
from strategies.fixed_timer_strategy import FixedTimerStrategy
from strategies.density_strategy import DensityStrategy
from strategies.rl_strategy import RLStrategy
from rl.train import train_tabular, train_dqn
from rl.agents import TabularQAgent
from rl.dqn import DQNAgent

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
PROFILES = list(rl_config.PROFILES)
SEEDS = [1, 2, 3]                 # multiple seeds for a stable average

TRAIN_EPISODES = 150              # training budget for the RL agents
EVAL_TICKS = 200                  # simulation ticks per evaluation run
OUT_DIR = "results"
IMG_DIR = "images"

def run_controller(strategy, profile_key, seed, max_ticks):
    """Run one controller on one (profile, seed) and return scalar KPIs."""
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
    s = sim.analytics.summary()
    mq = s.get("max_queue_by_movement", {})
    max_q = max(mq.values()) if isinstance(mq, dict) and mq else 0
    return {
        "avg_wait": float(s.get("average_waiting_time", 0.0)),
        "avg_queue": float(s.get("average_queue_length", 0.0)),
        "max_queue": float(max_q),
        "throughput": float(s.get("throughput", 0.0)),
        "congestion": float(s.get("congestion_ratio", 0.0)),
    }


def fresh_fixed_timer():
    return FixedTimerStrategy()


def fresh_density():
    return DensityStrategy()


def train_and_wrap_tabular(seed):
    agent, rewards = train_tabular(
        n_episodes=TRAIN_EPISODES,
        episode_length=rl_config.EPISODE_LENGTH,
        seed=seed,
        verbose=False,
    )
    rl = RLStrategy(agent=agent)
    rl.reset()
    return rl, rewards


def train_and_wrap_dqn(seed):
    agent, rewards = train_dqn(
        agent=DQNAgent(
            obs_dim=rl_config.OBS_DIM,
            n_actions=rl_config.NUM_PHASES,
            hidden=rl_config.DQN_HIDDEN_LAYERS,
            seed=seed,
        ),
        n_episodes=TRAIN_EPISODES,
        episode_length=rl_config.EPISODE_LENGTH,
        seed=seed,
        verbose=False,
    )
    rl = RLStrategy(agent=agent)
    rl.reset()
    return rl, rewards


def collect_all(controllers, max_ticks):
    """
    For each controller, run it on every (profile, seed) and aggregate.

    Returns:
        dict: controller_name -> list of per-(profile, seed) KPI dicts.
    """
    all_results = {name: [] for name in controllers}
    total = len(controllers) * len(PROFILES) * len(SEEDS)
    done = 0
    for name, strategy_factory in controllers.items():
        # A fresh strategy per (profile, seed) run so RL strategies are not
        # required to be stateless across runs (deterministic self-drive).
        for profile in PROFILES:
            for seed in SEEDS:
                strategy = strategy_factory()
                kpi = run_controller(strategy, profile, seed, max_ticks)
                kpi["profile"] = profile
                kpi["seed"] = seed
                kpi["controller"] = name
                all_results[name].append(kpi)
                done += 1
        print(f"  [{name}] completed ({len(SEEDS)} seeds x {len(PROFILES)} profiles)")
    print(f"  total runs: {done}/{total}")
    return all_results


def mean_across_profiles_seeds(rows):
    """Average scalar KPIs across all (profile, seed) runs of a controller."""
    n = len(rows)
    return {
        k: sum(r[k] for r in rows) / n for k in
        ("avg_wait", "avg_queue", "max_queue", "throughput", "congestion")
    }


def write_csv(all_results, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "controller", "profile", "seed",
            "avg_wait", "avg_queue", "max_queue", "throughput", "congestion",
        ])
        for name, rows in all_results.items():
            for r in rows:
                writer.writerow([
                    name, r["profile"], r["seed"],
                    round(r["avg_wait"], 3), round(r["avg_queue"], 3),
                    round(r["max_queue"], 3), round(r["throughput"], 3),
                    round(r["congestion"], 3),
                ])


def print_table(all_results):
    names = list(all_results.keys())
    means = {n: mean_across_profiles_seeds(all_results[n]) for n in names}

    print("\n" + "=" * 88)
    print("MODEL PERFORMANCE METRICS  (averaged over %d profiles x %d seeds)" %
          (len(PROFILES), len(SEEDS)))
    print("=" * 88)
    header = (f"{'Controller':<14}" + f"{'Avg Wait':>10}" + f"{'Avg Queue':>10}"
              + f"{'Max Queue':>10}" + f"{'Throughput':>12}" + f"{'Congest':>9}")
    print(header)
    print("-" * 88)
    for name in names:
        m = means[name]
        print(f"{name:<14}" + f"{m['avg_wait']:>10.1f}" + f"{m['avg_queue']:>10.1f}"
              + f"{m['max_queue']:>10.0f}" + f"{m['throughput']:>12.3f}"
              + f"{m['congestion']:>9.3f}")
    print("-" * 88)
    print("Best (early guide, not a substitute for reading the table):")
    best_wait = min(means, key=lambda n: means[n]["avg_wait"])
    best_q = min(means, key=lambda n: means[n]["avg_queue"])
    best_maxq = min(means, key=lambda n: means[n]["max_queue"])
    best_tp = max(means, key=lambda n: means[n]["throughput"])
    best_cong = min(means, key=lambda n: means[n]["congestion"])
    print(f"  lowest avg wait : {best_wait}")
    print(f"  lowest avg queue: {best_q}")
    print(f"  lowest max queue: {best_maxq}")
    print(f"  highest throughput: {best_tp}")
    print(f"  lowest congestion: {best_cong}")
    return means


def make_graphs(all_results, training_curves, out_dir=IMG_DIR):
    """
    Generate the 4 critical graphs with matplotlib (graceful ASCII fallback).
    """
    os.makedirs(out_dir, exist_ok=True)
    names = list(all_results.keys())
    means = {n: mean_across_profiles_seeds(all_results[n]) for n in names}
    metrics = ["avg_wait", "avg_queue", "max_queue", "throughput", "congestion"]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available - printing ASCII summary instead.")
        for name in names:
            m = means[name]
            print(f"{name:<12} wait={m['avg_wait']:.0f} q={m['avg_queue']:.0f} "
                  f"maxq={m['max_queue']:.0f} tp={m['throughput']:.2f} "
                  f"cong={m['congestion']:.2f}")
        return

    x = np.arange(len(names))
    width = 0.55

    # ---- Graph 1 : Average waiting time (lower better) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, [means[n]["avg_wait"] for n in names], width, color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Average waiting time (s)")
    ax.set_title("Controller Comparison - Average Waiting Time (lower is better)")
    for xi, n in zip(x, names):
        ax.text(xi, means[n]["avg_wait"], f"{means[n]['avg_wait']:.0f}",
                ha="center", va="bottom", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "G1_avg_wait_comparison.png"))
    plt.close(fig)

    # ---- Graph 2 : Throughput (higher better) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, [means[n]["throughput"] for n in names], width, color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Throughput (veh/s)")
    ax.set_title("Controller Comparison - Throughput (higher is better)")
    for xi, n in zip(x, names):
        ax.text(xi, means[n]["throughput"], f"{means[n]['throughput']:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "G2_throughput_comparison.png"))
    plt.close(fig)

    # ---- Graph 3 : Average queue (lower better) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, [means[n]["avg_queue"] for n in names], width, color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Average queue length (veh)")
    ax.set_title("Controller Comparison - Average Queue (lower is better)")
    for xi, n in zip(x, names):
        ax.text(xi, means[n]["avg_queue"], f"{means[n]['avg_queue']:.0f}",
                ha="center", va="bottom", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "G3_avg_queue_comparison.png"))
    plt.close(fig)

    # ---- Graph 4 : RL training curves (episode reward vs episode) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    if training_curves:
        for label, rewards in training_curves.items():
            ax.plot(rewards, label=label, linewidth=1.5)
            # Simple moving average overlay for readability.
            window = max(1, len(rewards) // 10)
            if len(rewards) >= window:
                kernel = np.ones(window) / window
                smooth = np.convolve(rewards, kernel, mode="valid")
                ax.plot(range(window - 1, len(rewards)), smooth,
                        linestyle="--", alpha=0.6)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative episode reward (-sum queue)")
    ax.set_title("RL Training Curves - Episode Reward (rising = learning)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "G4_rl_training_curves.png"))
    plt.close(fig)

    print(f"Saved graphs to {out_dir}/G1..G4_*.png")


def write_dataset_doc(all_results, training_curves, path):
    """Write the model card / dataset documentation markdown."""
    names = list(all_results.keys())
    means = {n: mean_across_profiles_seeds(all_results[n]) for n in names}

    def fmt(v):
        return f"{v:.3f}"

    lines = []
    lines.append("# Dataset & Model Documentation")
    lines.append("")
    lines.append("## Dataset Details")
    lines.append("")
    lines.append("**Source**: Synthetic traffic data generated using the ")
    lines.append("project's deterministic traffic simulator (no external dataset).")
    lines.append("")
    lines.append("**Traffic scenarios**: " + ", ".join(PROFILES) + ".")
    lines.append("")
    lines.append("**Simulation**: each run is a fixed window of ticks; multiple seeds ")
    lines.append(f"({SEEDS}) are used per scenario for a stable average.")
    lines.append("")
    lines.append("### State features (23-dimensional observation)")
    lines.append("")
    lines.append("- 4  queue lengths (per approach)")
    lines.append("- 4  percentile ranks (per approach)")
    lines.append("- 4  starvation counters (per approach)")
    lines.append("- 10 active-phase one-hot values")
    lines.append("- 1  elapsed phase time")
    lines.append("- **23 total**")
    lines.append("")
    lines.append("### Action space (10 discrete actions)")
    lines.append("")
    lines.append("- PHASE_1 ... PHASE_10")
    lines.append("")
    lines.append("### Reward")
    lines.append("")
    lines.append("```")
    lines.append("reward = -sum(queue_lengths)")
    lines.append("```")
    lines.append("")
    lines.append("## Models")
    lines.append("")
    lines.append("### 1. Tabular Q-Learning")
    lines.append("")
    lines.append("- State discretization: queue LOW/MED/HIGH per approach "
                 "(3^4 = 81) folded with last-active-phase (10) => **810 states**.")
    lines.append("- Q-table shape: (810, 10).")
    lines.append("- Updates: standard Q-learning with epsilon-greedy exploration.")
    lines.append("")
    lines.append("### 2. DQN (Deep Q-Network)")
    lines.append("")
    lines.append("```")
    lines.append("23 input features")
    lines.append("       |")
    lines.append("       64   (ReLU)")
    lines.append("       |")
    lines.append("       64   (ReLU)")
    lines.append("       |")
    lines.append("       10 Q-values")
    lines.append("```")
    lines.append("")
    lines.append("- Pure-numpy MLP (no deep-learning framework).")
    lines.append("- Experience replay + target network + epsilon-greedy.")
    lines.append("")
    lines.append("## Model Performance Metrics")
    lines.append("")
    lines.append("Averaged over all profiles (" + ", ".join(PROFILES) + ") and seeds ("
                 + ", ".join(str(s) for s in SEEDS) + ").")
    lines.append("")
    lines.append("| Controller | Avg Wait | Avg Queue | Max Queue | Throughput | Congestion |")
    lines.append("|------------|----------|-----------|-----------|------------|------------|")
    for name in names:
        m = means[name]
        lines.append(f"| {name} | {fmt(m['avg_wait'])} | {fmt(m['avg_queue'])} | "
                     f"{fmt(m['max_queue'])} | {fmt(m['throughput'])} | "
                     f"{fmt(m['congestion'])} |")
    lines.append("")
    lines.append("_Wait/Queue/Congestion: lower is better. Throughput: higher is better._")
    lines.append("")
    lines.append("## Training curves")
    lines.append("")
    for label, rewards in training_curves.items():
        if rewards:
            lines.append(f"- {label}: first episode reward {rewards[0]:.0f}, "
                         f"last episode reward {rewards[-1]:.0f} "
                         f"(peak {max(rewards):.0f}).")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote dataset/model documentation to {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    print("=" * 88)
    print("PHASE 4 - CONTROLLED RL EXPERIMENTS")
    print(f"profiles={PROFILES} seeds={SEEDS} eval_ticks={EVAL_TICKS}")
    print(f"RL training budget: {TRAIN_EPISODES} episodes per agent")
    print("=" * 88)

    # ---- Train the RL agents (inference-time wrappers) ----
    print("\n[1/3] Training RL agents...")
    tabular_rl, tabular_rewards = train_and_wrap_tabular(rl_config.SEED)
    dqn_rl, dqn_rewards = train_and_wrap_dqn(rl_config.SEED + 1)
    training_curves = {
        "tabular_q": tabular_rewards,
        "dqn": dqn_rewards,
    }
    print(f"  tabular last reward = {tabular_rewards[-1]:.0f}")
    print(f"  dqn last reward     = {dqn_rewards[-1]:.0f}")

    # ---- Evaluate all 4 controllers ----
    # Note: RL strategies self-drive in the Simulation (argmax over the
    # learned Q / policy net) - no training happens during evaluation.
    print("\n[2/3] Evaluating controllers...")
    controllers = {
        "Fixed Timer": fresh_fixed_timer,
        "Density": fresh_density,
        "Q-Learning": lambda: tabular_rl,
        "DQN": lambda: dqn_rl,
    }
    all_results = collect_all(controllers, EVAL_TICKS)

    # ---- Table + graphs ----
    print("\n[3/3] Building table + graphs + docs...")
    means = print_table(all_results)
    write_csv(all_results, os.path.join(OUT_DIR, "results_table.csv"))
    make_graphs(all_results, training_curves, IMG_DIR)
    write_dataset_doc(all_results, training_curves,
                      os.path.join(OUT_DIR, "model_cards.md"))

    print("\n" + "=" * 88)
    print(f"EXPERIMENTS COMPLETE in {time.time() - t0:.1f}s")
    print(f"  CSV table  -> {os.path.join(OUT_DIR, 'results_table.csv')}")
    print(f"  model docs -> {os.path.join(OUT_DIR, 'model_cards.md')}")
    print(f"  graphs     -> {IMG_DIR}/G1..G4_*.png")
    print("=" * 88)


if __name__ == "__main__":
    main()

