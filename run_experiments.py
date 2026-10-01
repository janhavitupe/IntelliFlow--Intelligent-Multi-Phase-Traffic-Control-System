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
    1. A console performance table (Avg Wait / Avg Queue / Max Lane Queue /
       Throughput / Congestion) - lower is better for Wait/Queue/Max/
       Congestion, higher is better for Throughput.
    2. 4 critical graphs (saved to images/):
       G1 avg_wait_comparison.png   - average waiting time (lower better)
       G2 throughput_comparison.png - throughput (higher better)
       G3 avg_queue_comparison.png  - average queue (lower better)
       G4 rl_training_curves.png    - greedy-policy validation wait vs
                                      training episode, with Fixed/Density
                                      reference lines (ML learned)
    3. results/results_table.csv   - the raw numbers
    4. results/model_cards.md      - the dataset/model documentation
    5. models/                     - trained agents (q_table.npy, dqn.npz)

Usage:
    python run_experiments.py              # train, save models, evaluate
    python run_experiments.py --use-saved  # skip training, load models/

No results are fabricated or cherry-picked: every number comes from an
actual simulation run.
"""
import os
import csv
import json
import sys
import time

import numpy as np

from config import rl as rl_config
from simulation import Simulation
from strategies.fixed_timer_strategy import FixedTimerStrategy
from strategies.density_strategy import DensityStrategy
from strategies.rl_strategy import RLStrategy
from rl.train import train_tabular, train_dqn, score_policy
from rl.agents import TabularQAgent
from rl.dqn import DQNAgent

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
PROFILES = list(rl_config.PROFILES)
SEEDS = [1, 2, 3]                 # multiple seeds for a stable average

EVAL_TICKS = 200                  # simulation ticks per evaluation run
OUT_DIR = "results"
IMG_DIR = "images"
MODEL_DIR = "models"
Q_TABLE_PATH = os.path.join(MODEL_DIR, "q_table.npy")
DQN_PATH = os.path.join(MODEL_DIR, "dqn.npz")

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
    agent, history = train_tabular(seed=seed, verbose=True)
    agent.save(Q_TABLE_PATH)
    return agent, history


def train_and_wrap_dqn(seed):
    agent, history = train_dqn(
        agent=DQNAgent(
            obs_dim=rl_config.OBS_DIM,
            n_actions=rl_config.NUM_PHASES,
            hidden=rl_config.DQN_HIDDEN_LAYERS,
            seed=seed,
        ),
        seed=seed,
        verbose=True,
    )
    agent.save(DQN_PATH)
    return agent, history


def load_saved_agents():
    """Load agents written by a previous training run."""
    tabular = TabularQAgent()
    tabular.load(Q_TABLE_PATH)
    dqn = DQNAgent(
        obs_dim=rl_config.OBS_DIM,
        n_actions=rl_config.NUM_PHASES,
        hidden=rl_config.DQN_HIDDEN_LAYERS,
    )
    dqn.load(DQN_PATH)
    return tabular, dqn


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
              + f"{'MaxLaneQ':>10}" + f"{'Throughput':>12}" + f"{'Congest':>9}")
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
    print(f"  lowest max lane queue: {best_maxq}")
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

    # ---- Graph 4 : RL learning curves (greedy validation wait) ----
    # Raw episode rewards mix five profiles of very different difficulty,
    # so the honest learning signal is the greedy policy's score on a fixed
    # held-out validation set, with the rule-based controllers as reference.
    fig, ax = plt.subplots(figsize=(8, 5))
    curves = (training_curves or {}).get("validation", {})
    for label, points in curves.items():
        eps, waits = zip(*points)
        ax.plot(eps, waits, marker="o", markersize=3, linewidth=1.5, label=label)
    styles = {"Fixed Timer": ":", "Density": "--"}
    for label, wait in (training_curves or {}).get("reference", {}).items():
        ax.axhline(wait, color="gray", linestyle=styles.get(label, "-."),
                   linewidth=1.2, label=f"{label} (reference)")
    ax.set_yscale("log")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Validation avg waiting time (log scale)")
    ax.set_title("RL Learning Curves - greedy policy on held-out seeds (lower is better)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
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
    lines.append("- 4  longest current wait (per approach; starvation signal)")
    lines.append("- 10 active-phase one-hot values")
    lines.append("- 1  elapsed green time of the active phase")
    lines.append("- **23 total**, computed by one function in both training and inference")
    lines.append("")
    lines.append("### Action space (10 discrete actions, extend-or-switch)")
    lines.append("")
    lines.append("- PHASE_1 ... PHASE_10, chosen each time the active green runs out.")
    lines.append(f"- Choosing the active phase extends it by {rl_config.GREEN_EXTENSION:.0f} s; "
                 f"choosing another starts it (after yellow) with {rl_config.MIN_GREEN:.0f} s green.")
    lines.append(f"- Green is capped at {rl_config.MAX_GREEN:.0f} s. These are the same bounds "
                 "the Density controller uses.")
    lines.append("")
    lines.append("### Reward")
    lines.append("")
    lines.append("```")
    lines.append(f"reward = -(sum over the step's ticks of total_queue * tick_seconds) / {rl_config.REWARD_SCALE:.0f}")
    lines.append("```")
    lines.append("")
    lines.append("Queueing delay in vehicle-seconds. Discount is per simulated second "
                 f"(gamma = {rl_config.GAMMA_PER_SECOND} ** step_seconds); episode time "
                 "limits are treated as truncation, not termination.")
    lines.append("")
    lines.append("## Models")
    lines.append("")
    lines.append("### 1. Tabular Q-Learning")
    lines.append("")
    lines.append("- State discretization: queue LOW/MED/HIGH per approach (3^4 = 81) "
                 "x active phase (10) x elapsed-green level (3) => **2430 states**.")
    lines.append("- Q-table shape: (2430, 10).")
    lines.append("- Updates: Q-learning with linearly decaying epsilon-greedy exploration.")
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
    lines.append("- Pure-numpy MLP and Adam optimizer (no deep-learning framework).")
    lines.append("- Experience replay, target network, Double-DQN targets, Huber loss, "
                 "gradient-norm clipping.")
    lines.append(f"- Training: {rl_config.DQN_EPISODES} episodes of {rl_config.EPISODE_LENGTH} "
                 "ticks; best checkpoint on held-out validation seeds is kept.")
    lines.append("")
    lines.append("## Model Performance Metrics")
    lines.append("")
    lines.append("Averaged over all profiles (" + ", ".join(PROFILES) + ") and seeds ("
                 + ", ".join(str(s) for s in SEEDS) + ").")
    lines.append("")
    lines.append("| Controller | Avg Wait | Avg Queue | Max Lane Queue | Throughput | Congestion |")
    lines.append("|------------|----------|-----------|-----------|------------|------------|")
    for name in names:
        m = means[name]
        lines.append(f"| {name} | {fmt(m['avg_wait'])} | {fmt(m['avg_queue'])} | "
                     f"{fmt(m['max_queue'])} | {fmt(m['throughput'])} | "
                     f"{fmt(m['congestion'])} |")
    lines.append("")
    lines.append("_Wait/Queue/Congestion: lower is better. Throughput: higher is better._")
    lines.append("_Avg Queue is the whole-intersection total (all 16 lanes); "
                 "Max Lane Queue is the longest single lane seen during the run._")
    lines.append("_RL rows come from a single training run (seed "
                 f"{rl_config.SEED} / {rl_config.SEED + 1}); variance across training "
                 "seeds is not yet included in this table._")
    lines.append("")
    lines.append("## Training curves")
    lines.append("")
    validation = training_curves.get("validation", {})
    if validation:
        lines.append(f"Greedy policy scored every {rl_config.VALIDATION_EVERY} episodes on "
                     f"held-out seeds {list(rl_config.VALIDATION_SEEDS)} "
                     f"({rl_config.VALIDATION_TICKS} ticks per profile), "
                     "average waiting time:")
        lines.append("")
        for label, points in validation.items():
            start, end = points[0][1], points[-1][1]
            best_ep, best = min(points, key=lambda p: p[1])
            lines.append(f"- {label}: {start:.0f} untrained -> {end:.0f} after "
                         f"{points[-1][0]} episodes; best {best:.0f} at episode "
                         f"{best_ep} (this checkpoint is the one evaluated).")
        for label, wait in training_curves.get("reference", {}).items():
            lines.append(f"- {label} on the same validation set: {wait:.0f}.")
    else:
        lines.append("- Agents loaded from models/ (no training this run).")
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
    print(f"RL training budget: {rl_config.TABULAR_EPISODES} (tabular) / "
          f"{rl_config.DQN_EPISODES} (DQN) episodes of {rl_config.EPISODE_LENGTH} ticks")
    print("=" * 88)

    # ---- Train (or load) the RL agents ----
    if "--use-saved" in sys.argv:
        print("\n[1/3] Loading RL agents from models/ ...")
        tabular_agent, dqn_agent = load_saved_agents()
        training_curves = {}
    else:
        print("\n[1/3] Training RL agents...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(" tabular Q-learning:")
        tabular_agent, tabular_hist = train_and_wrap_tabular(rl_config.SEED)
        print(" DQN:")
        dqn_agent, dqn_hist = train_and_wrap_dqn(rl_config.SEED + 1)
        training_curves = {
            "validation": {
                "Q-Learning": tabular_hist["validation"],
                "DQN": dqn_hist["validation"],
            },
            "reference": {
                "Fixed Timer": score_policy(fresh_fixed_timer),
                "Density": score_policy(fresh_density),
            },
        }

    # ---- Evaluate all 4 controllers ----
    # Note: RL strategies self-drive in the Simulation (argmax over the
    # learned Q / policy net) - no training happens during evaluation.
    print("\n[2/3] Evaluating controllers...")
    controllers = {
        "Fixed Timer": fresh_fixed_timer,
        "Density": fresh_density,
        # Fresh RLStrategy per run: it carries per-run bookkeeping
        # (starvation counters, phase start time) that must not leak.
        "Q-Learning": lambda: RLStrategy(agent=tabular_agent),
        "DQN": lambda: RLStrategy(agent=dqn_agent),
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

