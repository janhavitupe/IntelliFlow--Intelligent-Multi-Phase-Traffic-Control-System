"""
run_experiments.py

Controlled experiment harness that produces the Model Performance Metrics
featured in the submission.

It trains the RL agents (tabular Q-learning and DQN) and then evaluates all
four controllers through the SAME traffic simulator under IDENTICAL:
    - traffic profiles (LIGHT / NORMAL / RUSH_HOUR / NIGHT / CUSTOM)
    - simulation duration (EVAL_TICKS per run)
    - test seeds (TEST_SEEDS, disjoint from training and validation seeds)

The ONLY thing that differs between controllers is the scheduling policy
(a pluggable BaseStrategy), so the comparison is apples-to-apples.

Uncertainty is reported, not hidden:
    - Each RL agent is trained from TRAIN_SEEDS independent seeds (in
      parallel); RL rows show mean +/- std across those training runs.
    - Results are broken down per profile, and summarized as the mean
      % improvement over Fixed Timer across profiles, so the heavily
      oversaturated RUSH_HOUR profile cannot dominate a single average.
    - DQN vs Density gets a paired bootstrap 95% confidence interval.

Primary metric: average delay per vehicle (seconds), counting vehicles still
queued at the end with the delay accrued so far.

Outputs:
    1. Console tables.
    2. Graphs (images/):
       G1_avg_delay_comparison.png - avg delay per vehicle (+/- training std)
       G2_throughput_comparison.png - throughput
       G3_delay_by_profile.png     - avg delay per vehicle, per profile
       G4_rl_training_curves.png   - validation delay vs training episode
    3. results/results_table.csv   - one row per run (raw numbers)
    4. results/model_cards.md      - dataset / model / results documentation
    5. models/                     - trained agents, one file per seed

Usage:
    python run_experiments.py              # train, save models, evaluate
    python run_experiments.py --use-saved  # skip training, load models/

Environment variables (inherited by the worker processes):
    EXPERIMENTS_OUT=<dir>   write results/, images/, models/ under <dir>
    EXPERIMENTS_QUICK=1     tiny smoke-test configuration (seconds, not minutes)

No results are fabricated or cherry-picked: every number comes from an
actual simulation run.
"""
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

# One BLAS thread per process, set BEFORE numpy loads (workers inherit it).
# The networks are tiny, so threading gains nothing, and N processes x
# (one OpenBLAS thread + buffers per core) exhausts memory and kills workers.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402  (must follow the thread limits above)

from config import perception as perception_config
from config import rl as rl_config
from simulation import Simulation
from strategies.fixed_timer_strategy import FixedTimerStrategy
from strategies.density_strategy import DensityStrategy
from strategies.rl_strategy import RLStrategy
from perception import HeadingIntentPerception, NoisyPerception
from rl.train import train_tabular, train_dqn, score_policy
from rl.agents import TabularQAgent
from rl.dqn import DQNAgent

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
PROFILES = list(rl_config.PROFILES)
TEST_SEEDS = [1, 2, 3, 4, 5]
TRAIN_SEEDS = [42, 7, 123]        # independent training runs per RL agent
EVAL_TICKS = 1200                 # 10 simulated minutes per evaluation run
TRAIN_EPISODES = None             # None = config default
TRAIN_EPISODE_LENGTH = None

QUICK = os.environ.get("EXPERIMENTS_QUICK") == "1"
if QUICK:
    TEST_SEEDS, TRAIN_SEEDS, EVAL_TICKS = [1], [42, 7], 200
    TRAIN_EPISODES, TRAIN_EPISODE_LENGTH = 2, 200

_ROOT = os.environ.get("EXPERIMENTS_OUT", ".")
OUT_DIR = os.path.join(_ROOT, "results")
IMG_DIR = os.path.join(_ROOT, "images")
MODEL_DIR = os.path.join(_ROOT, "models")

RULE_BASED = ["Fixed Timer", "Density"]
RL_AGENTS = ["Q-Learning", "DQN"]
CONTROLLERS = RULE_BASED + RL_AGENTS

# Robustness / deployment-option evaluations of the SAME trained DQNs (no
# selection is made with these; they only report how the policy holds up).
VARIANT_ROWS = {
    "DQN (raw policy)": {"max_red": None, "serve_waiting": False},
    "DQN + camera noise": {"perception": "noisy"},
    "DQN + heading intent": {"perception": "intent"},
    "DQN + intent + camera noise": {"perception": "intent+noisy"},
}


def make_perception(kind):
    """Fresh perception source for one run (None = exact ground truth)."""
    if kind == "noisy":
        return NoisyPerception(seed=0)
    if kind == "intent":
        return HeadingIntentPerception(seed=0)
    if kind == "intent+noisy":
        return HeadingIntentPerception(base=NoisyPerception(seed=0), seed=0)
    return None

# (key, label, higher_is_better)
METRICS = [
    ("avg_delay", "Avg delay/veh (s)", False),
    ("p95_delay", "P95 delay (s)", False),
    ("max_delay", "Max delay (s)", False),
    ("ambulance_delay", "Ambulance delay (s)", False),
    ("avg_queue", "Avg queue (veh)", False),
    ("throughput", "Throughput (veh/s)", True),
    ("queued_veh_s", "Queued veh-s", False),
]
METRIC_KEYS = [k for k, _, _ in METRICS]


# ---------------------------------------------------------------------------
# Single runs
# ---------------------------------------------------------------------------

def run_controller(strategy, profile_key, seed, max_ticks):
    """Run one controller on one (profile, seed) and return scalar KPIs."""
    sim = Simulation(profile_key=profile_key, seed=seed, max_ticks=max_ticks,
                     live=False, log_to_csv=False, strategy=strategy)
    for _ in range(max_ticks):
        sim.step()
    a = sim.analytics
    amb = a.ambulance_average_delay
    return {
        "avg_delay": a.average_delay,
        "p95_delay": a.p95_delay,
        "max_delay": a.max_delay,
        "ambulance_delay": float("nan") if amb is None else amb,
        "avg_queue": a.average_queue_length,
        "throughput": a.throughput,
        "queued_veh_s": a.average_waiting_time,
    }


def history_path():
    return os.path.join(MODEL_DIR, "training_history.json")


def save_training(training):
    """Persist learning curves + references so --use-saved can report them."""
    data = {
        "validation": {f"{n}|{s}": pts for (n, s), pts in training["validation"].items()},
        "reference": training["reference"],
    }
    with open(history_path(), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)


def load_training():
    if not os.path.exists(history_path()):
        return None
    with open(history_path(), encoding="utf-8") as f:
        data = json.load(f)
    validation = {}
    for key, pts in data["validation"].items():
        name, seed = key.split("|")
        validation[(name, int(seed))] = [tuple(p) for p in pts]
    return {"validation": validation, "reference": data["reference"]}


def model_path(name, train_seed):
    ext = "npy" if name == "Q-Learning" else "npz"
    slug = "q_table" if name == "Q-Learning" else "dqn"
    return os.path.join(MODEL_DIR, f"{slug}_seed{train_seed}.{ext}")


def new_agent(name, seed=None):
    if name == "Q-Learning":
        return TabularQAgent(seed=seed)
    return DQNAgent(obs_dim=rl_config.OBS_DIM, n_actions=rl_config.NUM_PHASES,
                    hidden=rl_config.DQN_HIDDEN_LAYERS, seed=seed)


def load_agent(name, train_seed):
    agent = new_agent(name)
    agent.load(model_path(name, train_seed))
    return agent


def strategy_factory(name, train_seed=None):
    if name == "Fixed Timer":
        return FixedTimerStrategy
    if name == "Density":
        return DensityStrategy
    if name in VARIANT_ROWS:
        opts = VARIANT_ROWS[name]
        agent = load_agent("DQN", train_seed)
        safety = {k: opts[k] for k in ("max_red", "serve_waiting") if k in opts}
        return lambda: RLStrategy(agent=agent, **safety,
                                  perception=make_perception(opts.get("perception")))
    agent = load_agent(name, train_seed)
    return lambda: RLStrategy(agent=agent)


# ---------------------------------------------------------------------------
# Parallel workers (top-level so they can be pickled on Windows)
# ---------------------------------------------------------------------------

def _train_worker(job):
    name, train_seed = job
    train = train_tabular if name == "Q-Learning" else train_dqn
    agent, history = train(agent=new_agent(name, train_seed), seed=train_seed, verbose=False,
                           n_episodes=TRAIN_EPISODES, episode_length=TRAIN_EPISODE_LENGTH)
    agent.save(model_path(name, train_seed))
    return name, train_seed, history


def _eval_worker(job):
    name, train_seed = job
    factory = strategy_factory(name, train_seed)
    rows = []
    for profile in PROFILES:
        for seed in TEST_SEEDS:
            kpi = run_controller(factory(), profile, seed, EVAL_TICKS)
            kpi.update(controller=name, train_seed=train_seed, profile=profile, seed=seed)
            rows.append(kpi)
    return rows


def _reference_worker(name):
    return name, score_policy(strategy_factory(name))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def nanmean(values):
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else float("nan")


def controller_summary(rows, name):
    """
    Overall mean of every metric for one controller.

    For RL, the mean is taken per training seed first; the reported value is
    the mean of those and `std` is their spread (training-seed variance).
    """
    mine = [r for r in rows if r["controller"] == name]
    seeds = sorted({r["train_seed"] for r in mine}, key=str)
    per_seed = [{k: nanmean([r[k] for r in mine if r["train_seed"] == s]) for k in METRIC_KEYS}
                for s in seeds]
    mean = {k: nanmean([p[k] for p in per_seed]) for k in METRIC_KEYS}
    std = {k: float(np.std([p[k] for p in per_seed])) if len(per_seed) > 1 else 0.0
           for k in METRIC_KEYS}
    return mean, std, per_seed


def per_profile(rows, name, key="avg_delay"):
    return {p: nanmean([r[key] for r in rows if r["controller"] == name and r["profile"] == p])
            for p in PROFILES}


def improvement_vs_fixed(rows, name):
    """Mean over profiles of the % reduction in avg delay vs Fixed Timer."""
    fixed, mine = per_profile(rows, "Fixed Timer"), per_profile(rows, name)
    return sum(100.0 * (fixed[p] - mine[p]) / fixed[p] for p in PROFILES) / len(PROFILES)


def paired_bootstrap_ci(diffs, n_boot=10_000, seed=0):
    """Mean and 95% percentile-bootstrap CI of paired differences."""
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    return float(diffs.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_delay_diffs(rows, a, b):
    """
    avg_delay(a) - avg_delay(b) for each (profile, test seed) pair, averaging
    over training seeds where a controller has several.
    """
    diffs = []
    for p in PROFILES:
        for s in TEST_SEEDS:
            va = nanmean([r["avg_delay"] for r in rows
                          if r["controller"] == a and r["profile"] == p and r["seed"] == s])
            vb = nanmean([r["avg_delay"] for r in rows
                          if r["controller"] == b and r["profile"] == p and r["seed"] == s])
            diffs.append(va - vb)
    return diffs


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt(mean, std=None, digits=1):
    if math.isnan(mean):
        return "n/a"
    if std:
        return f"{mean:.{digits}f} ± {std:.{digits}f}"
    return f"{mean:.{digits}f}"


def write_csv(rows, path):
    fields = ["controller", "train_seed", "profile", "seed"] + METRIC_KEYS
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for r in rows:
            writer.writerow(["" if r[k] is None else
                             round(r[k], 4) if isinstance(r[k], float) else r[k]
                             for k in fields])


def print_tables(rows):
    print("\n" + "=" * 100)
    print(f"OVERALL (mean over {len(PROFILES)} profiles x {len(TEST_SEEDS)} test seeds, "
          f"{EVAL_TICKS} ticks; RL: mean ± std over {len(TRAIN_SEEDS)} training seeds)")
    print("=" * 100)
    cols = [("avg_delay", 1), ("p95_delay", 1), ("max_delay", 1), ("ambulance_delay", 1),
            ("avg_queue", 1), ("throughput", 3)]
    print(f"{'Controller':<13}" + "".join(f"{k:>16}" for k, _ in cols) + f"{'vs Fixed':>10}")
    for name in CONTROLLERS:
        mean, std, _ = controller_summary(rows, name)
        line = f"{name:<13}" + "".join(f"{fmt(mean[k], std[k], d):>16}" for k, d in cols)
        print(line + f"{improvement_vs_fixed(rows, name):>9.1f}%")

    print("\nAVG DELAY PER VEHICLE (s) BY PROFILE")
    print(f"{'Controller':<13}" + "".join(f"{p:>16}" for p in PROFILES))
    for name in CONTROLLERS:
        prof = per_profile(rows, name)
        print(f"{name:<13}" + "".join(f"{prof[p]:>16.1f}" for p in PROFILES))

    mean, lo, hi = paired_bootstrap_ci(paired_delay_diffs(rows, "DQN", "Density"))
    print(f"\nDQN - Density avg delay: {mean:+.2f} s  (95% CI {lo:+.2f} .. {hi:+.2f})")

    print("\nROBUSTNESS / OPTIONS (same trained DQNs)")
    for name in ["DQN"] + list(VARIANT_ROWS):
        mean, std, _ = controller_summary(rows, name)
        print(f"{name:<22}" + "".join(f"{fmt(mean[k], std[k], d):>16}" for k, d in cols))


def make_graphs(rows, training, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available - skipping graphs.")
        return
    os.makedirs(out_dir, exist_ok=True)
    x = np.arange(len(CONTROLLERS))
    summaries = {n: controller_summary(rows, n) for n in CONTROLLERS}

    def bar_chart(key, ylabel, title, filename, color, digits):
        fig, ax = plt.subplots(figsize=(8, 5))
        means = [summaries[n][0][key] for n in CONTROLLERS]
        stds = [summaries[n][1][key] for n in CONTROLLERS]
        ax.bar(x, means, 0.55, yerr=stds, capsize=6, color=color)
        for xi, m in zip(x, means):
            ax.text(xi, m, f"{m:.{digits}f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(CONTROLLERS)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, filename))
        plt.close(fig)

    bar_chart("avg_delay", "Average delay per vehicle (s)",
              "Average delay per vehicle (lower is better; RL error bars = training-seed std)",
              "G1_avg_delay_comparison.png", "#4C72B0", 1)
    bar_chart("throughput", "Throughput (veh/s)",
              "Throughput (higher is better; RL error bars = training-seed std)",
              "G2_throughput_comparison.png", "#55A868", 3)

    # G3: per-profile grouped bars.
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / len(CONTROLLERS)
    px = np.arange(len(PROFILES))
    for i, name in enumerate(CONTROLLERS):
        prof = per_profile(rows, name)
        ax.bar(px + (i - (len(CONTROLLERS) - 1) / 2) * width,
               [prof[p] for p in PROFILES], width, label=name)
    ax.set_xticks(px)
    ax.set_xticklabels(PROFILES)
    ax.set_ylabel("Average delay per vehicle (s)")
    ax.set_title("Average delay per vehicle by traffic profile (lower is better)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "G3_delay_by_profile.png"))
    plt.close(fig)

    # G4: learning curves (one line per training seed) + references.
    if training:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = {"Q-Learning": "#1f77b4", "DQN": "#ff7f0e"}
        for name in RL_AGENTS:
            for i, train_seed in enumerate(TRAIN_SEEDS):
                eps, vals = zip(*training["validation"][(name, train_seed)])
                ax.plot(eps, vals, color=colors[name], alpha=0.75, linewidth=1.3,
                        label=name if i == 0 else None)
        for name, style in (("Fixed Timer", ":"), ("Density", "--")):
            ax.axhline(training["reference"][name], color="gray", linestyle=style,
                       linewidth=1.2, label=f"{name} (reference)")
        ax.set_yscale("log")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Validation avg delay per vehicle (s, log scale)")
        ax.set_title(f"RL learning curves: greedy policy on held-out seeds "
                     f"({len(TRAIN_SEEDS)} training seeds each)")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "G4_rl_training_curves.png"))
        plt.close(fig)
    print(f"Saved graphs to {out_dir}/")


def write_model_card(rows, training, path):
    L = []
    add = L.append
    add("# Dataset & Model Documentation")
    add("")
    add("## Dataset Details")
    add("")
    add("**Source**: Synthetic traffic generated by the project's deterministic, seeded "
        "traffic simulator (no external dataset).")
    add("")
    add("**Traffic scenarios**: " + ", ".join(PROFILES) + ".")
    add("")
    add("**Seeds**: training episodes use seeds from "
        f"{TRAIN_SEEDS} upward, validation uses {list(rl_config.VALIDATION_SEEDS)}, "
        f"testing uses {TEST_SEEDS}. The three sets never overlap.")
    add("")
    add("**Known scenario property**: RUSH_HOUR demand (~3.3 veh/s) exceeds the "
        "intersection's service capacity (~2.7 veh/s), so queues grow under every "
        "controller, and ~1% of vehicles are ambulances (about one every 30 s), so most "
        "of that profile is spent in rule-based emergency preemption. It is kept as an "
        "oversaturated stress test; per-profile results below keep it from dominating.")
    add("")
    add(f"### State features ({rl_config.OBS_DIM}-dimensional observation)")
    add("")
    add("The controller sees the intersection only through a **perception** layer: per "
        "lane, the vehicles it can see and how long each has waited (what a camera with "
        "detection + tracking can provide). Features:")
    add("")
    add("- 16 lanes x 4: vehicle count, mean wait, longest wait, and number of vehicles "
        f"waiting more than {perception_config.LONG_WAIT_THRESHOLD:.0f} s")
    add("- 10 active-phase one-hot values")
    add("- 1  elapsed green time of the active phase")
    add(f"- **{rl_config.OBS_DIM} total**, computed by one function in both training and inference")
    add("")
    add("### Action space (10 discrete actions, extend-or-switch)")
    add("")
    add("- PHASE_1 ... PHASE_10, chosen each time the active green runs out.")
    add(f"- Choosing the active phase extends it by {rl_config.GREEN_EXTENSION:.0f} s; "
        f"choosing another starts it (after yellow) with {rl_config.MIN_GREEN:.0f} s green.")
    add(f"- Green is capped at {rl_config.MAX_GREEN:.0f} s. These are the same bounds "
        "the Density controller uses.")
    add("")
    add("### Reward")
    add("")
    add("```")
    add("cost per tick = sum over queued vehicles of "
        f"(1 + max(0, wait - {rl_config.WAIT_PENALTY_THRESHOLD:.0f}) / {rl_config.WAIT_PENALTY_SLOPE:.0f})"
        " * tick_seconds")
    add(f"reward = -(sum of tick costs over the step) / {rl_config.REWARD_SCALE:.0f}")
    add("```")
    add("")
    add("Every waiting vehicle costs 1 per second, and more the longer *it* has waited "
        f"(3/s at 120 s, 5/s at 180 s). Because the cost is summed over vehicles, many long "
        "waiters cost proportionally more than one. Discount is per simulated second "
        f"(gamma = {rl_config.GAMMA_PER_SECOND} ** step_seconds); episode time limits are "
        "treated as truncation, not termination.")
    add("")
    add("## Models")
    add("")
    add("### 1. Tabular Q-Learning")
    add("")
    add("- State: queue LOW/MED/HIGH per approach (3^4 = 81) x active phase (10) x "
        "elapsed-green level (3) => **2430 states**; Q-table (2430, 10).")
    add("- Q-learning with linearly decaying epsilon-greedy exploration.")
    add("")
    add("### 2. DQN (Deep Q-Network)")
    add("")
    add("```")
    add(f"{rl_config.OBS_DIM} inputs -> 64 (ReLU) -> 64 (ReLU) -> 10 Q-values")
    add("```")
    add("")
    add("- Pure-numpy MLP and Adam optimizer (no deep-learning framework).")
    add("- Experience replay, target network, Double-DQN targets, Huber loss, "
        "gradient-norm clipping.")
    add("")
    add(f"Both: {rl_config.DQN_EPISODES} episodes of {rl_config.EPISODE_LENGTH} ticks, "
        f"trained independently from {len(TRAIN_SEEDS)} seeds {TRAIN_SEEDS}; for each run "
        "the checkpoint with the best validation delay is kept.")
    add("")
    add("## Evaluation Protocol")
    add("")
    add(f"- {len(PROFILES)} profiles x {len(TEST_SEEDS)} test seeds x {EVAL_TICKS} ticks "
        f"({EVAL_TICKS * 0.5 / 60:.0f} simulated minutes) per controller; RL agents are "
        f"evaluated once per training seed ({len(PROFILES) * len(TEST_SEEDS) * len(TRAIN_SEEDS)} "
        "runs each).")
    add("- **Delay per vehicle** = seconds a vehicle spent queued. Vehicles still queued at "
        "the end count with the delay accrued so far, so leaving vehicles unserved is "
        "never rewarded.")
    add("- **Queued veh-s** is the legacy \"average waiting time\": total waiting "
        "time of everyone queued, averaged over ticks (vehicle-seconds, not seconds per vehicle).")
    add("")
    add("## Model Performance Metrics")
    add("")
    add("Mean over all profiles and test seeds. RL: mean ± std across training seeds.")
    add("")
    header = ["Controller"] + [label for _, label, _ in METRICS] + ["Delay cut vs Fixed"]
    add("| " + " | ".join(header) + " |")
    add("|" + "---|" * len(header))
    for name in CONTROLLERS:
        mean, std, _ = controller_summary(rows, name)
        cells = [fmt(mean[k], std[k], 3 if k == "throughput" else 1) for k in METRIC_KEYS]
        add(f"| {name} | " + " | ".join(cells) + f" | {improvement_vs_fixed(rows, name):+.1f}% |")
    add("")
    add("_All metrics lower is better except Throughput. \"Delay cut vs Fixed\" is the "
        "mean, over profiles, of the % reduction in avg delay vs Fixed Timer "
        "(positive = better), so every profile counts equally._")
    add("")
    add("### Average delay per vehicle (s) by profile")
    add("")
    add("| Controller | " + " | ".join(PROFILES) + " |")
    add("|" + "---|" * (len(PROFILES) + 1))
    for name in CONTROLLERS:
        prof = per_profile(rows, name)
        add(f"| {name} | " + " | ".join(f"{prof[p]:.1f}" for p in PROFILES) + " |")
    add("")
    mean, lo, hi = paired_bootstrap_ci(paired_delay_diffs(rows, "DQN", "Density"))
    add("### DQN vs Density")
    add("")
    add(f"Paired difference in avg delay (DQN − Density) over the "
        f"{len(PROFILES) * len(TEST_SEEDS)} (profile, test seed) pairs: "
        f"**{mean:+.2f} s** (95% bootstrap CI {lo:+.2f} to {hi:+.2f} s). "
        + ("The interval excludes 0." if lo > 0 or hi < 0 else
           "The interval includes 0: no significant difference."))
    add("")
    add("### Safety envelope and robustness (same trained DQNs)")
    add("")
    add("| Variant | " + " | ".join(label for _, label, _ in METRICS[:6]) + " |")
    add("|" + "---|" * 7)
    for name in ["DQN"] + list(VARIANT_ROWS):
        mean, std, _ = controller_summary(rows, name)
        add(f"| {name} | " + " | ".join(
            fmt(mean[k], std[k], 3 if k == "throughput" else 1) for k in METRIC_KEYS[:6]) + " |")
    add("")
    add("- **Camera noise**: observations from simulated cameras (5% missed vehicles, 2% "
        "phantom detections per lane, 15-vehicle view limit, 10% wait-estimate error, 5% "
        "lost tracks). Assumed rates, to be calibrated on real footage.")
    add(f"- **Heading intent**: the camera knows a vehicle's turn only from the way it faces. "
        f"Only the first {perception_config.INTENT_VISIBLE_DEPTH} vehicles of a turning lane "
        f"are angled toward their exit (and {perception_config.INTENT_MISREAD_RATE:.0%} of "
        "those are still read as straight); turning vehicles further back are counted as "
        "going straight.")
    add("- **DQN** (the deployed controller) = the learned policy inside a rule-based safety "
        "envelope: never give green to an empty phase while vehicles wait elsewhere; a lane "
        f"with vehicles is served once it has been red {rl_config.SAFETY_MAX_RED:.0f} s; on "
        "camera failure, fixed-order rotation. **DQN (raw policy)** is the learned policy "
        "alone. The envelope was chosen on the edge-case stress test (results/stress_test.md). "
        "Both RL rows in the tables above (Q-Learning and DQN) run inside the envelope.")
    add("")
    add("## Training curves")
    add("")
    if training:
        add(f"Greedy policy scored every {rl_config.VALIDATION_EVERY} episodes on validation "
            f"seeds {list(rl_config.VALIDATION_SEEDS)} ({rl_config.VALIDATION_TICKS} ticks per "
            "profile), avg delay per vehicle (s); best checkpoint kept:")
        add("")
        for name in RL_AGENTS:
            for s in TRAIN_SEEDS:
                pts = training["validation"][(name, s)]
                best_ep, best = min(pts, key=lambda p: p[1])
                add(f"- {name} (seed {s}): {pts[0][1]:.1f} untrained -> best {best:.1f} "
                    f"at episode {best_ep}.")
        for name in RULE_BASED:
            add(f"- {name} on the same validation set: {training['reference'][name]:.1f}.")
    else:
        add("- Agents loaded from models/ (no training this run).")
    add("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"Wrote model card to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    t0 = time.time()
    workers = min(os.cpu_count() or 1, len(RL_AGENTS) * len(TRAIN_SEEDS) + 2)

    print("=" * 100)
    print("CONTROLLED RL EXPERIMENTS")
    print(f"profiles={PROFILES} test_seeds={TEST_SEEDS} eval_ticks={EVAL_TICKS} "
          f"train_seeds={TRAIN_SEEDS}")
    print("=" * 100)

    training = None
    with ProcessPoolExecutor(max_workers=workers) as pool:
        if "--use-saved" not in sys.argv:
            print(f"\n[1/3] Training {len(RL_AGENTS)} agents x {len(TRAIN_SEEDS)} seeds "
                  f"in parallel ({workers} workers)...")
            jobs = [(n, s) for n in RL_AGENTS for s in TRAIN_SEEDS]
            training = {"validation": {}, "reference": {}}
            refs = pool.map(_reference_worker, RULE_BASED)
            for name, seed, history in pool.map(_train_worker, jobs):
                training["validation"][(name, seed)] = history["validation"]
                print(f"  {name} seed {seed}: best checkpoint at episode "
                      f"{history['best_episode']}")
            training["reference"] = dict(refs)
            save_training(training)
        else:
            print("\n[1/3] Using saved agents from models/ ...")
            training = load_training()

        print("\n[2/3] Evaluating controllers...")
        jobs = ([(n, None) for n in RULE_BASED]
                + [(n, s) for n in list(RL_AGENTS) + list(VARIANT_ROWS) for s in TRAIN_SEEDS])
        rows = [row for result in pool.map(_eval_worker, jobs) for row in result]
    print(f"  {len(rows)} evaluation runs")

    print("\n[3/3] Tables, graphs and documentation...")
    print_tables(rows)
    write_csv(rows, os.path.join(OUT_DIR, "results_table.csv"))
    make_graphs(rows, training, IMG_DIR)
    write_model_card(rows, training, os.path.join(OUT_DIR, "model_cards.md"))
    print(f"\nEXPERIMENTS COMPLETE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
