"""Per-vehicle delay KPIs and the experiment harness's statistics."""
import os
import subprocess
import sys

import pytest

from analytics.statistics import Statistics
from core.enums import MovementType, VehicleType
import run_experiments as rx
from helpers import TICK, add, ambulance, car


def test_delay_counts_served_and_still_queued_vehicles(intersection):
    stats = Statistics(intersection, interval=TICK)
    movement = intersection.get_approach("North").straight
    stats.record_served([(movement, car(waiting=10.0)), (movement, car(waiting=20.0))])
    add(intersection, "East", car(waiting=60.0))          # never served

    assert sorted(stats.vehicle_delays()) == [10.0, 20.0, 60.0]
    assert stats.average_delay == pytest.approx(30.0)
    assert stats.max_delay == 60.0


def test_p95_delay(intersection):
    stats = Statistics(intersection, interval=TICK)
    movement = intersection.get_approach("North").straight
    stats.record_served([(movement, car(waiting=float(w))) for w in range(1, 101)])
    assert stats.p95_delay == 96.0                        # index 95 of 1..100


def test_ambulance_delay_is_tracked_separately(intersection):
    stats = Statistics(intersection, interval=TICK)
    assert stats.ambulance_average_delay is None
    movement = intersection.get_approach("North").straight
    amb = ambulance()
    amb.waiting_time = 4.0
    stats.record_served([(movement, amb), (movement, car(waiting=50.0))])
    queued = add(intersection, "West", ambulance(), MovementType.LEFT)
    queued.waiting_time = 8.0
    assert stats.ambulance_average_delay == pytest.approx(6.0)


def test_empty_statistics_are_zero(intersection):
    stats = Statistics(intersection, interval=TICK)
    assert stats.average_delay == stats.p95_delay == stats.max_delay == 0.0


def test_paired_bootstrap_ci_brackets_the_mean():
    mean, lo, hi = rx.paired_bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0] * 5)
    assert mean == pytest.approx(3.0)
    assert lo < mean < hi
    assert rx.paired_bootstrap_ci([2.0] * 10) == pytest.approx((2.0, 2.0, 2.0))


def _row(controller, profile, delay, train_seed=None, seed=1):
    row = {k: 0.0 for k in rx.METRIC_KEYS}
    row.update(controller=controller, profile=profile, seed=seed,
               train_seed=train_seed, avg_delay=delay)
    return row


def test_rl_summary_reports_training_seed_spread():
    rows = [_row("DQN", p, 10.0, train_seed=1) for p in rx.PROFILES] + \
           [_row("DQN", p, 20.0, train_seed=2) for p in rx.PROFILES]
    mean, std, per_seed = rx.controller_summary(rows, "DQN")
    assert mean["avg_delay"] == pytest.approx(15.0)
    assert std["avg_delay"] == pytest.approx(5.0)
    assert [p["avg_delay"] for p in per_seed] == [10.0, 20.0]


def test_improvement_vs_fixed_weights_every_profile_equally():
    # A huge absolute gain on one profile must not swamp the others.
    fixed = {p: 10.0 for p in rx.PROFILES}
    fixed["RUSH_HOUR"] = 1000.0
    other = dict(fixed, RUSH_HOUR=500.0)               # -50% on one profile only
    rows = [_row("Fixed Timer", p, d) for p, d in fixed.items()] + \
           [_row("Density", p, d) for p, d in other.items()]
    assert rx.improvement_vs_fixed(rows, "Density") == pytest.approx(50.0 / len(rx.PROFILES))


def test_harness_runs_end_to_end_in_quick_mode(tmp_path):
    env = dict(os.environ, EXPERIMENTS_QUICK="1", EXPERIMENTS_OUT=str(tmp_path))
    result = subprocess.run([sys.executable, "run_experiments.py"], env=env,
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]
    for rel in ["results/results_table.csv", "results/model_cards.md",
                "images/G1_avg_delay_comparison.png", "images/G4_rl_training_curves.png",
                "models/dqn_seed42.npz", "models/q_table_seed7.npy"]:
        assert (tmp_path / rel).exists(), rel

    card = (tmp_path / "results/model_cards.md").read_text(encoding="utf-8")
    csv_first = (tmp_path / "results/results_table.csv").read_bytes()

    saved = subprocess.run([sys.executable, "run_experiments.py", "--use-saved"], env=env,
                           capture_output=True, text=True, timeout=600)
    assert saved.returncode == 0, saved.stderr[-2000:]
    # Saved agents + saved training history reproduce the outputs exactly.
    assert (tmp_path / "results/results_table.csv").read_bytes() == csv_first
    assert (tmp_path / "results/model_cards.md").read_text(encoding="utf-8") == card
