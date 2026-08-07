"""
_validate_density_strategy.py

Validates the PERCENTILE-BASED ADAPTIVE DENSITY controller (Phase 3).

The adaptive controller observes ONLY approach-level queue counts (North /
South / East / West). It never inspects a vehicle's destination_movement.
Green time is a CONTINUOUS discharge interval (Teemo Attacking / interval-
merging intuition), NOT `count x seconds`.

Tests:
  TEST 1  - Highest-ranked approach is selected first.
  TEST 2  - Ranking updates every scheduling cycle.
  TEST 3  - Equal densities behave deterministically.
  TEST 4  - LOW-density approaches never starve.
  TEST 5  - Green duration increases with sustained traffic flow.
  TEST 6  - Green duration always stays within configured limits.
  TEST 7  - Emergency override interrupts adaptive control.
  TEST 8  - Adaptive control resumes after emergency.
  TEST 9  - Fixed seed produces deterministic results.
  TEST 10 - Existing Phase 2 validation suite still passes.
  TEST 11 - Structural check: no destination_movement / no queue mutation /
            no direct signal control in the density strategy.
  TEST 12 - Every normal phase is reachable under at least one traffic
            scenario (no phase is mathematically impossible to select).
"""
import ast
import inspect
import subprocess
import sys

from core.enums import MovementType, PhaseType, Priority, VehicleType
from core.intersection import Intersection
from core.vehicle import Vehicle
from config import density as density_config
from strategies.density_strategy import DensityStrategy
from scheduler.traffic_scheduler import TrafficScheduler
from simulation import Simulation

APPROACHES = ("North", "South", "East", "West")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))
    return passed


def make_intersection(counts):
    """
    Build an Intersection with the given per-approach total queue counts.

    counts: dict {approach_name: total_vehicles}. Vehicles are distributed
    across the four lanes of each approach (Left/Straight/Right/UTurn).
    """
    intersection = Intersection()
    for approach_name, total in counts.items():
        approach = intersection.get_approach(approach_name)
        # Distribute the total across the four movement lanes.
        lanes = list(approach.lanes.values())
        for i in range(total):
            lane = lanes[i % len(lanes)]
            vehicle = Vehicle(
                vehicle_type=VehicleType.CAR,
                arrival_time=intersection.time,
                destination_movement=lane.movement,
                priority=Priority.NORMAL,
            )
            lane.queue._vehicles.append(vehicle)
            vehicle.current_lane = lane
    return intersection


def strategy_decision(strategy, intersection, current_phase=None, time=0.0):
    """Run decide_next_phase and return (phase_type, green, last_decision)."""
    phase_type, green = strategy.decide_next_phase(
        intersection, current_phase, time
    )
    return phase_type, green, strategy.last_decision


def fresh_density():
    """Return a fresh DensityStrategy with default configuration."""
    return DensityStrategy()


def score_phase(strategy, intersection, phase_type, weights):
    """Compute the phase score for a phase given approach weights."""
    phase = strategy._phase_for(intersection, phase_type)
    score = 0.0
    for approach_name in strategy.approach_order:
        coverage = strategy._phase_coverage(phase, approach_name)
        score += weights.get(approach_name, 1) * coverage
    return score


def spawn_ambulance(intersection, approach_name, movement_type):
    """Spawn an ambulance on the given approach/lane (front of queue)."""
    vehicle = Vehicle(
        vehicle_type=VehicleType.AMBULANCE,
        arrival_time=intersection.time,
        destination_movement=movement_type,
        priority=Priority.HIGH,
    )
    lane = intersection.get_approach(approach_name).get_lane(movement_type)
    lane.queue._vehicles.appendleft(vehicle)
    vehicle.current_lane = lane
    return vehicle


# ---------------------------------------------------------------------------
# TEST 1 : Highest-ranked approach is selected first
# ---------------------------------------------------------------------------
# East has the most vehicles -> the selected phase must serve East.
intersection = make_intersection({
    "North": 18, "South": 44, "East": 72, "West": 31,
})
strategy = fresh_density()
phase_type, green, decision = strategy_decision(strategy, intersection)

served_east = "East" in decision["served_approaches"]
rank1 = decision["rankings"][1]
test1 = (
    rank1 == "East"
    and decision["densities"]["East"] == "HIGH"
    and served_east
)
results = []
results.append(check(
    "TEST 1 Highest-ranked approach selected first",
    test1,
    f"rank1={rank1} selected={decision['selected_phase']} "
    f"served={decision['served_approaches']}",
))


# ---------------------------------------------------------------------------
# TEST 2 : Ranking updates every scheduling cycle
# ---------------------------------------------------------------------------
# Same intersection, two different queue states -> rankings must differ.
intersection_a = make_intersection({
    "North": 60, "South": 10, "East": 20, "West": 5,
})
intersection_b = make_intersection({
    "North": 5, "South": 60, "East": 20, "West": 10,
})
strategy_a = fresh_density()
strategy_b = fresh_density()
_, _, decision_a = strategy_decision(strategy_a, intersection_a)
_, _, decision_b = strategy_decision(strategy_b, intersection_b)

rankings_a = decision_a["rankings"]
rankings_b = decision_b["rankings"]
test2 = rankings_a != rankings_b and rankings_a[1] == "North" and rankings_b[1] == "South"
results.append(check(
    "TEST 2 Ranking updates every scheduling cycle",
    test2,
    f"a={rankings_a} b={rankings_b}",
))


# ---------------------------------------------------------------------------
# TEST 3 : Equal densities behave deterministically
# ---------------------------------------------------------------------------
# All four approaches have identical counts -> tie-break is deterministic.
intersection = make_intersection({
    "North": 25, "South": 25, "East": 25, "West": 25,
})
strategy = fresh_density()
phase_type, green, decision = strategy_decision(strategy, intersection)
rankings = decision["rankings"]

expected_order = ["North", "South", "East", "West"]  # APPROACH_ORDER
tie_ok = [rankings[i] for i in range(1, 5)] == expected_order

# Two fresh strategies on identical intersections must produce identical
# decisions (same selected phase, same rankings, same green).
strategy2 = fresh_density()
phase2, green2, decision2 = strategy_decision(strategy2, make_intersection({
    "North": 25, "South": 25, "East": 25, "West": 25,
}))
deterministic = (
    phase2 == phase_type
    and green2 == green
    and decision2["rankings"] == rankings
)
results.append(check(
    "TEST 3 Equal densities behave deterministically",
    tie_ok and deterministic,
    f"order={expected_order} deterministic={deterministic}",
))


# ---------------------------------------------------------------------------
# TEST 4 : LOW-density approaches never starve
# ---------------------------------------------------------------------------
# Repeatedly give one approach massive demand and others small demand; the
# LOW-density approaches must still eventually receive service (fairness).
strategy = fresh_density()
# Manually drive starvation: mark West as starved by setting its counter.
strategy._starvation_cycles["West"] = strategy.max_starvation_cycles

intersection = make_intersection({
    "North": 80, "South": 80, "East": 80, "West": 5,
})
phase_type, green, decision = strategy_decision(strategy, intersection)
west_served = "West" in decision["served_approaches"]
fairness_fired = decision["fairness_active"]
test4 = fairness_fired and west_served
results.append(check(
    "TEST 4 LOW-density approaches never starve",
    test4,
    f"fairness_active={fairness_fired} west_served={west_served} "
    f"selected={decision['selected_phase']}",
))


# ---------------------------------------------------------------------------
# TEST 5 : Green duration increases with sustained traffic flow
# ---------------------------------------------------------------------------
# More sustained queued traffic -> longer green, but never > max_green.
strategy = fresh_density()

light = make_intersection({
    "North": 5, "South": 5, "East": 5, "West": 5,
})
heavy = make_intersection({
    "North": 50, "South": 50, "East": 50, "West": 50,
})
_, green_light, _ = strategy_decision(strategy, light)
_, green_heavy, _ = strategy_decision(strategy, heavy)

test5 = green_heavy > green_light
results.append(check(
    "TEST 5 Green duration increases with sustained traffic flow",
    test5,
    f"light={green_light}s heavy={green_heavy}s",
))


# ---------------------------------------------------------------------------
# TEST 6 : Green duration always stays within configured limits
# ---------------------------------------------------------------------------
strategy = fresh_density()
limits_ok = True
detail = ""
for total in range(0, 201, 10):
    intersection = make_intersection({
        "North": total, "South": total, "East": total, "West": total,
    })
    _, green, _ = strategy_decision(strategy, intersection)
    if not (strategy.min_green <= green <= strategy.max_green):
        limits_ok = False
        detail += f"total={total} green={green}; "
        break
results.append(check(
    "TEST 6 Green duration within configured limits",
    limits_ok,
    detail or f"all greens in [{strategy.min_green}, {strategy.max_green}]",
))


# ---------------------------------------------------------------------------
# TEST 7 : Emergency override interrupts adaptive control
# ---------------------------------------------------------------------------
intersection = Intersection()
strategy = fresh_density()
scheduler = TrafficScheduler(
    intersection, strategy,
    yellow_duration=2.0,
    emergency_yellow_duration=2.0,
    emergency_max_timeout=30.0,
)
# Start normal adaptive scheduling.
scheduler.update(1.0)
assert scheduler.active_phase_type != PhaseType.EMERGENCY_OVERRIDE

# Spawn an ambulance on North -> emergency preemption begins.
spawn_ambulance(intersection, "North", MovementType.STRAIGHT)
scheduler.update(1.0)                    # begin emergency (yellow clearance)
scheduler._emergency_clearance_remaining = 0.0
scheduler._advance_emergency(1.0)        # emergency green active

test7 = scheduler.active_phase_type == PhaseType.EMERGENCY_OVERRIDE
results.append(check(
    "TEST 7 Emergency override interrupts adaptive control",
    test7,
    f"phase={scheduler.active_phase_type}",
))


# ---------------------------------------------------------------------------
# TEST 8 : Adaptive control resumes after emergency
# ---------------------------------------------------------------------------
# (Continue from TEST 7's scheduler state.)
# Remove the ambulance -> emergency ends -> adaptive scheduling resumes.
north = intersection.get_approach("North")
for lane in north.lanes.values():
    lane.queue._vehicles.clear()
    lane.queue._total_waiting_time = 0.0
scheduler._advance_emergency(1.0)        # end emergency
ok_ended = scheduler._emergency_approach is None

scheduler.update(1.0)                    # resume normal scheduling
ok_resumed = (
    scheduler.active_phase_type is not None
    and scheduler.active_phase_type != PhaseType.EMERGENCY_OVERRIDE
)
# The resumed phase must be a decision from the adaptive strategy.
ok_adaptive = strategy.last_decision is not None
results.append(check(
    "TEST 8 Adaptive control resumes after emergency",
    ok_ended and ok_resumed and ok_adaptive,
    f"emergency_ended={ok_ended} resumed={scheduler.active_phase_type} "
    f"adaptive_decision={ok_adaptive}",
))


# ---------------------------------------------------------------------------
# TEST 9 : Fixed seed produces deterministic results
# ---------------------------------------------------------------------------
def run_density_seeded(max_ticks, seed=42):
    sim = Simulation(
        profile_key="RUSH_HOUR",
        max_ticks=max_ticks,
        live=False,
        seed=seed,
        strategy_key="density",
    )
    for _ in range(max_ticks):
        sim.step()
    return sim


sim_a = run_density_seeded(60, seed=7)
sim_b = run_density_seeded(60, seed=7)
test9 = (
    sim_a.analytics.summary() == sim_b.analytics.summary()
    and sim_a.analytics.adaptive_decision_count > 0
)
results.append(check(
    "TEST 9 Fixed seed produces deterministic results",
    test9,
    f"adaptive_decisions={sim_a.analytics.adaptive_decision_count} "
    f"repro={sim_a.analytics.summary() == sim_b.analytics.summary()}",
))


# ---------------------------------------------------------------------------
# TEST 10 : Existing Phase 2 validation suite still passes
# ---------------------------------------------------------------------------
# Run the Phase 2 regression in a subprocess. The density strategy is NOT the
# default, so the existing suite runs with the preserved FixedTimer strategy.
proc = subprocess.run(
    [sys.executable, "_validate_simulations.py"],
    capture_output=True,
    text=True,
    cwd=".",
)
phase2_ok = proc.returncode == 0
results.append(check(
    "TEST 10 Existing Phase 2 validation suite still passes",
    phase2_ok,
    "returncode=0" if phase2_ok else proc.stdout[-500:],
))


# ---------------------------------------------------------------------------
# TEST 11 : Structural check (controller boundary)
# ---------------------------------------------------------------------------
# The density strategy must:
#   - never reference destination_movement in EXECUTABLE code
#   - never call queue mutation methods (enqueue/dequeue/remove_front_vehicle)
#   - never call signal control methods (activate/deactivate/turn_green/...)
#
# This check is AST-based so prose in docstrings/comments (which legitimately
# describe the controller observation boundary) is NOT treated as a violation.
source = inspect.getsource(DensityStrategy)

# Method names that would violate the controller boundary if called from the
# strategy. The strategy may READ queue lengths / phase geometry but must never
# mutate queues or flip signals.
_QUEUE_MUTATORS = {
    "enqueue", "dequeue", "remove_front_vehicle",
    "append", "appendleft", "popleft", "clear", "insert", "extend",
}
_SIGNAL_CONTROLS = {
    "activate", "deactivate", "turn_green", "turn_red", "turn_yellow",
}


def _structural_violations(src: str) -> list:
    """Return a list of controller-boundary violations in the strategy source."""
    tree = ast.parse(src)
    found = []
    for node in ast.walk(tree):
        # Attribute access to a forbidden symbol (executable code only).
        if isinstance(node, ast.Attribute):
            if node.attr == "destination_movement":
                found.append("attribute 'destination_movement'")
            if node.attr in _QUEUE_MUTATORS:
                found.append(f"queue mutation call: {node.attr}")
            if node.attr in _SIGNAL_CONTROLS:
                found.append(f"signal control call: {node.attr}")
        # Bare name reference (e.g. an imported or assigned variable).
        if isinstance(node, ast.Name) and node.id == "destination_movement":
            found.append("name 'destination_movement'")
    return sorted(set(found))


violations = _structural_violations(source)
test11 = not violations
results.append(check(
    "TEST 11 Structural check (no destination_movement / no queue mutation "
    "/ no signal control)",
    test11,
    f"violations={violations}" if violations else "clean",
))


# ---------------------------------------------------------------------------
# TEST 12 : Every normal phase is reachable under some traffic scenario
# ---------------------------------------------------------------------------
# Some phases in the preserved 10-phase plan are coverage-dominated by a
# superset phase (e.g. PHASE_1 movements are a strict subset of PHASE_8 with
# lower West coverage). A purely coverage-based score could never select such
# phases. The phase-recency anti-starvation mechanism solves this: a phase
# that goes MAX_PHASE_STARVATION_CYCLES consecutive cycles without selection
# receives a growing bonus (PHASE_RECENCY_BONUS x recency), so it eventually
# out-scores every recently-used phase.
#
# This test proves every normal phase is reachable: for each phase, drive its
# recency far past the threshold and verify it becomes the selected phase.
phase_order = [
    PhaseType.PHASE_1, PhaseType.PHASE_2, PhaseType.PHASE_3,
    PhaseType.PHASE_4, PhaseType.PHASE_5, PhaseType.PHASE_6,
    PhaseType.PHASE_7, PhaseType.PHASE_8, PhaseType.PHASE_9,
    PhaseType.PHASE_10,
]

reachable = []
unreachable = []
intersection = Intersection()  # empty; only phase geometry is used
strategy = fresh_density()
assert strategy.phase_recency_enabled, "phase-recency must be enabled for TEST 12"

for pt in phase_order:
    strategy.reset()
    # Reset every phase's recency, then starve ONLY the target phase far past
    # the threshold so its recency bonus dominates all coverage scores.
    for name in strategy._phase_recency:
        strategy._phase_recency[name] = 0
    strategy._phase_recency[pt.name] = strategy.max_phase_starvation_cycles * 10

    phase_type, green, decision = strategy_decision(strategy, intersection)
    if phase_type == pt and decision.get("phase_recency_active"):
        reachable.append(pt)
    else:
        unreachable.append(pt)

test12 = len(reachable) == 10 and len(unreachable) == 0
results.append(check(
    "TEST 12 Every normal phase reachable under some scenario",
    test12,
    f"reachable={[p.name for p in reachable]}"
    if test12 else f"unreachable={[p.name for p in unreachable]}",
))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("DENSITY STRATEGY VALIDATION SUMMARY")
print("=" * 60)
passed = sum(results)
total = len(results)
for i, r in enumerate(results, 1):
    print(f"  Test {i}: {'PASS' if r else 'FAIL'}")
print(f"{passed}/{total} checks passed")
raise SystemExit(0 if passed == total else 1)

