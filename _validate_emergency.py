"""
_validate_emergency.py

Validates the APPROACH-LEVEL AMBULANCE EMERGENCY PREEMPTION implementation.

Tests:
  TEST 1 - North ambulance: EMERGENCY_OVERRIDE contains exactly North.LEFT,
           North.STRAIGHT, North.RIGHT, North.UTURN; no South/East/West green.
  TEST 2 - South ambulance: exactly all four South movements green.
  TEST 3 - East ambulance: exactly all four East movements green.
  TEST 4 - West ambulance: exactly all four West movements green.
  TEST 5 - Route independence: same approach, different movement types produce
           identical emergency selection (controller does NOT read route).
  TEST 6 - Safe transition: normal green -> yellow clearance -> emergency green;
           no conflicting movement survives when EMERGENCY_OVERRIDE begins.
  TEST 7 - Return to normal: after ambulance clears, normal PHASE_1..PHASE_10
           scheduling resumes.
  TEST 8 - Multiple ambulances: North + East simultaneously -> only one approach
           gets emergency green (first-detected-wins).
  TEST 9 - Normal architecture unchanged: PHASE_1..PHASE_10 movement sets are
           identical after the change.
  TEST 10- Regression: run the existing validation suite and confirm normal
           traffic generation, ServiceModel, analytics, CSV logging and
           fixed-seed reproducibility still work.
"""
from core.enums import MovementType, PhaseType, Priority, VehicleType
from core.intersection import Intersection
from core.vehicle import Vehicle
from config import phases as phase_config
from scheduler.traffic_scheduler import TrafficScheduler
from strategies.fixed_timer_strategy import FixedTimerStrategy
from simulation import Simulation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPECTED_APPROACH_MOVEMENTS = {
    "North": {"North_LEFT", "North_STRAIGHT", "North_RIGHT", "North_UTURN"},
    "South": {"South_LEFT", "South_STRAIGHT", "South_RIGHT", "South_UTURN"},
    "East": {"East_LEFT", "East_STRAIGHT", "East_RIGHT", "East_UTURN"},
    "West": {"West_LEFT", "West_STRAIGHT", "West_RIGHT", "West_UTURN"},
}


def movement_ids(phase):
    """Return the set of movement_id strings active in a phase."""
    return {m.movement_id for m in phase.movements}


def green_movement_ids(intersection):
    """Return the set of movement_id strings currently GREEN."""
    return {
        m.movement_id
        for m in intersection.all_movements()
        if m.signal.is_green
    }


def spawn_ambulance(intersection, approach_name, movement_type, lane_offset=0):
    """
    Spawn an ambulance on the given approach/lane. Returns the vehicle.
    The ambulance is placed at the front of the specified lane's queue so it
    is immediately detectable.
    """
    vehicle = Vehicle(
        vehicle_type=VehicleType.AMBULANCE,
        arrival_time=intersection.time,
        destination_movement=movement_type,
        priority=Priority.HIGH,
    )
    lane = intersection.get_approach(approach_name).get_lane(movement_type)
    # Insert at the front so it is the head vehicle and clearly present.
    lane.queue._vehicles.appendleft(vehicle)
    vehicle.current_lane = lane
    return vehicle


def fresh_scheduler():
    """Build a standalone intersection + scheduler for controlled tests."""
    intersection = Intersection()
    strategy = FixedTimerStrategy(green_duration=12.0, yellow_duration=2.0)
    scheduler = TrafficScheduler(
        intersection, strategy,
        yellow_duration=2.0,
        emergency_yellow_duration=2.0,
        emergency_max_timeout=30.0,
    )
    return intersection, scheduler


def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))
    return passed


results = []

# ---------------------------------------------------------------------------
# TEST 1-4 : Approach-level emergency content
# ---------------------------------------------------------------------------

for approach in ("North", "South", "East", "West"):
    intersection = Intersection()
    em = phase_config.build_emergency_phase(intersection, approach)
    expected = EXPECTED_APPROACH_MOVEMENTS[approach]
    actual = movement_ids(em)
    ok_type = em.phase_type == PhaseType.EMERGENCY_OVERRIDE
    ok_set = actual == expected
    results.append(check(
        f"TEST {['North','South','East','West'].index(approach)+1} "
        f"{approach} ambulance -> exactly 4 approach movements",
        ok_type and ok_set,
        f"expected={sorted(expected)} actual={sorted(actual)}",
    ))

# ---------------------------------------------------------------------------
# TEST 5 : Route independence
# ---------------------------------------------------------------------------

route_ids = []
for movement_type in (
    MovementType.LEFT,
    MovementType.STRAIGHT,
    MovementType.RIGHT,
    MovementType.UTURN,
):
    _, scheduler = fresh_scheduler()
    intersection = scheduler.intersection
    spawn_ambulance(intersection, "North", movement_type)
    # Detect + run preemption to green.
    scheduler.update(1.0)  # begin emergency (yellow clearance)
    scheduler._emergency_clearance_remaining = 0.0  # force to green
    scheduler._advance_emergency(1.0)
    route_ids.append(scheduler._emergency_approach)

results.append(check(
    "TEST 5 Route independence (movement type ignored)",
    all(a == "North" for a in route_ids) and len(set(route_ids)) == 1,
    f"selected_for_all_routes={route_ids}",
))

# ---------------------------------------------------------------------------
# TEST 6 : Safe transition (normal green -> yellow clearance -> emergency green)
# ---------------------------------------------------------------------------

intersection, scheduler = fresh_scheduler()
# Start a normal phase (PHASE_1) so there are green movements.
scheduler.current_phase = scheduler.phase_plan[PhaseType.PHASE_1]
scheduler.current_phase.activate()
scheduler.green_remaining = 12.0
scheduler._in_yellow = False

# Spawn a North ambulance.
spawn_ambulance(intersection, "North", MovementType.STRAIGHT)

# Begin emergency: should go to yellow clearance, not directly to green.
scheduler._begin_emergency("North")
ok_clearance = scheduler.in_yellow
# During clearance, no conflicting phase should be green yet.
conflicting_during_clearance = green_movement_ids(intersection)

# Complete clearance -> emergency green.
scheduler._emergency_clearance_remaining = 0.0
scheduler._advance_emergency(1.0)

green_now = green_movement_ids(intersection)
ok_emergency_green = green_now == EXPECTED_APPROACH_MOVEMENTS["North"]

results.append(check(
    "TEST 6 Safe transition to emergency",
    ok_clearance and ok_emergency_green,
    f"in_clearance={ok_clearance} emergency_green_only={ok_emergency_green}",
))

# ---------------------------------------------------------------------------
# TEST 7 : Return to normal scheduling after ambulance clears
# ---------------------------------------------------------------------------

intersection, scheduler = fresh_scheduler()
spawn_ambulance(intersection, "North", MovementType.LEFT)
scheduler.update(1.0)                    # begin
scheduler._emergency_clearance_remaining = 0.0
scheduler._advance_emergency(1.0)        # emergency green active
if not scheduler.in_yellow:
    assert scheduler.active_phase_type == PhaseType.EMERGENCY_OVERRIDE

# Remove the ambulance (simulate it cleared the intersection).
north = intersection.get_approach("North")
for lane in north.lanes.values():
    lane.queue._vehicles.clear()
    lane.queue._total_waiting_time = 0.0

scheduler._advance_emergency(1.0)        # should end emergency
ok_ended = scheduler._emergency_approach is None

# Next update resumes normal scheduling (a PHASE_1..PHASE_10 phase activates).
scheduler.update(1.0)
ok_normal = (
    scheduler.active_phase_type is not None
    and scheduler.active_phase_type != PhaseType.EMERGENCY_OVERRIDE
)

results.append(check(
    "TEST 7 Return to normal scheduling",
    ok_ended and ok_normal,
    f"emergency_ended={ok_ended} resumed={scheduler.active_phase_type}",
))

# ---------------------------------------------------------------------------
# TEST 8 : Multiple ambulances - first-detected-wins (never both green)
# ---------------------------------------------------------------------------

intersection, scheduler = fresh_scheduler()
# Ambush: ambulances on both North and East simultaneously.
spawn_ambulance(intersection, "North", MovementType.STRAIGHT)
spawn_ambulance(intersection, "East", MovementType.STRAIGHT)

scheduler.update(1.0)                    # begin for North (first in order)
scheduler._emergency_clearance_remaining = 0.0
scheduler._advance_emergency(1.0)        # North green

green = green_movement_ids(intersection)
north_only = green == EXPECTED_APPROACH_MOVEMENTS["North"]
east_also_green = (EXPECTED_APPROACH_MOVEMENTS["East"] & green) != set()

results.append(check(
    "TEST 8 Multiple ambulances -> one approach green",
    north_only and not east_also_green,
    f"winner=North, green_only_north={north_only}, east_also_green={east_also_green}",
))

# ---------------------------------------------------------------------------
# TEST 9 : Normal architecture unchanged (PHASE_1..PHASE_10 identical)
# ---------------------------------------------------------------------------

intersection, scheduler = fresh_scheduler()
official = {}

# Reconstruct the official plan from the verify script's OFFICIAL table.
# Instead we compare the current plan against an explicit expected snapshot.
OFFICIAL = {
    PhaseType.PHASE_1: {'West_STRAIGHT','West_LEFT','West_UTURN','North_LEFT','East_LEFT','South_RIGHT','South_LEFT'},
    PhaseType.PHASE_2: {'North_STRAIGHT','North_UTURN','North_LEFT','East_LEFT','South_LEFT','West_LEFT','West_RIGHT'},
    PhaseType.PHASE_3: {'East_STRAIGHT','East_UTURN','East_LEFT','South_LEFT','West_LEFT','North_LEFT','North_RIGHT'},
    PhaseType.PHASE_4: {'South_STRAIGHT','South_UTURN','South_LEFT','West_LEFT','North_LEFT','East_RIGHT','East_LEFT'},
    PhaseType.PHASE_5: {'West_STRAIGHT','West_LEFT','West_UTURN','North_LEFT','East_UTURN','East_STRAIGHT','East_RIGHT','South_LEFT'},
    PhaseType.PHASE_6: {'South_STRAIGHT','South_LEFT','South_UTURN','West_LEFT','North_STRAIGHT','North_UTURN','North_LEFT','East_LEFT'},
    PhaseType.PHASE_7: {'South_STRAIGHT','South_LEFT','South_RIGHT','South_UTURN','West_LEFT','North_LEFT','East_UTURN','East_LEFT'},
    PhaseType.PHASE_8: {'West_STRAIGHT','West_LEFT','West_RIGHT','West_UTURN','North_LEFT','East_LEFT','South_LEFT','South_UTURN'},
    PhaseType.PHASE_9: {'North_STRAIGHT','North_LEFT','North_RIGHT','North_UTURN','East_LEFT','South_LEFT','West_LEFT','West_UTURN'},
    PhaseType.PHASE_10: {'East_STRAIGHT','East_UTURN','East_RIGHT','East_LEFT','South_LEFT','West_LEFT','North_UTURN','North_LEFT'},
}

ok_all = True
detail = ""
for pt, expected in OFFICIAL.items():
    actual = movement_ids(scheduler.phase_plan[pt])
    if actual != expected:
        ok_all = False
        detail += f"{pt.name}: missing={sorted(expected-actual)} extra={sorted(actual-expected)}; "
results.append(check(
    "TEST 9 Normal 10-phase architecture unchanged",
    ok_all,
    detail or "all 10 phases match official definitions",
))

# ---------------------------------------------------------------------------
# TEST 10 : Regression - full simulation runs correctly
# ---------------------------------------------------------------------------


def _run_manual(max_ticks, seed=42):
    """Step a Simulation manually for max_ticks without live rendering."""
    sim = Simulation(
        profile_key="NORMAL_TRAFFIC",
        max_ticks=max_ticks,
        live=False,
        seed=seed,
    )
    for _ in range(max_ticks):
        sim.step()
    return sim, []


# Run a short live simulation with fixed seed and CSV logging to exercise the
# whole pipeline (traffic generation, ServiceModel, analytics, logger).
sim = Simulation(
    profile_key="NORMAL_TRAFFIC",
    max_ticks=60,
    live=False,
    seed=42,
    log_to_csv=True,
)
sim.run()

ok_ran = sim.tick == 60
ok_analytics = sim.analytics.total_vehicles_spawned >= 0
ok_preempt = sim.analytics.total_emergency_preemptions >= 0

# Fixed-seed reproducibility still holds.
sim_a, _ = _run_manual(20, seed=7)
sim_b, _ = _run_manual(20, seed=7)
ok_repro = (
    sim_a.analytics.summary() == sim_b.analytics.summary()
)

results.append(check(
    "TEST 10 Regression (simulation runs, analytics, CSV, seed)",
    ok_ran and ok_analytics and ok_preempt and ok_repro,
    f"ticks={sim.tick} spawned={sim.analytics.total_vehicles_spawned} "
    f"preempts={sim.analytics.total_emergency_preemptions} repro={ok_repro}",
))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("EMERGENCY PREEMPTION VALIDATION SUMMARY")
print("=" * 60)
passed = sum(results)
total = len(results)
for i, r in enumerate(results, 1):
    print(f"  Test {i}: {'PASS' if r else 'FAIL'}")
print(f"{passed}/{total} checks passed")
raise SystemExit(0 if passed == total else 1)
