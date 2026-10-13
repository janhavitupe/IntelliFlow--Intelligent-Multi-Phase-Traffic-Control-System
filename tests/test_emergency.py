"""
Ambulance preemption: yellow + all-red clearance -> emergency green for the
ambulance's approach -> yellow + all-red -> resume normal scheduling, plus the
fail-safe timeout + cooldown.
"""
from core.enums import PhaseType, SignalState
from helpers import TICK, add, ambulance, run

CLEARANCE = 4.0            # 2 s yellow + 2 s all-red


def _start_normal(scheduler):
    scheduler.update(TICK)
    assert scheduler.active_phase_type == PhaseType.PHASE_1


def _emergency_green(scheduler):
    return scheduler.active_phase_type == PhaseType.EMERGENCY_OVERRIDE and not scheduler.in_yellow


def test_preemption_clears_then_greens_only_ambulance_approach(fixed_scheduler, intersection):
    activations = []
    fixed_scheduler.set_emergency_callback(activations.append)
    _start_normal(fixed_scheduler)
    add(intersection, "East", ambulance())

    fixed_scheduler.update(TICK)                 # detection -> yellow
    assert fixed_scheduler.in_yellow and fixed_scheduler.active_phase_type == PhaseType.PHASE_1
    run(fixed_scheduler, 2.5)                    # into the all-red
    assert {m.signal.state for m in intersection.all_movements()} == {SignalState.RED}
    assert activations == []

    run(fixed_scheduler, CLEARANCE - 2.5)
    assert _emergency_green(fixed_scheduler)
    assert activations == ["East"]
    for movement in intersection.all_movements():
        expected = SignalState.GREEN if movement.approach == "East" else SignalState.RED
        assert movement.signal.state == expected, movement.movement_id


def test_emergency_green_ends_with_clearance_then_normal_scheduling(fixed_scheduler, intersection):
    _start_normal(fixed_scheduler)
    add(intersection, "East", ambulance())
    run(fixed_scheduler, TICK + CLEARANCE)
    assert _emergency_green(fixed_scheduler)

    intersection.get_approach("East").straight.lane.remove_front_vehicle()
    fixed_scheduler.update(TICK)                 # ambulance gone -> the emergency green yellows out
    assert fixed_scheduler.in_yellow and fixed_scheduler._emergency_approach is None
    run(fixed_scheduler, CLEARANCE)
    assert fixed_scheduler.active_phase_type not in (None, PhaseType.EMERGENCY_OVERRIDE)
    assert not fixed_scheduler.in_yellow


def test_failsafe_timeout_releases_and_cooldown_prevents_immediate_retrigger(
    fixed_scheduler, intersection
):
    """Regression: the still-queued ambulance used to re-preempt on the next tick."""
    _start_normal(fixed_scheduler)
    add(intersection, "East", ambulance())       # never discharged in this test
    run(fixed_scheduler, TICK + CLEARANCE)       # detection + clearance
    run(fixed_scheduler, fixed_scheduler.emergency_max_timeout)
    assert not _emergency_green(fixed_scheduler)

    # Throughout the cooldown normal phases run despite the waiting ambulance.
    for _ in range(round(fixed_scheduler.emergency_cooldown / TICK) - 2):
        fixed_scheduler.update(TICK)
        assert not _emergency_green(fixed_scheduler)
        assert not fixed_scheduler._emergency_approach

    # After the cooldown the ambulance is served again.
    run(fixed_scheduler, 2.0)
    assert fixed_scheduler._emergency_approach == "East"


def test_cooldown_only_blocks_the_approach_that_timed_out(fixed_scheduler, intersection):
    _start_normal(fixed_scheduler)
    add(intersection, "East", ambulance())
    run(fixed_scheduler, TICK + CLEARANCE + fixed_scheduler.emergency_max_timeout)
    assert fixed_scheduler._emergency_approach is None

    add(intersection, "North", ambulance())
    fixed_scheduler.update(TICK)
    assert fixed_scheduler._emergency_approach == "North"


def test_new_ambulance_during_a_clearance_waits_for_it_to_finish(fixed_scheduler, intersection):
    _start_normal(fixed_scheduler)
    add(intersection, "East", ambulance())
    run(fixed_scheduler, TICK + CLEARANCE)
    intersection.get_approach("East").straight.lane.remove_front_vehicle()
    fixed_scheduler.update(TICK)                 # East emergency ends: clearance starts
    add(intersection, "West", ambulance())
    fixed_scheduler.update(TICK)                 # West detected mid-clearance
    assert fixed_scheduler._emergency_approach == "West" and not _emergency_green(fixed_scheduler)
    run(fixed_scheduler, CLEARANCE)
    assert _emergency_green(fixed_scheduler)
