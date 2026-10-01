"""
Ambulance preemption: yellow clearance -> emergency green for the ambulance's
approach -> resume normal scheduling, plus the fail-safe timeout + cooldown.
"""
from core.enums import PhaseType, SignalState
from helpers import TICK, add, ambulance, run


def _start_normal(scheduler):
    scheduler.update(TICK)
    assert scheduler.active_phase_type == PhaseType.PHASE_1


def test_preemption_clears_with_yellow_then_greens_only_ambulance_approach(
    fixed_scheduler, intersection
):
    activations = []
    fixed_scheduler.set_emergency_callback(activations.append)
    _start_normal(fixed_scheduler)
    add(intersection, "East", ambulance())

    fixed_scheduler.update(TICK)                 # detection -> yellow clearance
    assert fixed_scheduler.in_yellow
    assert fixed_scheduler.active_phase_type == PhaseType.PHASE_1
    assert activations == []

    run(fixed_scheduler, 2.0)                    # EMERGENCY_YELLOW_TIME
    assert fixed_scheduler.active_phase_type == PhaseType.EMERGENCY_OVERRIDE
    assert activations == ["East"]
    for movement in intersection.all_movements():
        expected = SignalState.GREEN if movement.approach == "East" else SignalState.RED
        assert movement.signal.state == expected, movement.movement_id


def test_normal_scheduling_resumes_once_ambulance_has_cleared(fixed_scheduler, intersection):
    _start_normal(fixed_scheduler)
    add(intersection, "East", ambulance())
    run(fixed_scheduler, 2.5)
    assert fixed_scheduler.active_phase_type == PhaseType.EMERGENCY_OVERRIDE

    intersection.get_approach("East").straight.lane.remove_front_vehicle()
    fixed_scheduler.update(TICK)                 # ambulance gone -> emergency ends
    assert fixed_scheduler.active_phase_type is None
    fixed_scheduler.update(TICK)                 # strategy picks the next phase
    assert fixed_scheduler.active_phase_type not in (None, PhaseType.EMERGENCY_OVERRIDE)


def test_failsafe_timeout_releases_and_cooldown_prevents_immediate_retrigger(
    fixed_scheduler, intersection
):
    """Regression: the still-queued ambulance used to re-preempt on the next tick."""
    _start_normal(fixed_scheduler)
    add(intersection, "East", ambulance())       # never discharged in this test
    run(fixed_scheduler, 2.5)                    # detection + clearance
    run(fixed_scheduler, fixed_scheduler.emergency_max_timeout)
    assert fixed_scheduler.active_phase_type != PhaseType.EMERGENCY_OVERRIDE

    # Throughout the cooldown normal phases run despite the waiting ambulance.
    for _ in range(round(fixed_scheduler.emergency_cooldown / TICK) - 2):
        fixed_scheduler.update(TICK)
        assert fixed_scheduler.active_phase_type != PhaseType.EMERGENCY_OVERRIDE
        assert not fixed_scheduler._emergency_approach

    # After the cooldown the ambulance is served again.
    run(fixed_scheduler, 2.0 + 2.0)
    assert fixed_scheduler._emergency_approach == "East"


def test_cooldown_only_blocks_the_approach_that_timed_out(fixed_scheduler, intersection):
    _start_normal(fixed_scheduler)
    add(intersection, "East", ambulance())
    run(fixed_scheduler, 2.5 + fixed_scheduler.emergency_max_timeout)
    assert fixed_scheduler._emergency_approach is None

    add(intersection, "North", ambulance())
    fixed_scheduler.update(TICK)
    assert fixed_scheduler._emergency_approach == "North"
