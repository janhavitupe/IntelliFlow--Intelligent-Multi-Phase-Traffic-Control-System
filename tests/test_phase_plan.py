"""
The signal phase plan: safe (no crossing paths), complete, and as designed.
"""
import pytest

from config.phases import all_phase_types, build_emergency_phase, build_phase_plan
from core.enums import PhaseType, SignalState
from core.intersection import Intersection
from evaluation.phase_geometry import conflict, phase_conflicts

PLAN = {
    "PHASE_1": {"North_STRAIGHT", "North_UTURN", "South_STRAIGHT", "South_UTURN",
                "North_LEFT", "South_LEFT", "East_LEFT", "West_LEFT"},
    "PHASE_2": {"North_LEFT", "North_STRAIGHT", "North_RIGHT", "North_UTURN",
                "South_LEFT", "East_LEFT", "West_LEFT", "South_UTURN", "West_UTURN"},
    "PHASE_3": {"South_LEFT", "South_STRAIGHT", "South_RIGHT", "South_UTURN",
                "North_LEFT", "East_LEFT", "West_LEFT", "North_UTURN", "East_UTURN"},
    "PHASE_4": {"East_STRAIGHT", "East_UTURN", "West_STRAIGHT", "West_UTURN",
                "North_LEFT", "South_LEFT", "East_LEFT", "West_LEFT"},
    "PHASE_5": {"East_LEFT", "East_STRAIGHT", "East_RIGHT", "East_UTURN",
                "North_LEFT", "South_LEFT", "West_LEFT", "North_UTURN", "West_UTURN"},
    "PHASE_6": {"West_LEFT", "West_STRAIGHT", "West_RIGHT", "West_UTURN",
                "North_LEFT", "South_LEFT", "East_LEFT", "East_UTURN", "South_UTURN"},
}


@pytest.fixture
def plan():
    return build_phase_plan(Intersection())


def test_plan_matches_design(plan):
    actual = {pt.name: {m.movement_id for m in ph.movements} for pt, ph in plan.items()}
    assert actual == PLAN


@pytest.mark.parametrize("phase_type", all_phase_types(), ids=lambda p: p.name)
def test_no_phase_lets_crossing_paths_go_together(plan, phase_type):
    """The safety property behind the plan: green paths never cross or merge."""
    assert phase_conflicts(plan[phase_type]) == []


def test_conflict_checker_catches_a_known_crossing():
    """A right turn crosses the opposing straight; the checker must say so."""
    assert conflict(("East", "RIGHT"), ("West", "STRAIGHT")) == "cross"
    assert conflict(("North", "STRAIGHT"), ("South", "STRAIGHT")) is None   # opposite straights


def test_plan_has_exactly_the_normal_phases(plan):
    assert list(plan) == all_phase_types()
    assert len(plan) == 6
    assert PhaseType.EMERGENCY_OVERRIDE not in plan


def test_phases_have_no_duplicate_movements(plan):
    for phase in plan.values():
        assert len(phase.movements) == len(set(phase.movements))


def test_every_movement_is_served_by_some_phase(plan):
    intersection = Intersection()
    served = {m.movement_id for ph in plan.values() for m in ph.movements}
    assert served == {m.movement_id for m in intersection.all_movements()}
    assert len(served) == 16


def test_left_turns_are_free_in_every_phase(plan):
    for phase in plan.values():
        lefts = {m.movement_id for m in phase.movements if m.movement_type.name == "LEFT"}
        assert lefts == {"North_LEFT", "South_LEFT", "East_LEFT", "West_LEFT"}


@pytest.mark.parametrize("approach", ["North", "South", "East", "West"])
def test_emergency_phase_is_exactly_one_approach(approach):
    intersection = Intersection()
    phase = build_emergency_phase(intersection, approach)
    assert phase.phase_type == PhaseType.EMERGENCY_OVERRIDE
    assert {m.movement_id for m in phase.movements} == {
        m.movement_id for m in intersection.get_approach(approach)
    }
    assert phase_conflicts(phase) == []


def test_activating_a_phase_greens_only_its_movements():
    intersection = Intersection()
    phase = build_phase_plan(intersection)[PhaseType.PHASE_5]
    phase.activate()
    for movement in intersection.all_movements():
        expected = SignalState.GREEN if movement in phase.movements else SignalState.RED
        assert movement.signal.state == expected, movement.movement_id
