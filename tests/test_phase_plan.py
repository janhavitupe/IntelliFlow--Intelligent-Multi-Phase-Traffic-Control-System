"""
The official 10-phase signal plan.

The plan is pinned movement-for-movement: it comes from the official phase
diagrams, so any change to it must be deliberate (update OFFICIAL_PLAN in
the same commit). There is no conflict matrix in the codebase, so movement
compatibility itself is not re-derived here.
"""
import pytest

from config.phases import all_phase_types, build_phase_plan, build_emergency_phase
from core.enums import PhaseType, SignalState
from core.intersection import Intersection

OFFICIAL_PLAN = {
    "PHASE_1": {"West_STRAIGHT", "West_LEFT", "West_UTURN", "North_LEFT", "East_LEFT",
                "South_RIGHT", "South_LEFT"},
    "PHASE_2": {"North_STRAIGHT", "North_UTURN", "North_LEFT", "East_LEFT", "South_LEFT",
                "West_LEFT", "West_RIGHT"},
    "PHASE_3": {"East_STRAIGHT", "East_UTURN", "East_LEFT", "South_LEFT", "West_LEFT",
                "North_LEFT", "North_RIGHT"},
    "PHASE_4": {"South_STRAIGHT", "South_UTURN", "South_LEFT", "West_LEFT", "North_LEFT",
                "East_RIGHT", "East_LEFT"},
    "PHASE_5": {"West_STRAIGHT", "West_LEFT", "West_UTURN", "North_LEFT", "East_UTURN",
                "East_STRAIGHT", "East_RIGHT", "South_LEFT"},
    "PHASE_6": {"South_STRAIGHT", "South_LEFT", "South_UTURN", "West_LEFT", "North_STRAIGHT",
                "North_UTURN", "North_LEFT", "East_LEFT"},
    "PHASE_7": {"South_STRAIGHT", "South_LEFT", "South_RIGHT", "South_UTURN", "West_LEFT",
                "North_LEFT", "East_UTURN", "East_LEFT"},
    "PHASE_8": {"West_STRAIGHT", "West_LEFT", "West_RIGHT", "West_UTURN", "North_LEFT",
                "East_LEFT", "South_LEFT", "South_UTURN"},
    "PHASE_9": {"North_STRAIGHT", "North_LEFT", "North_RIGHT", "North_UTURN", "East_LEFT",
                "South_LEFT", "West_LEFT", "West_UTURN"},
    "PHASE_10": {"East_STRAIGHT", "East_UTURN", "East_RIGHT", "East_LEFT", "South_LEFT",
                 "West_LEFT", "North_UTURN", "North_LEFT"},
}


@pytest.fixture
def plan():
    return build_phase_plan(Intersection())


def test_plan_matches_official_diagrams(plan):
    actual = {pt.name: {m.movement_id for m in ph.movements} for pt, ph in plan.items()}
    assert actual == OFFICIAL_PLAN


def test_plan_has_exactly_the_ten_normal_phases(plan):
    assert list(plan) == all_phase_types()
    assert len(plan) == 10
    assert PhaseType.EMERGENCY_OVERRIDE not in plan


def test_phases_have_no_duplicate_movements(plan):
    for phase in plan.values():
        assert len(phase.movements) == len(set(phase.movements))


def test_every_movement_is_served_by_some_phase(plan):
    intersection = Intersection()
    served = {m.movement_id for ph in plan.values() for m in ph.movements}
    assert served == {m.movement_id for m in intersection.all_movements()}
    assert len(served) == 16


def test_phases_are_distinct(plan):
    sets = [frozenset(m.movement_id for m in ph.movements) for ph in plan.values()]
    assert len(set(sets)) == len(sets)


@pytest.mark.parametrize("approach", ["North", "South", "East", "West"])
def test_emergency_phase_is_exactly_one_approach(approach):
    intersection = Intersection()
    phase = build_emergency_phase(intersection, approach)
    assert phase.phase_type == PhaseType.EMERGENCY_OVERRIDE
    assert {m.movement_id for m in phase.movements} == {
        m.movement_id for m in intersection.get_approach(approach)
    }


def test_activating_a_phase_greens_only_its_movements():
    intersection = Intersection()
    phase = build_phase_plan(intersection)[PhaseType.PHASE_5]
    phase.activate()
    for movement in intersection.all_movements():
        expected = SignalState.GREEN if movement in phase.movements else SignalState.RED
        assert movement.signal.state == expected, movement.movement_id
