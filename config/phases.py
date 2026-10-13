"""
phases.py

The signal phase plan: which movements may be green together.

Vehicles drive on the LEFT. Each approach has four movements, from the centre
line outwards: U-turn, right, straight, left. Their geometry decides what can
run together:

  * Left turns hug the kerb and cross no other path, so every phase gives
    every approach its left turn ("free left").
  * A right turn crosses the opposing straight, so right turns only run in
    their own approach's phase.
  * Opposite straights (North with South, East with West) never cross;
    adjacent straights always do.
  * A U-turn crosses only the straight of the approach on its left
    (e.g. North U-turn x West straight).

The plan is six phases, each a maximal conflict-free group of movements
(derived from the geometry; tests/test_phase_plan.py re-checks every phase
for crossing paths):

  PHASE_1  North-South through  N & S straight + U-turn, all lefts
  PHASE_2  North approach       all North movements + S/W U-turn, all lefts
  PHASE_3  South approach       all South movements + N/E U-turn, all lefts
  PHASE_4  East-West through    E & W straight + U-turn, all lefts
  PHASE_5  East approach        all East movements + N/W U-turn, all lefts
  PHASE_6  West approach        all West movements + E/S U-turn, all lefts

Every movement is served: right turns in their approach phase, straights in
their approach phase and the through phase, lefts in all six.

EMERGENCY_OVERRIDE is built at runtime for the ambulance's approach.
"""
from core.enums import PhaseType
from core.phase import Phase

# Short descriptions, used by documentation and the simulation viewer.
PHASE_DESCRIPTIONS = {
    PhaseType.PHASE_1: "North-South through",
    PhaseType.PHASE_2: "North approach",
    PhaseType.PHASE_3: "South approach",
    PhaseType.PHASE_4: "East-West through",
    PhaseType.PHASE_5: "East approach",
    PhaseType.PHASE_6: "West approach",
}


def build_phase_plan(intersection) -> dict:
    """
    Build the phase plan for the given intersection.

    Returns:
        dict: {PhaseType: Phase} for the six normal phases.
    """
    n = intersection.get_approach("North")
    s = intersection.get_approach("South")
    e = intersection.get_approach("East")
    w = intersection.get_approach("West")
    all_lefts = [n.left, s.left, e.left, w.left]

    def approach_phase(main, uturn_a, uturn_b):
        others = [mv for mv in all_lefts if mv not in (main.left,)]
        return [main.left, main.straight, main.right, main.uturn, *others,
                uturn_a.uturn, uturn_b.uturn]

    return {
        PhaseType.PHASE_1: Phase(PhaseType.PHASE_1,
                                 [n.straight, n.uturn, s.straight, s.uturn, *all_lefts]),
        PhaseType.PHASE_2: Phase(PhaseType.PHASE_2, approach_phase(n, s, w)),
        PhaseType.PHASE_3: Phase(PhaseType.PHASE_3, approach_phase(s, n, e)),
        PhaseType.PHASE_4: Phase(PhaseType.PHASE_4,
                                 [e.straight, e.uturn, w.straight, w.uturn, *all_lefts]),
        PhaseType.PHASE_5: Phase(PhaseType.PHASE_5, approach_phase(e, n, w)),
        PhaseType.PHASE_6: Phase(PhaseType.PHASE_6, approach_phase(w, e, s)),
    }


def build_emergency_phase(intersection, approach_name: str) -> Phase:
    """
    Build an EMERGENCY_OVERRIDE phase for the approach the ambulance is
    arriving from: all four of its movements green, everything else red.
    """
    approach = intersection.get_approach(approach_name)
    return Phase(
        PhaseType.EMERGENCY_OVERRIDE,
        [approach.straight, approach.right, approach.left, approach.uturn],
    )


def all_phase_types() -> list:
    """The normal phase types in rotation order (PHASE_1 .. PHASE_6)."""
    return [pt for pt in PhaseType if pt != PhaseType.EMERGENCY_OVERRIDE]
