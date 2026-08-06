"""
phases.py

Defines the signal phase plan for the intersection by grouping compatible
Movement objects into Phase objects. The TrafficScheduler only ever
activates these Phase objects; all compatible-movement knowledge lives
here (and in core.phase.Phase).

The official 10-phase signal plan is defined here. Each phase activates
exactly the compatible movements listed (interpreted from the driver's
perspective) and no others. The scheduler only ever activates these
predefined Phase objects.

EMERGENCY_OVERRIDE is intentionally NOT statically defined here - it is
built dynamically at runtime based on the ambulance approach.
"""
from core.enums import PhaseType
from core.phase import Phase


def build_phase_plan(intersection) -> dict:
    """
    Build the standard phase plan for the given intersection.

    Returns:
        dict: {PhaseType: Phase} mapping for the ten normal phases,
        each activating exactly the compatible movements listed below.
    """
    north = intersection.get_approach("North")
    south = intersection.get_approach("South")
    east = intersection.get_approach("East")
    west = intersection.get_approach("West")

    phases = {
        # ----------------------------------------------------------
        # PHASE 1
        #   West.Straight, West.Left, West.UTurn
        #   North.Left
        #   East.Left
        #   South.Right, South.Left
        # ----------------------------------------------------------
        PhaseType.PHASE_1: Phase(
            PhaseType.PHASE_1,
            [
                west.straight,
                west.left,
                west.uturn,
                north.left,
                east.left,
                south.right,
                south.left,
            ],
        ),
        # ----------------------------------------------------------
        # PHASE 2
        #   North.Straight, North.UTurn, North.Left
        #   East.Left
        #   South.Left
        #   West.Left, West.Right
        # ----------------------------------------------------------
        PhaseType.PHASE_2: Phase(
            PhaseType.PHASE_2,
            [
                north.straight,
                north.uturn,
                north.left,
                east.left,
                south.left,
                west.left,
                west.right,
            ],
        ),
        # ----------------------------------------------------------
        # PHASE 3
        #   East.Straight, East.UTurn, East.Left
        #   South.Left
        #   West.Left
        #   North.Left, North.Right
        # ----------------------------------------------------------
        PhaseType.PHASE_3: Phase(
            PhaseType.PHASE_3,
            [
                east.straight,
                east.uturn,
                east.left,
                south.left,
                west.left,
                north.left,
                north.right,
            ],
        ),
        # ----------------------------------------------------------
        # PHASE 4
        #   South.Straight, South.UTurn, South.Left
        #   West.Left
        #   North.Left
        #   East.Right, East.Left
        # ----------------------------------------------------------
        PhaseType.PHASE_4: Phase(
            PhaseType.PHASE_4,
            [
                south.straight,
                south.uturn,
                south.left,
                west.left,
                north.left,
                east.right,
                east.left,
            ],
        ),
        # ----------------------------------------------------------
        # PHASE 5
        #   West.Straight, West.Left, West.UTurn
        #   North.Left
        #   East.UTurn, East.Straight, East.Right
        #   South.Left
        # ----------------------------------------------------------
        PhaseType.PHASE_5: Phase(
            PhaseType.PHASE_5,
            [
                west.straight,
                west.left,
                west.uturn,
                north.left,
                east.uturn,
                east.straight,
                east.right,
                south.left,
            ],
        ),
        # ----------------------------------------------------------
        # PHASE 6
        #   South.Straight, South.Left, South.UTurn
        #   West.Left
        #   North.Straight, North.UTurn, North.Left
        #   East.Left
        # ----------------------------------------------------------
        PhaseType.PHASE_6: Phase(
            PhaseType.PHASE_6,
            [
                south.straight,
                south.left,
                south.uturn,
                west.left,
                north.straight,
                north.uturn,
                north.left,
                east.left,
            ],
        ),
        # ----------------------------------------------------------
        # PHASE 7
        #   South.Straight, South.Left, South.Right, South.UTurn
        #   West.Left
        #   North.Left
        #   East.UTurn, East.Left
        # ----------------------------------------------------------
        PhaseType.PHASE_7: Phase(
            PhaseType.PHASE_7,
            [
                south.straight,
                south.left,
                south.right,
                south.uturn,
                west.left,
                north.left,
                east.uturn,
                east.left,
            ],
        ),
        # ----------------------------------------------------------
        # PHASE 8
        #   West.Straight, West.Left, West.Right, West.UTurn
        #   North.Left
        #   East.Left
        #   South.Left, South.UTurn
        # ----------------------------------------------------------
        PhaseType.PHASE_8: Phase(
            PhaseType.PHASE_8,
            [
                west.straight,
                west.left,
                west.right,
                west.uturn,
                north.left,
                east.left,
                south.left,
                south.uturn,
            ],
        ),
        # ----------------------------------------------------------
        # PHASE 9
        #   North.Straight, North.Left, North.Right, North.UTurn
        #   East.Left
        #   South.Left
        #   West.Left, West.UTurn
        # ----------------------------------------------------------
        PhaseType.PHASE_9: Phase(
            PhaseType.PHASE_9,
            [
                north.straight,
                north.left,
                north.right,
                north.uturn,
                east.left,
                south.left,
                west.left,
                west.uturn,
            ],
        ),
        # ----------------------------------------------------------
        # PHASE 10
        #   East.Straight, East.UTurn, East.Right, East.Left
        #   South.Left
        #   West.Left
        #   North.UTurn, North.Left
        # ----------------------------------------------------------
        PhaseType.PHASE_10: Phase(
            PhaseType.PHASE_10,
            [
                east.straight,
                east.uturn,
                east.right,
                east.left,
                south.left,
                west.left,
                north.uturn,
                north.left,
            ],
        ),
    }
    return phases


def build_emergency_phase(intersection, approach_name: str) -> Phase:
    """
    Build an EMERGENCY_OVERRIDE phase dynamically for the approach the
    ambulance is arriving from.

    All movements of the ambulance approach receive green (including the
    UTurn movement); all other movements stay red. This is separate from
    the approved normal phases and is the foundation for future ambulance
    preemption logic.
    """
    approach = intersection.get_approach(approach_name)
    return Phase(
        PhaseType.EMERGENCY_OVERRIDE,
        [
            approach.straight,
            approach.right,
            approach.left,
            approach.uturn,
        ],
    )


def all_phase_types() -> list:
    """
    List of the standard (non-emergency) phase types in rotation order.

    This is the official 10-phase rotation:
        PHASE_1 .. PHASE_10
    """
    return [
        PhaseType.PHASE_1,
        PhaseType.PHASE_2,
        PhaseType.PHASE_3,
        PhaseType.PHASE_4,
        PhaseType.PHASE_5,
        PhaseType.PHASE_6,
        PhaseType.PHASE_7,
        PhaseType.PHASE_8,
        PhaseType.PHASE_9,
        PhaseType.PHASE_10,
    ]
