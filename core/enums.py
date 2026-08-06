"""
enums.py

Central definition of all enumerations used across the simulation.
Keeping them in one module avoids circular imports and centralizes
domain constants for future extensions (YOLO, SUMO, React, DB).
"""
from enum import Enum, auto


class MovementType(Enum):
    """The four possible movement directions at an incoming approach."""
    LEFT = auto()
    STRAIGHT = auto()
    RIGHT = auto()
    UTURN = auto()


class VehicleType(Enum):
    """Vehicle categories. Extendable with e.g. MOTORCYCLE, AMBULANCE."""
    CAR = auto()
    BIKE = auto()
    BUS = auto()
    TRUCK = auto()
    AMBULANCE = auto()


class Priority(Enum):
    """Vehicle priority levels. Ambulance/emergency vehicles are HIGH."""
    NORMAL = auto()
    HIGH = auto()


class SignalState(Enum):
    """States of a traffic signal."""
    RED = auto()
    YELLOW = auto()
    GREEN = auto()


class PhaseType(Enum):
    """
    Types of traffic phases.

    The ten normal phases map to the official 10-phase signal plan. Each
    Phase object wraps exactly the compatible movements listed for that
    phase (interpreted from the driver's perspective). The scheduler only
    ever activates these predefined phases.

    EMERGENCY_OVERRIDE is separate: it is built dynamically for ambulance
    preemption and does not modify the approved normal phase definitions.
    """
    PHASE_1 = auto()
    PHASE_2 = auto()
    PHASE_3 = auto()
    PHASE_4 = auto()
    PHASE_5 = auto()
    PHASE_6 = auto()
    PHASE_7 = auto()
    PHASE_8 = auto()
    PHASE_9 = auto()
    PHASE_10 = auto()
    EMERGENCY_OVERRIDE = auto()
