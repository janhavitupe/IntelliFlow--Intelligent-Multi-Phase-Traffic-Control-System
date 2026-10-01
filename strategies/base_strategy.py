"""
base_strategy.py

Abstract base class for all phase-scheduling strategies.

The scheduler depends only on this interface. Each concrete strategy
implement the decision logic for which Phase to activate next. This
supports plugging in FixedTimer, Density, Queue Relaxation, and Emergency
strategies without touching the scheduler.
"""
from abc import ABC, abstractmethod

# Returned by on_green_end() to make the scheduler hold the current green
# without advancing (the RL environment uses this to pause for an action).
HOLD = object()


class BaseStrategy(ABC):
    """
    Strategy interface for selecting the next traffic phase.

    Subclasses implement:
        - decide_next_phase(...): choose the next PhaseType/duration.
    """

    def __init__(self, name: str = "base"):
        self.name = name

    @abstractmethod
    def decide_next_phase(self, intersection, current_phase, time):
        """
        Decide which phase should be active next.

        Returns:
            tuple[PhaseType|None, float|None]:
                (phase_type, suggested_green_duration).
                The Emergency/Density strategies may return None values
                to signal no change / use defaults.
        """
        raise NotImplementedError

    def on_green_end(self, intersection, current_phase, time):
        """
        Called when the active phase's green time runs out.

        Returns:
            None   -> end the phase normally (yellow, then decide_next_phase).
            float  -> extend the green by this many seconds.
            HOLD   -> keep the green and ask again next update.

        The default never extends, so strategies that size their green up
        front (FixedTimer, Density) behave exactly as before.
        """
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
