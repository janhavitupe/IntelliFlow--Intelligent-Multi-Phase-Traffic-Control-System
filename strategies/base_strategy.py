"""
base_strategy.py

Abstract base class for all phase-scheduling strategies.

The scheduler depends only on this interface. Each concrete strategy
implement the decision logic for which Phase to activate next. This
supports plugging in FixedTimer, Density, Queue Relaxation, and Emergency
strategies without touching the scheduler.
"""
from abc import ABC, abstractmethod


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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
