"""
strategies package

Implements the Strategy Design Pattern for phase scheduling.
The TrafficScheduler depends only on the abstract BaseStrategy interface,
so new algorithms can be added without modifying the scheduler.
"""
from .base_strategy import BaseStrategy
from .fixed_timer_strategy import FixedTimerStrategy
from .density_strategy import DensityStrategy

__all__ = ["BaseStrategy", "FixedTimerStrategy", "DensityStrategy"]
