"""Shared fixtures for the test suite."""
import pytest

from core.intersection import Intersection
from scheduler.traffic_scheduler import TrafficScheduler
from strategies.fixed_timer_strategy import FixedTimerStrategy


@pytest.fixture
def intersection():
    return Intersection()


@pytest.fixture
def fixed_scheduler(intersection):
    return TrafficScheduler(
        intersection, FixedTimerStrategy(green_duration=12.0, yellow_duration=2.0),
        yellow_duration=2.0,
    )
