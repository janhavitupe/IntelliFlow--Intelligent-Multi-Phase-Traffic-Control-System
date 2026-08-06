"""
analytics package

Collects and aggregates traffic statistics. Designed to be extensible so
it can later feed a database, React dashboard, or real-time analytics.
It also provides a CSV logger for per-tick snapshots.
"""
from .statistics import Statistics
from .logger import CsvLogger

__all__ = ["Statistics", "CsvLogger"]
