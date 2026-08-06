"""
logger.py

Optionally persists per-tick simulation snapshots to CSV files under a
logs/ directory. This is the foundation for later database logging and
feeding a React dashboard.

The logger is intentionally decoupled from the Statistics aggregator:
- Statistics keeps in-memory aggregates.
- Logger writes raw per-tick rows to disk.

Rows are appended lazily and flushed on close(), so the writer is safe
for long-running simulations.
"""
import csv
import os
from datetime import datetime

from config import simulation as sim_config


class CsvLogger:
    """
    Writes per-tick simulation rows to a CSV file.

    Args:
        log_dir (str): directory for log files (default from config).
        filename (str|None): optional explicit filename; else timestamped.
        fields (list[str]|None): CSV header; derived from first write if None.
    """

    # One CSV row per simulation tick. Lane-level detail is stored as a
    # compact JSON string so the row stays flat and pandas-friendly.
    FIELDS = [
        "simulation_time",
        "tick",
        "active_phase",
        "phase_remaining",
        "vehicles_spawned",
        "vehicles_served",
        "total_queue",
        "average_wait",
        "throughput",
        "congestion_ratio",
        "lane_queues_json",
        "lane_waits_json",
    ]

    def __init__(self, log_dir=None, filename=None, fields=None):
        self.log_dir = log_dir or sim_config.LOG_DIR
        self.fieldnames = fields or self.FIELDS
        self._file = None
        self._writer = None
        self._path = None
        self._closed = False

        os.makedirs(self.log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = filename or os.path.join(
            self.log_dir, f"{sim_config.LOG_FILENAME_PREFIX}_{ts}.csv"
        )

    # -------- Lifecycle --------

    def write(self, row: dict):
        """
        Append a single row (dict) to the CSV. Creates the file lazily on
        first write so an empty run produces no file.
        """
        if self._closed:
            raise RuntimeError("CsvLogger is closed.")
        if self._file is None:
            self._file = open(self._path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
            self._writer.writeheader()
        # Only keep known fields, in defined order.
        ordered = {key: row.get(key, "") for key in self.fieldnames}
        self._writer.writerow(ordered)

    def close(self):
        """Flush and close the underlying file. Safe to call multiple times."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self) -> str:
        return f"CsvLogger(path={self._path}, closed={self._closed})"
