"""
simulation.py

Central configuration for the traffic simulation. All "magic numbers" that
controlled the simulation live here, so the simulator becomes configurable
without editing source code.

The values in this module are plain data. They are consumed by the
Simulation orchestrator, the ServiceModel, and the traffic sources.
"""

# ---------------------------------------------------------------------------
# Timing parameters (seconds)
# ---------------------------------------------------------------------------
TICK_DURATION = 0.5          # seconds per simulation tick
GREEN_TIME = 12.0            # default green duration per phase (seconds)
YELLOW_TIME = 2.0            # yellow transition duration (seconds)
SIMULATION_DURATION = 100    # default number of ticks (None = run forever)

# ---------------------------------------------------------------------------
# Vehicle service times (seconds to clear the intersection head-vehicle)
#
# Each lane accumulates elapsed green time; a vehicle departs only once
# enough green time has built up to satisfy its type's service time.
# ---------------------------------------------------------------------------
SERVICE_TIMES = {
    "BIKE": 0.6,
    "CAR": 1.0,
    "BUS": 1.8,
    "TRUCK": 2.2,
    "AMBULANCE": 0.8,
}

# ---------------------------------------------------------------------------
# Vehicle mix (relative weights). Used to derive a probability distribution.
# Keys are VehicleType enum member names.
# ---------------------------------------------------------------------------
VEHICLE_MIX = {
    "CAR": 70,
    "BIKE": 18,
    "BUS": 6,
    "TRUCK": 5,
    "AMBULANCE": 1,
}

# ---------------------------------------------------------------------------
# Default traffic profile key. Must match a profile defined in
# config/traffic_profiles.py.
# ---------------------------------------------------------------------------
TRAFFIC_PROFILE = "NORMAL_TRAFFIC"

# ---------------------------------------------------------------------------
# Random seed (None = nondeterministic). Setting a seed makes every run
# reproducible, which is essential for fair algorithm comparisons.
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Emergency vehicle probability (fraction of spawned vehicles that are
# HIGH-priority / ambulances). Used when the profile does not specify its
# own ambulance rate.
# ---------------------------------------------------------------------------
EMERGENCY_PROBABILITY = 0.01

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = "logs"
LOG_FILENAME_PREFIX = "simulation"
