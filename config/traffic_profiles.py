"""
traffic_profiles.py

Defines realistic, time-dependent traffic scenarios for the simulation.

Each profile is a schedule of time windows. Each window specifies an
independent arrival rate (vehicles/second) for every one of the 16
incoming movements (Left/Straight/Right/UTurn across the four
approaches), plus a vehicle-mix distribution.

UTurn arrivals are intentionally low but non-zero (a small fraction of
the through movement volume), reflecting real-world usage.
"""

from core.enums import MovementType

_MOVEMENTS = tuple(
    f"{approach}.{movement.name}"
    for approach in ("North", "South", "East", "West")
    for movement in MovementType
)


def _base_rates(north_s, south_s, east_s, west_s,
                north_l=0, south_l=0, east_l=0, west_l=0,
                north_r=0, south_r=0, east_r=0, west_r=0,
                north_u=0, south_u=0, east_u=0, west_u=0):
    """Build a 16-movement rate dict from the given per-movement rates.

    Rates are keyed as "Approach.MOVEMENT" (e.g. "North.STRAIGHT").
    UTurn rates default to a small non-zero fraction (5%) of the straight
    rate unless explicitly overridden.
    """
    return {
        "North.STRAIGHT": north_s,
        "North.LEFT": north_l,
        "North.RIGHT": north_r,
        "North.UTURN": north_u if north_u else north_s * 0.05,
        "South.STRAIGHT": south_s,
        "South.LEFT": south_l,
        "South.RIGHT": south_r,
        "South.UTURN": south_u if south_u else south_s * 0.05,
        "East.STRAIGHT": east_s,
        "East.LEFT": east_l,
        "East.RIGHT": east_r,
        "East.UTURN": east_u if east_u else east_s * 0.05,
        "West.STRAIGHT": west_s,
        "West.LEFT": west_l,
        "West.RIGHT": west_r,
        "West.UTURN": west_u if west_u else west_s * 0.05,
    }


MIX_DEFAULT = {"CAR": 70, "BIKE": 18, "BUS": 6, "TRUCK": 5, "AMBULANCE": 1}
MIX_NIGHT = {"CAR": 60, "BIKE": 5, "BUS": 5, "TRUCK": 28, "AMBULANCE": 2}
MIX_RUSH = {"CAR": 64, "BIKE": 22, "BUS": 8, "TRUCK": 5, "AMBULANCE": 1}


LIGHT_TRAFFIC = {
    "key": "LIGHT_TRAFFIC",
    "description": "Low volume, balanced across all movements.",
    "schedule": [
        (0, 1e9, {
            "rates": _base_rates(0.02, 0.02, 0.02, 0.02,
                                 0.01, 0.01, 0.01, 0.01,
                                 0.006, 0.006, 0.006, 0.006),
            "mix": MIX_DEFAULT,
        }),
    ],
}

NORMAL_TRAFFIC = {
    "key": "NORMAL_TRAFFIC",
    "description": "Average daytime flow with mild asymmetry.",
    "schedule": [
        (0, 1e9, {
            "rates": _base_rates(0.25, 0.25, 0.20, 0.20,
                                 0.12, 0.12, 0.10, 0.10,
                                 0.08, 0.08, 0.06, 0.06),
            "mix": MIX_DEFAULT,
        }),
    ],
}

RUSH_HOUR = {
    "key": "RUSH_HOUR",
    "description": "Heavy morning/evening commuting. Asymmetric.",
    "schedule": [
        (0, 1e9, {
            "rates": _base_rates(0.70, 0.45, 0.35, 0.30,
                                 0.30, 0.20, 0.15, 0.25,
                                 0.10, 0.12, 0.08, 0.15),
            "mix": MIX_RUSH,
        }),
    ],
}

NIGHT = {
    "key": "NIGHT",
    "description": "Very low traffic, truck-heavy freight hours.",
    "schedule": [
        (0, 1e9, {
            "rates": _base_rates(0.04, 0.04, 0.03, 0.03,
                                 0.01, 0.01, 0.01, 0.01,
                                 0.02, 0.02, 0.01, 0.01),
            "mix": MIX_NIGHT,
        }),
    ],
}

CUSTOM = {
    "key": "CUSTOM",
    "description": "Time-dependent: Morning -> Rush -> Normal -> Evening.",
    "schedule": [
        (0, 30, {
            "rates": _base_rates(0.30, 0.20, 0.15, 0.15,
                                 0.12, 0.08, 0.06, 0.10,
                                 0.06, 0.06, 0.04, 0.08),
            "mix": MIX_DEFAULT,
        }),
        (30, 60, {
            "rates": _base_rates(0.70, 0.45, 0.35, 0.30,
                                 0.30, 0.20, 0.15, 0.25,
                                 0.10, 0.12, 0.08, 0.15),
            "mix": MIX_RUSH,
        }),
        (60, 90, {
            "rates": _base_rates(0.25, 0.25, 0.20, 0.20,
                                 0.12, 0.12, 0.10, 0.10,
                                 0.08, 0.08, 0.06, 0.06),
            "mix": MIX_DEFAULT,
        }),
        (90, 1e9, {
            "rates": _base_rates(0.18, 0.15, 0.30, 0.35,
                                 0.08, 0.06, 0.12, 0.15,
                                 0.06, 0.05, 0.15, 0.20),
            "mix": MIX_DEFAULT,
        }),
    ],
}


PROFILES = {
    "LIGHT_TRAFFIC": LIGHT_TRAFFIC,
    "NORMAL_TRAFFIC": NORMAL_TRAFFIC,
    "RUSH_HOUR": RUSH_HOUR,
    "NIGHT": NIGHT,
    "CUSTOM": CUSTOM,
}


def get_profile(key: str) -> dict:
    """Return the profile dict for the given key, or raise KeyError."""
    if key not in PROFILES:
        raise KeyError(f"Unknown traffic profile: {key}. Available: {sorted(PROFILES)}")
    return PROFILES[key]


def profile_for_time(key: str, time: float) -> dict:
    """Resolve the active window spec for a profile at a simulation time."""
    profile = get_profile(key)
    for start, end, spec in profile["schedule"]:
        if start <= time < end:
            return spec
    return profile["schedule"][-1][2]


def all_movement_keys() -> tuple:
    """Return the canonical 16 movement keys in a stable order."""
    return _MOVEMENTS
