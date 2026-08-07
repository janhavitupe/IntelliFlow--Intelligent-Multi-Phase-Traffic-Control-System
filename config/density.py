"""
density.py

Central configuration for the PERCENTILE-BASED ADAPTIVE DENSITY scheduler
(Phase 3). All "magic numbers" for the adaptive decision-making layer live
here so the algorithm is fully configurable without editing source code.

The adaptive controller observes ONLY the number of vehicles waiting on each
approach (North / South / East / West). It never inspects a vehicle's
destination_movement (Left/Straight/Right/UTurn), preserving the controller
observation boundary.

Configuration is organized into:
    - GREEN_LIMITS      : minimum / maximum green + extension step (seconds)
    - DISCHARGE         : continuous-flow model parameters (Teemo-inspired)
    - PERCENTILE        : how to convert relative ranking into density classes
    - DENSITY_WEIGHTS   : numeric weight per density class (used in phase scoring)
    - FAIRNESS          : anti-starvation mechanism
    - APPROACH_ORDER    : deterministic tie-breaking order
"""

# ---------------------------------------------------------------------------
# Adaptive green-time limits (seconds)
#
# A green phase is a CONTINUOUS DISCHARGE interval. It starts at MIN_GREEN
# and is extended (in EXTENSION_STEP increments) while significant discharge
# continues, up to MAX_GREEN. It is NEVER computed as `vehicles x seconds`.
# ---------------------------------------------------------------------------
MIN_GREEN = 10.0          # smallest green duration a phase may receive
MAX_GREEN = 40.0          # hard cap on any single green duration
EXTENSION_STEP = 5.0      # how much each green extension adds

# ---------------------------------------------------------------------------
# Continuous discharge model (Teemo Attacking / interval-merging intuition)
#
# Represents how many vehicles can clear the approach per second while the
# signal stays green. Used only to ESTIMATE how long a continuous discharge
# interval should last given the current queue span - it is NOT a per-vehicle
# service multiplier.
# ---------------------------------------------------------------------------
APPROACH_DISCHARGE_RATE = 2.0   # vehicles/second clearing an approach
SIGNIFICANT_DISCHARGE_THRESHOLD = 15  # queue length below which no extension

# ---------------------------------------------------------------------------
# Percentile / density classification
#
# The four approaches are ranked each cycle (1 = most loaded). The ranking is
# converted into density classes using CONTIGUOUS_BUCKETS. Each strategy lists
# the density class assigned to each rank position (rank 1, 2, 3, 4).
#
#   "quartile" -> rank1 HIGH, rank2 MEDIUM, rank3 MEDIUM, rank4 LOW
#   "median"   -> rank1 HIGH, rank2 HIGH,  rank3 LOW,   rank4 LOW
# ---------------------------------------------------------------------------
PERCENTILE_STRATEGY = "quartile"   # "quartile" | "median"
PERCENTILE_BUCKETS = {
    "quartile": ["HIGH", "MEDIUM", "MEDIUM", "LOW"],
    "median": ["HIGH", "HIGH", "LOW", "LOW"],
}

# Density class -> numeric weight used when scoring the 10 phases.
#   HIGH = 3, MEDIUM = 2, LOW = 1
DENSITY_WEIGHTS = {
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1,
}

# ---------------------------------------------------------------------------
# Phase scoring (ALL 10 phases are scored every cycle)
#
#   Phase Score = sum over approaches of (
#                    Approach Weight x Phase Coverage
#                 )
#
#   Approach Weight : density weight of that approach this cycle.
#   Phase Coverage  : fraction of THAT approach's movements served by the
#                     phase (served_movements / total_movements_of_approach).
#
# Ties are broken deterministically by phase index (PHASE_1 first).
# ---------------------------------------------------------------------------
TIE_BREAK_PHASE_ORDER_ASCENDING = True  # lower phase index wins a tie

# ---------------------------------------------------------------------------
# Fairness (anti-starvation)
#
# A LOW-density approach must never starve. When an approach has had no green
# service for MAX_STARVATION_CYCLES consecutive scheduling cycles AND still has
# queued vehicles, it receives a FAIRNESS_BOOST that is added to its approach
# weight (so it can out-score the busy approaches and win a phase).
# ---------------------------------------------------------------------------
FAIRNESS_ENABLED = True
MAX_STARVATION_CYCLES = 4     # consecutive cycles without service -> starved
FAIRNESS_BOOST = 5.0          # added to a starved approach's weight
IGNORE_EMPTY_APPROACHES = True  # empty approaches should not be boosted

# ---------------------------------------------------------------------------
# Phase recency (anti-phase-starvation)
#
# The preserved 10-phase architecture contains some phases whose movement set
# is a strict subset of another phase (e.g. PHASE_1 movements are a subset of
# PHASE_8 with lower West coverage). A purely coverage-based score would never
# select such dominated phases, making them unreachable.
#
# To guarantee every normal phase remains reachable (and the full 10-phase
# controller is actually utilized), each phase tracks how many consecutive
# scheduling cycles it has NOT been selected. When a phase's recency reaches
# MAX_PHASE_STARVATION_CYCLES, it receives a bonus added to its score. The
# bonus is PHASE_RECENCY_BONUS x RECENCY, so it GROWS the longer a phase goes
# unselected. Because selected phases reset their recency to zero, a long-
# starved phase eventually out-scores every recently-used phase - even a
# coverage-dominated one that is a strict subset of another phase. This is
# the phase-level analogue of the approach-level fairness mechanism.
# ---------------------------------------------------------------------------
PHASE_RECENCY_ENABLED = True
MAX_PHASE_STARVATION_CYCLES = 6   # cycles without selection -> boosted
PHASE_RECENCY_BONUS = 3.0         # bonus per starvation cycle (scales with recency)

# ---------------------------------------------------------------------------
# Deterministic approach scan order
#
# Used for sorting/tie-breaking when two approaches hold equal queue counts.
# The first approach in this order wins the higher rank. Fixed for stable,
# reproducible behavior across runs.
# ---------------------------------------------------------------------------
APPROACH_ORDER = ("North", "South", "East", "West")


def density_for_rank(rank_index: int, strategy: str = None) -> str:
    """
    Return the density class for a given rank position (0-based).

    Args:
        rank_index (int): 0 = highest load, 3 = lowest load.
        strategy (str|None): percentile strategy key ("quartile"/"median").
            Defaults to PERCENTILE_STRATEGY.

    Returns:
        str: one of "HIGH", "MEDIUM", "LOW".
    """
    strategy = strategy or PERCENTILE_STRATEGY
    buckets = PERCENTILE_BUCKETS[strategy]
    return buckets[rank_index]
