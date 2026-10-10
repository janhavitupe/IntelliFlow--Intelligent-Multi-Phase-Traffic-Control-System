"""
perception.py

Configuration for the perception layer: what the controller "sees".

The controller never reads the simulator directly. It receives a per-lane
summary (vehicle count + how long each visible vehicle has waited), which
the simulator produces today and a camera pipeline (detection + tracking)
will produce later.

The NOISY_* values model typical camera errors. They are ASSUMPTIONS to be
calibrated against real footage once the visual-input phase exists.
"""

# Vehicles that have waited longer than this count as "long waiters" (s).
LONG_WAIT_THRESHOLD = 60.0

# ---------------------------------------------------------------------------
# Simulated camera errors (NoisyPerception)
# ---------------------------------------------------------------------------
# Probability that a queued vehicle is not detected (occlusion, e.g. behind a bus).
NOISY_MISS_RATE = 0.05
# Probability per lane per observation of a phantom vehicle (false detection).
NOISY_FALSE_POSITIVE_RATE = 0.02
# Camera field of view: only the first N vehicles of each lane's queue are visible.
NOISY_VIEW_LIMIT = 15
# Relative error of an estimated wait (std of a multiplicative Gaussian).
NOISY_WAIT_ERROR = 0.10
# Probability that a vehicle's track was lost and re-acquired, so its wait is
# underestimated (only a random fraction of the true wait is known).
NOISY_TRACK_LOSS_RATE = 0.05

# ---------------------------------------------------------------------------
# Turn intent read from vehicle heading (HeadingIntentPerception)
# ---------------------------------------------------------------------------
# A camera cannot see where a driver wants to go, only which way the car faces.
# Drivers who will turn angle their vehicle toward the exit as they near the
# stop line, so only the front of each queue shows its intent.
# How many vehicles at the front of a lane are angled enough to read intent:
INTENT_VISIBLE_DEPTH = 3
# Probability that an angled (turning) vehicle is still read as going straight:
INTENT_MISREAD_RATE = 0.05
