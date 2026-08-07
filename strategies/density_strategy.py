"""
density_strategy.py

PERCENTILE-BASED ADAPTIVE DENSITY scheduling strategy (Phase 3).

The controller observes ONLY the number of vehicles waiting on each approach
(North / South / East / West). It never inspects a vehicle's intended
movement (destination_movement). All adaptive decisions therefore operate at
the APPROACH LEVEL.

Decision pipeline (runs once per scheduling decision):

    1. OBSERVE     - count vehicles waiting on each of the four approaches.
    2. RANK        - rank approaches by load (1 = most loaded), relative to
                     the current intersection state (percentile-based, NOT
                     fixed thresholds).
    3. CLASSIFY    - convert rank into density classes (HIGH / MEDIUM / LOW)
                     using a configurable percentile bucket mapping.
    4. FAIRNESS    - if an approach has starved for MAX_STARVATION_CYCLES
                     consecutive cycles and still holds queued vehicles, add
                     a fairness boost so it can win (LOW never starves).
    5. SCORE PHASES - score ALL 10 normal phases every cycle:

                        Phase Score = sum over approaches of (
                            Approach Weight x Phase Coverage
                        )

                     Approach Weight : density weight of that approach.
                     Phase Coverage  : fraction of THAT approach's movements
                                       served by the phase (served/total).
                     The highest-scoring phase wins; ties break by phase
                     index (PHASE_1 first).
    6. GREEN TIME   - compute a CONTINUOUS discharge green duration
                     (Teemo Attacking interval-merging intuition). The green
                     starts at MIN_GREEN and is extended in EXTENSION_STEP
                     increments while significant discharge continues, up to
                     MAX_GREEN. It is NEVER `count x service_time`.

Responsibilities (per the phase spec):
    - observe approach counts (read-only)
    - compute percentile ranking
    - classify density
    - choose next phase
    - compute adaptive green duration
    - enforce fairness

It must NOT:
    - move vehicles
    - discharge vehicles
    - manipulate queues
    - control signals directly
"""
from config.phases import all_phase_types
from config import density as density_config
from .base_strategy import BaseStrategy

# Density class ranks used for ordering when scoring.
_DENSITY_RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


class DensityStrategy(BaseStrategy):
    """
    Adaptive, percentile-based density scheduler.

    Attributes:
        name (str): strategy identifier.
        percentile_strategy (str): "quartile" | "median".
        min_green (float): floor for green duration.
        max_green (float): cap for green duration.
        extension_step (float): how much each green extension adds.
        discharge_rate (float): vehicles/second clearing an approach.
        significant_threshold (int): queue span below which no extension.
        fairness_enabled (bool): whether anti-starvation is active.
        max_starvation_cycles (int): consecutive no-service cycles before
            an approach is considered starved.
        fairness_boost (float): weight bonus for a starved approach.
        ignore_empty (bool): do not boost empty approaches.
        last_decision (dict|None): snapshot of the most recent decision for
            the analytics layer.
    """

    def __init__(self, config=None):
        super().__init__(name="density")
        cfg = config or density_config

        # Green-time limits + extension.
        self.min_green = getattr(cfg, "MIN_GREEN", 10.0)
        self.max_green = getattr(cfg, "MAX_GREEN", 40.0)
        self.extension_step = getattr(cfg, "EXTENSION_STEP", 5.0)

        # Continuous discharge model.
        self.discharge_rate = getattr(cfg, "APPROACH_DISCHARGE_RATE", 2.0)
        self.significant_threshold = getattr(
            cfg, "SIGNIFICANT_DISCHARGE_THRESHOLD", 15
        )

        # Percentile classification.
        self.percentile_strategy = getattr(cfg, "PERCENTILE_STRATEGY", "quartile")
        self.percentile_buckets = getattr(cfg, "PERCENTILE_BUCKETS", {})

        # Density weights used in phase scoring.
        self.density_weights = getattr(cfg, "DENSITY_WEIGHTS", {})

        # Fairness (approach-level anti-starvation).
        self.fairness_enabled = getattr(cfg, "FAIRNESS_ENABLED", True)
        self.max_starvation_cycles = getattr(cfg, "MAX_STARVATION_CYCLES", 4)
        self.fairness_boost = getattr(cfg, "FAIRNESS_BOOST", 5.0)
        self.ignore_empty = getattr(cfg, "IGNORE_EMPTY_APPROACHES", True)

        # Phase recency (phase-level anti-starvation). Guarantees even dominated
        # phases (strict subsets of another phase) remain reachable.
        self.phase_recency_enabled = getattr(cfg, "PHASE_RECENCY_ENABLED", True)
        self.max_phase_starvation_cycles = getattr(
            cfg, "MAX_PHASE_STARVATION_CYCLES", 6
        )
        self.phase_recency_bonus = getattr(cfg, "PHASE_RECENCY_BONUS", 3.0)

        # Deterministic ordering.
        self.approach_order = tuple(getattr(cfg, "APPROACH_ORDER", ()))

        # Fairness bookkeeping: approach -> consecutive cycles without service.
        self._starvation_cycles = {name: 0 for name in self.approach_order}

        # Phase recency bookkeeping: PhaseType.name -> consecutive cycles
        # without selection (used to keep every normal phase reachable).
        self._phase_recency = {
            pt.name: 0 for pt in all_phase_types()
        }

        # Selection counts per approach (for analytics).
        self.priority_selections = {name: 0 for name in self.approach_order}
        self.fairness_activations = 0

        # Snapshot of the most recent decision (for analytics/CSV).
        self.last_decision = None
        self._decision_counter = 0
        self._phase_recency_active = False
        self._scores_by_phase = {}

    # ------------------------------------------------------------------
    # Public strategy interface
    # ------------------------------------------------------------------

    def decide_next_phase(self, intersection, current_phase, time):
        """
        Choose the next phase and its adaptive green duration.

        Returns:
            (PhaseType, float): the chosen phase type and green seconds.
        """
        # 1. Observe approach counts (read-only, approach level).
        counts = self._observe_approach_counts(intersection)

        # 2. Rank approaches relative to current state.
        rankings = self._rank_approaches(counts)

        # 3. Classify density from rank positions.
        densities = self._classify_densities(rankings)

        # 4. Enforce fairness (update starvation, apply boosts).
        weights, fairness_active = self._apply_fairness(counts, densities)

        # 5. Score ALL 10 phases and pick the best.
        phase = self._select_phase(intersection, counts, densities, weights)

        # 6. Compute adaptive green duration (continuous discharge).
        green = self._compute_green_duration(counts, densities)

        # Advance starvation bookkeeping: the selected phase serves one or
        # more approaches; those approaches reset their starvation counter.
        # Also advances the phase-recency counters so every phase stays
        # reachable (phase-level anti-starvation).
        self._update_starvation(phase, counts)

        # Build the analytics snapshot.
        self._decision_counter += 1
        self.last_decision = {
            "decision_id": self._decision_counter,
            "time": time,
            "approach_counts": dict(counts),
            "rankings": dict(rankings),           # rank -> approach
            "densities": dict(densities),          # approach -> class
            "weights": dict(weights),              # approach -> effective weight
            "fairness_active": fairness_active,
            "phase_recency_active": self._phase_recency_active,
            "selected_phase": phase.phase_type.name,
            "green_duration": round(green, 2),
            "scores": dict(self._scores_by_phase),
            "phase_recency": dict(self._phase_recency),
            "served_approaches": self.served_approaches_for(phase.phase_type),
        }
        return phase.phase_type, green

    # ------------------------------------------------------------------
    # 1. Observation (controller boundary)
    # ------------------------------------------------------------------

    def _observe_approach_counts(self, intersection) -> dict:
        """
        Count vehicles waiting on each approach.

        The controller sees ONLY the number of queued vehicles per approach -
        it never reads a vehicle's destination_movement. This is the only
        signal the adaptive layer is allowed to observe.
        """
        return {
            name: intersection.get_approach(name).total_queue_length()
            for name in self.approach_order
        }

    # ------------------------------------------------------------------
    # 2. Percentile ranking
    # ------------------------------------------------------------------

    def _rank_approaches(self, counts: dict) -> dict:
        """
        Rank the four approaches from 1 (most loaded) to 4 (least loaded).

        Ranking is RELATIVE to the current intersection state (no fixed
        thresholds). Deterministic tie-breaking: when two approaches hold the
        same count, the one earlier in APPROACH_ORDER wins the higher rank.

        Returns:
            dict: rank (1..4) -> approach name.
        """
        # Sort by (-count, position-in-APPROACH_ORDER) for stability.
        ordered = sorted(
            self.approach_order,
            key=lambda name: (
                -counts[name],
                self.approach_order.index(name),
            ),
        )
        return {rank: name for rank, name in enumerate(ordered, start=1)}

    # ------------------------------------------------------------------
    # 3. Density classification
    # ------------------------------------------------------------------

    def _classify_densities(self, rankings: dict) -> dict:
        """
        Convert rank positions into density classes.

        Uses the configured percentile bucket mapping:
            "quartile" -> rank1 HIGH, rank2 MEDIUM, rank3 MEDIUM, rank4 LOW
            "median"   -> rank1 HIGH, rank2 HIGH,  rank3 LOW,   rank4 LOW

        Returns:
            dict: approach -> "HIGH" | "MEDIUM" | "LOW".
        """
        buckets = self.percentile_buckets.get(self.percentile_strategy)
        if buckets is None:
            buckets = ["HIGH", "MEDIUM", "MEDIUM", "LOW"]
        densities = {}
        for rank, approach in rankings.items():
            densities[approach] = buckets[rank - 1]
        return densities

    # ------------------------------------------------------------------
    # 4. Fairness (anti-starvation)
    # ------------------------------------------------------------------

    def _apply_fairness(self, counts: dict, densities: dict) -> tuple:
        """
        Apply the fairness mechanism and return effective weights.

        A starved approach (no service for >= max_starvation_cycles) that
        still holds queued vehicles receives a weight boost, guaranteeing it
        can out-score the busy approaches and eventually receive green.

        Returns:
            (dict approach->weight, bool fairness_active_this_cycle)
        """
        weights = {
            name: self.density_weights.get(densities[name], 1)
            for name in self.approach_order
        }
        fairness_active = False

        if not self.fairness_enabled:
            return weights, fairness_active

        for name in self.approach_order:
            starved = self._starvation_cycles.get(name, 0) >= self.max_starvation_cycles
            empty = counts.get(name, 0) == 0
            if starved and not (empty and self.ignore_empty):
                weights[name] += self.fairness_boost
                fairness_active = True

        return weights, fairness_active

    # ------------------------------------------------------------------
    # 5. Score all 10 phases
    # ------------------------------------------------------------------

    def _select_phase(self, intersection, counts, densities, weights):
        """
        Score ALL 10 normal phases and select the best.

        For each phase:
            Phase Score = sum over approaches of (
                Approach Weight x Phase Coverage
            )
        where
            Approach Weight : effective density weight this cycle.
            Phase Coverage  : fraction of THAT approach's movements served by
                              the phase (served_movements / total_movements).

        Ties are broken deterministically by phase index (PHASE_1 first).

        Returns:
            Phase: the winning Phase object (so callers can inspect its
            movements for starvation bookkeeping / analytics).
        """
        best_phase_type = None
        best_phase_obj = None
        best_score = None
        scores = {}
        phase_plan = all_phase_types()

        # Track whether any phase received the phase-recency (anti-phase-
        # starvation) bonus this cycle (for analytics).
        self._phase_recency_active = False

        for phase_type in phase_plan:
            phase = intersection and self._phase_for(intersection, phase_type)
            if phase is None:
                continue
            score = 0.0
            for approach_name in self.approach_order:
                coverage = self._phase_coverage(phase, approach_name)
                score += weights.get(approach_name, 1) * coverage

            # Phase-level anti-starvation: a phase that has not been selected
            # for MAX_PHASE_STARVATION_CYCLES consecutive cycles receives a
            # bonus so it can win a turn. The bonus SCALES with recency: the
            # longer a phase goes unselected, the larger its bonus grows, so
            # it eventually out-scores every other phase. This keeps every
            # normal phase reachable even when it is coverage-dominated by a
            # superset phase (e.g. PHASE_1 vs PHASE_8).
            if self.phase_recency_enabled:
                recency = self._phase_recency.get(phase_type.name, 0)
                if recency >= self.max_phase_starvation_cycles:
                    score += self.phase_recency_bonus * recency
                    self._phase_recency_active = True

            scores[phase_type.name] = round(score, 4)

            # Deterministic tie-break: prefer lower phase index.
            if best_phase_type is None or score > best_score:
                best_phase_type = phase_type
                best_phase_obj = phase
                best_score = score

        self._scores_by_phase = scores
        if best_phase_obj is None:
            best_phase_obj = self._phase_for(intersection, phase_plan[0])
        return best_phase_obj

    def _phase_for(self, intersection, phase_type):
        """Return the Phase object for the given type."""
        from config.phases import build_phase_plan
        plan = getattr(self, "_phase_plan", None)
        if plan is None:
            plan = build_phase_plan(intersection)
            self._phase_plan = plan
        return plan[phase_type]

    def _phase_coverage(self, phase, approach_name: str) -> float:
        """
        Fraction of the given approach's movements served by the phase.

        Coverage = served movements of that approach / total movements of
        that approach (4 in the current architecture: Left/Straight/Right/
        UTurn). Using a per-approach denominator keeps the algorithm
        future-proof for asymmetric approaches.
        """
        total = 4  # movements per approach in the preserved architecture
        served = sum(
            1 for m in phase.movements if getattr(m, "approach", None) == approach_name
        )
        return served / total if total else 0.0

    # ------------------------------------------------------------------
    # 6. Continuous green-time allocation (Teemo-inspired)
    # ------------------------------------------------------------------

    def _compute_green_duration(self, counts: dict, densities: dict) -> float:
        """
        Estimate a CONTINUOUS discharge green duration.

        The green phase is treated as one continuous discharge interval:
        it starts at MIN_GREEN and is extended in EXTENSION_STEP increments
        while significant discharge continues, capped at MAX_GREEN.

        Green is NEVER computed as `count x seconds`.

        The discharge model is inspired by the interval-merging idea from
        LeetCode 495 (Teemo Attacking): overlapping discharge intervals merge
        into a single continuous green, extended rather than multiplied.

        Returns:
            float: green duration within [min_green, max_green].
        """
        span = max(counts.values()) if counts else 0
        # Continuous discharge: how long one extended interval should last
        # given the current approach span (vehicles / rate), then stepped.
        ideal = span / self.discharge_rate

        # Start from the minimum and extend in discrete steps while the
        # queue span is significant (>= threshold) and we are under the cap.
        green = self.min_green
        if span >= self.significant_threshold:
            extensions = int((ideal - self.min_green) // self.extension_step)
            for _ in range(max(0, extensions)):
                candidate = green + self.extension_step
                if candidate > self.max_green:
                    break
                green = candidate

        return round(min(green, self.max_green), 2)

    # ------------------------------------------------------------------
    # Starvation bookkeeping
    # ------------------------------------------------------------------

    def _update_starvation(self, phase, counts: dict):
        """
        Update per-approach starvation counters after a phase selection.

        Approaches served by the selected phase reset to 0; all others
        increment by 1 (only if they still hold queued vehicles, so empty
        approaches do not accumulate starvation).

        Also advances the phase-recency counters: the selected phase resets
        to 0; all other normal phases increment by 1. This guarantees every
        normal phase remains reachable even if coverage-dominated.
        """
        selected_name = phase.phase_type.name

        # Approach-level starvation.
        served_approaches = {
            getattr(m, "approach", None) for m in phase.movements
        }
        for name in self.approach_order:
            if name in served_approaches:
                self._starvation_cycles[name] = 0
            elif self.ignore_empty and counts.get(name, 0) == 0:
                self._starvation_cycles[name] = 0
            else:
                self._starvation_cycles[name] += 1

        # Phase-level recency (anti-phase-starvation).
        for pt_name in self._phase_recency:
            if pt_name == selected_name:
                self._phase_recency[pt_name] = 0
            else:
                self._phase_recency[pt_name] += 1

    # ------------------------------------------------------------------
    # Introspection (for analytics)
    # ------------------------------------------------------------------

    def record_selection(self, approach_names):
        """
        Record that the selected phase served the given approaches.

        Called by the simulation/analytics layer after a decision is applied.
        Increments per-approach priority-selection counters used for the
        "number of priority selections per approach" analytics metric.
        """
        for name in approach_names:
            if name in self.priority_selections:
                self.priority_selections[name] += 1
        # If any starved approach was served, count a fairness activation.
        if self.last_decision and self.last_decision.get("fairness_active"):
            self.fairness_activations += 1

    def served_approaches_for(self, phase_type):
        """Return the list of approach names a phase serves."""
        from config.phases import build_phase_plan
        plan = getattr(self, "_phase_plan", None)
        if plan is None:
            return []
        phase = plan.get(phase_type)
        if phase is None:
            return []
        return sorted({getattr(m, "approach", None) for m in phase.movements})

    def reset(self):
        """Reset all internal state (used between runs/tests)."""
        self._starvation_cycles = {name: 0 for name in self.approach_order}
        self._phase_recency = {pt.name: 0 for pt in all_phase_types()}
        self.priority_selections = {name: 0 for name in self.approach_order}
        self.fairness_activations = 0
        self.last_decision = None
        self._decision_counter = 0
        self._scores_by_phase = {}
        self._phase_recency_active = False
        self._phase_plan = None
        self._approach_totals = {}

    def __repr__(self) -> str:
        return (
            f"DensityStrategy({self.percentile_strategy}, "
            f"min={self.min_green}, max={self.max_green})"
        )

