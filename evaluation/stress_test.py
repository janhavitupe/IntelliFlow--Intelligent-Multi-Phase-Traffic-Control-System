"""
stress_test.py

Edge-case stress test for every controller.

Each scenario is a hand-built traffic situation (empty junction, a lone car
at night, a flooded lane, a burst, ambulances everywhere, an ambulance stuck
behind a queue, a camera outage, ...). Every controller runs every scenario
while an InvariantMonitor checks the safety rules on EVERY tick:

    I1 signals    : only the active phase's movements are green; nothing is
                    green during yellow; everything else is red
    I2 yellow     : a green phase never switches to another without yellow
    I3 green time : every normal green lasts 10-40 s (unless an ambulance
                    preempts it)
    I4 vehicles   : spawned = served + still queued (no vehicle lost)
    I5 ambulances : every ambulance that arrived early enough has been served

Performance (delay per vehicle, longest wait, unserved vehicles, time to
clear a burst) is reported next to the baselines. "DQN" is the deployed
controller (learned policy + safety envelope: no empty green, max red 90 s,
camera-failure fallback); "DQN (raw policy)" is the learned policy alone. "Optimal in every case"
cannot be proven for any controller; this checks the guaranteed properties
and measures behaviour where controllers typically fail.

Usage:
    python -m evaluation.stress_test            # all scenarios -> results/stress_test.md
    python -m evaluation.stress_test lone_cars  # a single scenario
    python -m evaluation.stress_test --max-red=off,120,90   # compare DQN safety settings
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

from config import traffic_profiles as tp
from config.phases import all_phase_types, build_phase_plan
from core.enums import MovementType, PhaseType, SignalState, VehicleType
from simulation import Simulation
from traffic_source.scenario_traffic_source import ScenarioTrafficSource

TICK = 0.5
MIN_GREEN, MAX_GREEN = 10.0, 40.0
APPROACHES = ("North", "South", "East", "West")
NO_AMBULANCE_MIX = {"CAR": 71, "BIKE": 18, "BUS": 6, "TRUCK": 5}
LIGHT = tp.LIGHT_TRAFFIC["schedule"][0][2]["rates"]
NORMAL = tp.NORMAL_TRAFFIC["schedule"][0][2]["rates"]
RUSH = tp.RUSH_HOUR["schedule"][0][2]["rates"]


def _window(rates, mix=NO_AMBULANCE_MIX, start=0.0, end=1e9):
    return (start, end, {"rates": rates, "mix": mix})


def _scaled(rates, k):
    return {key: r * k for key, r in rates.items()}


@dataclass
class Scenario:
    name: str
    description: str
    seconds: float
    schedule: list = field(default_factory=list)
    events: list = field(default_factory=list)
    blackout: tuple = None          # (start, end): camera outage (affects perception users)
    clearance: bool = False         # report time until the junction is empty


ALL_LANES = [(a, m) for a in APPROACHES for m in MovementType]

SCENARIOS = [
    Scenario("empty", "No traffic at all for 5 minutes.", 300),
    Scenario("lone_cars", "Quiet night: a single car arrives on each of the 16 lanes, "
             "30 s apart, nothing else.", 600,
             events=[(10 + 30 * i, a, m, VehicleType.CAR, 1) for i, (a, m) in enumerate(ALL_LANES)]),
    Scenario("single_lane_flood", "North straight flooded (0.8 veh/s); a trickle on every "
             "other lane.", 600,
             schedule=[_window(tp._base_rates(0.8, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01,
                                              0.01, 0.01, 0.01, 0.01))]),
    Scenario("main_vs_side", "Busy main road (North-South) and a quiet side road "
             "(East-West).", 600,
             schedule=[_window(tp._base_rates(0.45, 0.45, 0.04, 0.04, 0.1, 0.1, 0.01, 0.01,
                                              0.05, 0.05, 0.01, 0.01))]),
    Scenario("burst_drain", "25 cars appear in every lane at once (400 vehicles), then "
             "nothing: how fast does the junction clear?", 900, clearance=True,
             events=[(1.0, a, m, VehicleType.CAR, 25) for a, m in ALL_LANES]),
    Scenario("surge", "Light traffic, a 3-minute rush-hour surge, then light again.", 600,
             schedule=[_window(LIGHT, end=150), _window(RUSH, start=150, end=330),
                       _window(LIGHT, start=330)]),
    Scenario("all_trucks", "Normal volume, but every vehicle is a slow truck.", 600,
             schedule=[_window(NORMAL, mix={"TRUCK": 1})]),
    Scenario("ambulances_everywhere", "Normal traffic; ambulances arrive on all four "
             "approaches at the same moment, twice.", 600,
             schedule=[_window(NORMAL)],
             events=[(t, a, MovementType.STRAIGHT, VehicleType.AMBULANCE, 1)
                     for t in (120, 300) for a in APPROACHES]),
    Scenario("ambulance_behind_queue", "30 cars queue on East, then an ambulance joins the "
             "back of that queue.", 600,
             schedule=[_window(LIGHT)],
             events=[(1.0, "East", MovementType.STRAIGHT, VehicleType.CAR, 30),
                     (20.0, "East", MovementType.STRAIGHT, VehicleType.AMBULANCE, 1)]),
    Scenario("ambulance_stream", "Normal traffic plus an ambulance on North every 20 s.", 600,
             schedule=[_window(NORMAL)],
             events=[(t, "North", MovementType.STRAIGHT, VehicleType.AMBULANCE, 1)
                     for t in range(30, 570, 20)]),
    Scenario("oversaturated", "Demand 30% above rush hour (beyond capacity), no "
             "ambulances.", 600,
             schedule=[_window(_scaled(RUSH, 1.3))]),
    Scenario("camera_blackout", "Normal traffic; the camera is down from 120 s to 300 s "
             "(controllers that read the camera see nothing).", 600,
             schedule=[_window(NORMAL)], blackout=(120.0, 300.0)),
]
SCENARIO_BY_NAME = {s.name: s for s in SCENARIOS}


# ---------------------------------------------------------------------------
# Invariant monitor
# ---------------------------------------------------------------------------

class InvariantMonitor:
    """Checks the safety invariants after every simulation tick."""

    def __init__(self, sim):
        self.sim = sim
        plan = build_phase_plan(sim.intersection)
        self.normal = {pt: {m.movement_id for m in plan[pt].movements} for pt in all_phase_types()}
        self.violations = []
        self._prev = None                 # (phase_type, in_yellow)
        self._green_start = None
        self.green_lengths = []

    def _fail(self, rule, msg):
        if len(self.violations) < 20:
            self.violations.append(f"{rule} t={self.sim.intersection.time:.1f}s: {msg}")

    def check(self):
        sch, inter = self.sim.scheduler, self.sim.intersection
        phase = sch.current_phase
        pt = phase.phase_type if phase else None
        greens = {m.movement_id for m in inter.all_movements() if m.signal.state == SignalState.GREEN}
        yellows = {m.movement_id for m in inter.all_movements() if m.signal.state == SignalState.YELLOW}
        members = {m.movement_id for m in phase.movements} if phase else set()

        # I1 signals
        if phase is None and (greens or yellows):
            self._fail("I1", f"signals lit with no active phase: {sorted(greens | yellows)}")
        elif phase is not None and sch.in_yellow:
            if greens:
                self._fail("I1", f"green during yellow: {sorted(greens)}")
            if not yellows <= members:
                self._fail("I1", f"yellow outside the phase: {sorted(yellows - members)}")
        elif phase is not None:
            if greens != members:
                self._fail("I1", f"{pt.name} green set mismatch: extra {sorted(greens - members)}"
                                 f" missing {sorted(members - greens)}")
            if yellows:
                self._fail("I1", f"yellow during green: {sorted(yellows)}")

        # I2 yellow before every change away from a green phase; I3 green length
        state = (pt, sch.in_yellow)
        if self._prev is not None:
            prev_pt, prev_yellow = self._prev
            if prev_pt is not None and not prev_yellow and pt != prev_pt and pt is not None:
                self._fail("I2", f"{prev_pt.name} -> {pt.name} without yellow")
            was_green = prev_pt in self.normal and not prev_yellow
            now_green = pt == prev_pt and not sch.in_yellow
            if was_green and not now_green and self._green_start is not None:
                length = inter.time - self._green_start
                preempted = sch._emergency_approach is not None
                self.green_lengths.append(length)
                if length > MAX_GREEN + TICK:
                    self._fail("I3", f"{prev_pt.name} green lasted {length:.1f}s")
                if length < MIN_GREEN - TICK and not preempted:
                    self._fail("I3", f"{prev_pt.name} green only {length:.1f}s")
        if pt in self.normal and not sch.in_yellow and (self._prev is None or self._prev != state):
            self._green_start = inter.time - TICK
        self._prev = state

        # I4 vehicle conservation
        a = self.sim.analytics
        if a.total_vehicles_spawned != a.total_vehicles_served + inter.total_queue_length():
            self._fail("I4", f"spawned {a.total_vehicles_spawned} != served "
                             f"{a.total_vehicles_served} + queued {inter.total_queue_length()}")

    def finish(self, seconds):
        # I5: ambulances that arrived >= 60 s before the end must have been served
        for lane in self.sim.intersection.all_lanes():
            for v in lane.queue:
                if v.vehicle_type == VehicleType.AMBULANCE and v.arrival_time < seconds - 60:
                    self._fail("I5", f"ambulance from t={v.arrival_time:.0f}s never served")


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def controller_factory(name, seed=None, scenario=None):
    from strategies.density_strategy import DensityStrategy
    from strategies.fixed_timer_strategy import FixedTimerStrategy
    if name == "Fixed Timer":
        return FixedTimerStrategy
    if name == "Density":
        return DensityStrategy
    from perception import BlackoutPerception
    from strategies.rl_strategy import RLStrategy
    import run_experiments as rx
    agent = rx.load_agent("DQN", seed)
    blackout = scenario.blackout if scenario is not None else None
    if name == RAW_DQN:                      # the learned policy alone
        safety = {"max_red": None, "serve_waiting": False}
    elif "max-red" in name:
        safety = {"max_red": float(name.split("max-red ")[1].rstrip(" s"))}
    else:                                    # "DQN": deployed, with the safety envelope
        safety = {}

    def make():
        perception = BlackoutPerception(start=blackout[0], end=blackout[1]) if blackout else None
        return RLStrategy(agent=agent, perception=perception, **safety)
    return make


def run_scenario(scenario, factory, seed=1):
    source = ScenarioTrafficSource(scenario.schedule, scenario.events, seed=seed, tick_duration=TICK)
    ticks = int(scenario.seconds / TICK)
    sim = Simulation(max_ticks=ticks, live=False, strategy=factory(), traffic_source=source)
    monitor = InvariantMonitor(sim)
    clear_at = None
    for _ in range(ticks):
        sim.step()
        monitor.check()
        if (scenario.clearance and clear_at is None and sim.intersection.time > 2
                and sim.intersection.total_queue_length() == 0):
            clear_at = sim.intersection.time
    monitor.finish(scenario.seconds)
    a = sim.analytics
    amb = a.ambulance_average_delay
    return {
        "avg_delay": a.average_delay,
        "max_delay": a.max_delay,
        "spawned": a.total_vehicles_spawned,
        "unserved": sim.intersection.total_queue_length(),
        "ambulance_delay": amb,
        "clearance_s": clear_at,
        "longest_green": max(monitor.green_lengths, default=0.0),
        "violations": monitor.violations,
    }


RAW_DQN = "DQN (raw policy)"
CONTROLLERS = ["Fixed Timer", "Density", RAW_DQN, "DQN"]
DQN_SEEDS = [42, 7, 123]


def dqn_name(max_red):
    return RAW_DQN if max_red is None else f"DQN max-red {max_red:.0f} s"


def _job(args):
    scenario_name, controller, seed = args
    scenario = SCENARIO_BY_NAME[scenario_name]
    try:
        result = run_scenario(scenario, controller_factory(controller, seed, scenario))
    except Exception as exc:                        # a crash is a finding, not a harness error
        result = {"violations": [f"CRASH: {type(exc).__name__}: {exc}"]}
    return scenario_name, controller, seed, result


def main(names=None, max_reds=None):
    global CONTROLLERS
    if max_reds is not None:
        CONTROLLERS = ["Fixed Timer", "Density"] + [dqn_name(m) for m in max_reds]
    names = names or [s.name for s in SCENARIOS]
    jobs = [(n, c, s) for n in names for c in CONTROLLERS
            for s in (DQN_SEEDS if c.startswith("DQN") else [None])]
    with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as pool:
        results = list(pool.map(_job, jobs))
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "stress_test.json"), "w") as f:
        json.dump([{"scenario": n, "controller": c, "seed": s, **r} for n, c, s, r in results],
                  f, indent=1)
    write_report(results, names)
    return results


def _fmt(v, d=1):
    return "–" if v is None else f"{v:.{d}f}"


def write_report(results, names):
    lines = ["# Stress test: edge cases and safety invariants", "",
             "Generated by `python -m evaluation.stress_test`. Every controller runs every "
             "scenario; the DQN runs once per training seed and the **worst** seed is shown. "
             "Invariants (I1–I5, see the module docstring) are checked on every tick.", "",
             "| Scenario | Controller | Avg delay (s) | Longest wait (s) | Unserved at end | "
             "Ambulance delay (s) | Clear time (s) | Invariants |",
             "|---|---|---|---|---|---|---|---|"]
    total_violations = 0
    for name in names:
        sc = SCENARIO_BY_NAME[name]
        for controller in CONTROLLERS:
            rows = [r for n, c, s, r in results if n == name and c == controller]
            viol = [v for r in rows for v in r.get("violations", [])]
            total_violations += len(viol)
            ok = [r for r in rows if "avg_delay" in r]
            if not ok:
                lines.append(f"| {name} | {controller} | – | – | – | – | – | ❌ {viol[0]} |")
                continue
            worst = max(ok, key=lambda r: (r["max_delay"], r["avg_delay"]))
            label = controller + (f" (worst of {len(rows)} seeds)" if len(rows) > 1 else "")
            lines.append(
                f"| {name} | {label} | {_fmt(worst['avg_delay'])} | {_fmt(worst['max_delay'])} | "
                f"{worst['unserved']}/{worst['spawned']} | {_fmt(worst['ambulance_delay'])} | "
                f"{_fmt(worst['clearance_s'], 0)} | {'✅' if not viol else '❌ ' + viol[0]} |")
        lines.append(f"| | _{sc.description}_ | | | | | | |")
    lines += ["", f"**Total invariant violations: {total_violations}**", ""]
    with open(os.path.join("results", "stress_test.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    text = "\n".join(lines)
    enc = sys.stdout.encoding or "utf-8"
    print(text.encode(enc, "replace").decode(enc))   # Windows consoles lack emoji


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--max-red=")]
    reds = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--max-red=")]
    max_reds = [None if v in ("", "off") else float(v) for v in reds[0].split(",")] if reds else None
    main(args or None, max_reds)
