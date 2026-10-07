"""
visualize.py

Record the real controllers on every scenario and build an interactive,
self-contained simulation viewer (one HTML file, no server needed).

Fixed Timer, Density and the deployed DQN (learned policy + safety envelope)
run on IDENTICAL traffic: the 12 edge cases from evaluation/stress_test.py
and the 5 standard profiles. One frame is recorded per simulated second:
signal state of all 16 lanes, queue lengths, the head vehicle's wait,
ambulances, the active phase, safety-envelope overrides and running stats.

Usage:
    python visualize.py                  # all scenarios -> simulation_viewer.html (opens it)
    python visualize.py lone_cars surge  # only these scenarios
    python visualize.py --no-open        # don't open a browser

Requires trained models in models/ (python run_experiments.py).
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import webbrowser
from concurrent.futures import ProcessPoolExecutor

from config import rl as rl_config
from core.enums import PhaseType, SignalState, VehicleType
from evaluation.stress_test import SCENARIOS, SCENARIO_BY_NAME, TICK, controller_factory
from perception import LANE_ORDER
from simulation import Simulation
from traffic_source.scenario_traffic_source import ScenarioTrafficSource

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "viewer", "template.html")
OUTPUT = os.path.join(HERE, "simulation_viewer.html")
CONTROLLERS = ["Fixed Timer", "Density", "DQN"]
DQN_SEED = 42
STANDARD = [
    ("LIGHT_TRAFFIC", "Light traffic", "Low, balanced volume (standard test profile)."),
    ("NORMAL_TRAFFIC", "Normal traffic", "Average daytime flow with mild asymmetry."),
    ("RUSH_HOUR", "Rush hour", "Heavy, asymmetric commuting. Demand exceeds capacity and "
                               "~1% of vehicles are ambulances."),
    ("NIGHT", "Night", "Very low volume, truck-heavy."),
    ("CUSTOM", "Custom day", "Morning, rush, normal and evening windows."),
]
PHASES = [pt.name for pt in PhaseType if pt != PhaseType.EMERGENCY_OVERRIDE]
SIGNAL_CODE = {SignalState.GREEN: "g", SignalState.YELLOW: "y", SignalState.RED: "r"}


def _scenario_meta():
    meta = [{"id": f"std:{key}", "label": label, "group": "Standard profiles",
             "description": desc, "seconds": rl_config.EPISODE_LENGTH * TICK}
            for key, label, desc in STANDARD]
    meta += [{"id": f"edge:{s.name}", "label": s.name.replace("_", " ").capitalize(),
              "group": "Edge cases", "description": s.description, "seconds": s.seconds,
              "blackout": list(s.blackout) if s.blackout else None} for s in SCENARIOS]
    return meta


def _make_sim(scenario_id, controller):
    kind, name = scenario_id.split(":", 1)
    if kind == "std":
        factory = controller_factory(controller, DQN_SEED)
        ticks = rl_config.EPISODE_LENGTH
        sim = Simulation(profile_key=name, seed=1, max_ticks=ticks, live=False, strategy=factory())
    else:
        sc = SCENARIO_BY_NAME[name]
        factory = controller_factory(controller, DQN_SEED, sc)
        ticks = int(sc.seconds / TICK)
        source = ScenarioTrafficSource(sc.schedule, sc.events, seed=1, tick_duration=TICK)
        sim = Simulation(max_ticks=ticks, live=False, strategy=factory(), traffic_source=source)
    return sim, ticks


def record(job):
    """Run one (scenario, controller) and return compact per-second frames."""
    scenario_id, controller = job
    sim, ticks = _make_sim(scenario_id, controller)
    lanes = {f"{a}_{m.name}": lane for a, ap in sim.intersection.approaches.items()
             for m, lane in ap.lanes.items()}
    movements = {m.movement_id: m for m in sim.intersection.all_movements()}
    strat = sim.strategy
    f = {k: [] for k in ("sig", "q", "w", "amb", "phase", "avg", "maxw", "served", "queued",
                         "override", "camera")}
    last_overrides = 0
    for tick in range(ticks):
        sim.step()
        if tick % 2:                       # record once per simulated second
            continue
        sch = sim.scheduler
        f["sig"].append("".join(SIGNAL_CODE[movements[mid].signal.state] for mid in LANE_ORDER))
        f["q"].append([lanes[mid].queue_length for mid in LANE_ORDER])
        f["w"].append([int(lanes[mid].queue.peek().waiting_time) if not lanes[mid].is_empty else 0
                       for mid in LANE_ORDER])
        f["amb"].append(sum(1 << i for i, mid in enumerate(LANE_ORDER)
                            if any(v.vehicle_type == VehicleType.AMBULANCE for v in lanes[mid].queue)))
        pt = sch.active_phase_type
        f["phase"].append(-1 if pt is None else (10 if pt == PhaseType.EMERGENCY_OVERRIDE
                                                 else PHASES.index(pt.name)))
        a = sim.analytics
        f["avg"].append(round(a.average_delay, 1))
        f["maxw"].append(int(a.max_delay))
        f["served"].append(a.total_vehicles_served)
        f["queued"].append(sim.intersection.total_queue_length())
        overrides = (getattr(strat, "shield_overrides", 0) + getattr(strat, "empty_green_overrides", 0)
                     + getattr(strat, "fallback_decisions", 0))
        f["override"].append(1 if overrides > last_overrides else 0)
        last_overrides = overrides
        obs = getattr(strat, "last_observation", None)
        f["camera"].append(0 if obs is not None and not obs.available else 1)
    return scenario_id, controller, f


def build(scenario_ids=None, open_browser=True):
    meta = _scenario_meta()
    if scenario_ids:
        meta = [m for m in meta if m["id"].split(":", 1)[1] in scenario_ids]
    jobs = [(m["id"], c) for m in meta for c in CONTROLLERS]
    with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as pool:
        results = list(pool.map(record, jobs))
    runs = {}
    for scenario_id, controller, frames in results:
        runs.setdefault(scenario_id, {})[controller] = frames
    data = {"lanes": list(LANE_ORDER), "phases": PHASES, "controllers": CONTROLLERS,
            "scenarios": meta, "runs": runs,
            "safety": {"max_red": rl_config.SAFETY_MAX_RED}}
    with open(TEMPLATE, encoding="utf-8") as fh:
        page = fh.read().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    # The template has no document skeleton (it is also published as a web
    # page that adds its own); add one for opening the file locally.
    html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            '</head>\n<body>\n' + page + '\n</body>\n</html>\n')
    with open(OUTPUT, "w", encoding="utf-8") as fh:
        fh.write(html)
    with open(OUTPUT.replace(".html", ".fragment.html"), "w", encoding="utf-8") as fh:
        fh.write(page)
    size = os.path.getsize(OUTPUT) / 1e6
    print(f"Wrote {OUTPUT} ({size:.1f} MB, {len(meta)} scenarios x {len(CONTROLLERS)} controllers)")
    if open_browser:
        webbrowser.open("file://" + OUTPUT)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    build(args or None, open_browser="--no-open" not in sys.argv)
