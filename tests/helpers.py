"""Plain helper functions shared by the tests (fixtures live in conftest.py)."""
from core.enums import MovementType, Priority, VehicleType
from core.vehicle import Vehicle

TICK = 0.5


def car(waiting=0.0, vehicle_type=VehicleType.CAR):
    v = Vehicle(vehicle_type)
    v.waiting_time = waiting
    return v


def ambulance():
    return Vehicle(VehicleType.AMBULANCE, priority=Priority.HIGH)


def add(intersection, approach, vehicle, movement=MovementType.STRAIGHT):
    """Queue `vehicle` on approach/movement and return it."""
    intersection.spawn_vehicle(approach, movement, vehicle)
    return vehicle


def run(scheduler, seconds, delta=TICK):
    """Advance the scheduler by `seconds` in ticks of `delta`."""
    for _ in range(round(seconds / delta)):
        scheduler.update(delta)


def phase_timeline(scheduler, ticks, delta=TICK):
    """
    Update `ticks` times and return the run-length encoded state sequence
    [(phase_name, "G"|"Y", n_ticks), ...].
    """
    runs = []
    for _ in range(ticks):
        scheduler.update(delta)
        pt = scheduler.active_phase_type
        state = (pt.name if pt else None, "Y" if scheduler.in_yellow else "G")
        if runs and runs[-1][:2] == state:
            runs[-1] = (*state, runs[-1][2] + 1)
        else:
            runs.append((*state, 1))
    return runs
