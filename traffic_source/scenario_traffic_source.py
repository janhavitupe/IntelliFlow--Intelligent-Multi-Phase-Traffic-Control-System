"""
scenario_traffic_source.py

A traffic source for hand-built test scenarios: random arrivals from an
inline schedule (same mechanism as ProfileTrafficSource) PLUS scripted
events such as "25 cars in every lane at t=0" or "an ambulance joins the
back of the East queue at t=90 s".
"""
from core.enums import MovementType, Priority, VehicleType
from core.vehicle import Vehicle
from .profile_traffic_source import ProfileTrafficSource


class ScenarioTrafficSource(ProfileTrafficSource):
    """
    Args:
        schedule: list of (start_s, end_s, {"rates": {...}, "mix": {...}}),
            rates keyed "Approach.MOVEMENT" in vehicles/second (may be empty).
        events: list of (time_s, approach, MovementType, VehicleType, count).
        seed, tick_duration: as for ProfileTrafficSource.
    """

    def __init__(self, schedule=(), events=(), seed=None, tick_duration=0.5):
        super().__init__(profile_key=None, seed=seed, tick_duration=tick_duration)
        self.schedule = list(schedule)
        self.events = sorted(events, key=lambda e: e[0])

    def _spec(self, time):
        for start, end, spec in self.schedule:
            if start <= time < end:
                return spec
        return {"rates": {}}

    def generate_spawns(self, time: float):
        spawns = self._spawns_from_spec(self._spec(time), time)
        for at, approach, movement, vtype, count in self.events:
            if time <= at < time + self.tick_duration:
                for _ in range(count):
                    priority = Priority.HIGH if vtype == VehicleType.AMBULANCE else Priority.NORMAL
                    spawns.append((approach, movement,
                                   Vehicle(vtype, arrival_time=time,
                                           destination_movement=movement, priority=priority)))
        return spawns

    def __repr__(self) -> str:
        return f"ScenarioTrafficSource({len(self.schedule)} windows, {len(self.events)} events)"
