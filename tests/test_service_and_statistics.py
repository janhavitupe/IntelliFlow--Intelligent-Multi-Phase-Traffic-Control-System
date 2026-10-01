"""Vehicle discharge (service model) and KPI definitions (statistics)."""
import pytest

from analytics.statistics import Statistics
from core.enums import PhaseType, VehicleType
from config.phases import build_phase_plan
from services.service_model import ServiceModel
from helpers import TICK, add, car


@pytest.fixture
def green_west_straight(intersection):
    """PHASE_1 active (includes West_STRAIGHT); returns that movement."""
    phase = build_phase_plan(intersection)[PhaseType.PHASE_1]
    phase.activate()
    return intersection.get_approach("West").straight


def _tick(model, movement):
    model.accumulate([movement], TICK)
    return model.discharge([movement])


def test_car_needs_one_second_of_green(intersection, green_west_straight):
    add(intersection, "West", car())
    model = ServiceModel()
    assert _tick(model, green_west_straight) == []         # 0.5 s banked
    served = _tick(model, green_west_straight)             # 1.0 s banked
    assert [v.vehicle_type for _, v in served] == [VehicleType.CAR]


def test_truck_needs_longer_and_blocks_the_lane(intersection, green_west_straight):
    add(intersection, "West", car(vehicle_type=VehicleType.TRUCK))
    add(intersection, "West", car())
    model = ServiceModel()
    served = []
    for _ in range(4):                                     # 2.0 s < 2.2 s truck time
        served += _tick(model, green_west_straight)
    assert served == []
    served += _tick(model, green_west_straight)            # 2.5 s: truck leaves
    assert [v.vehicle_type for _, v in served] == [VehicleType.TRUCK]


def test_red_movement_never_discharges(intersection):
    movement = intersection.get_approach("North").straight  # red by default
    add(intersection, "North", car())
    model = ServiceModel()
    for _ in range(10):
        assert _tick(model, movement) == []


def test_statistics_kpi_definitions(intersection):
    stats = Statistics(intersection, interval=TICK)
    for _ in range(12):
        add(intersection, "North", car())
    intersection.update_waiting_times(2.0)   # every queued vehicle has waited 2 s
    stats.sample()                       # total queue 12 (congested: >= 10)
    intersection.get_approach("North").straight.lane.remove_front_vehicle()
    intersection.get_approach("North").straight.lane.remove_front_vehicle()
    intersection.get_approach("North").straight.lane.remove_front_vehicle()
    stats.sample()                       # total queue 9 (not congested)

    assert stats.average_queue_length == pytest.approx((12 + 9) / 2)
    assert stats.congestion_ratio == pytest.approx(0.5)
    # NOTE: "average waiting time" is the per-tick SUM of accumulated waits of
    # all queued vehicles, averaged over ticks (vehicle-seconds), not seconds
    # per vehicle. Pinned here so any redefinition is deliberate.
    assert stats.average_waiting_time == pytest.approx((12 * 2.0 + 9 * 2.0) / 2)


def test_throughput_is_vehicles_served_per_simulated_second(intersection):
    stats = Statistics(intersection, interval=TICK)
    movement = intersection.get_approach("North").straight
    stats.record_served([(movement, car()), (movement, car()), (movement, car())])
    intersection.advance_time(6.0)
    assert stats.throughput == pytest.approx(0.5)
