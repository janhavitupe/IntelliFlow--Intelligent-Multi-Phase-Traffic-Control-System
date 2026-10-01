"""Normal (non-emergency) scheduling: timing, rotation and the green-end hook."""
from core.enums import PhaseType, SignalState
from scheduler.traffic_scheduler import TrafficScheduler
from strategies.base_strategy import BaseStrategy, HOLD
from helpers import TICK, phase_timeline


def test_fixed_timer_green_then_yellow_then_next_phase(fixed_scheduler):
    timeline = phase_timeline(fixed_scheduler, 60)
    assert timeline[0] == ("PHASE_1", "G", 24)    # 12 s green
    assert timeline[1] == ("PHASE_1", "Y", 4)     # 2 s yellow
    assert timeline[2][:2] == ("PHASE_2", "G")


def test_fixed_timer_rotates_through_all_ten_phases_and_wraps(fixed_scheduler):
    timeline = phase_timeline(fixed_scheduler, 28 * 11)
    greens = [name for name, state, _ in timeline if state == "G"]
    assert greens == [f"PHASE_{i}" for i in range(1, 11)] + ["PHASE_1"]


def test_only_active_phase_movements_are_green(fixed_scheduler, intersection):
    fixed_scheduler.update(TICK)
    active = set(fixed_scheduler.active_movements())
    for movement in intersection.all_movements():
        expected = SignalState.GREEN if movement in active else SignalState.RED
        assert movement.signal.state == expected


def test_yellow_movements_cannot_discharge(fixed_scheduler, intersection):
    from helpers import add, car
    phase_timeline(fixed_scheduler, 24)          # end of PHASE_1 green
    phase_timeline(fixed_scheduler, 1)           # now yellow
    assert fixed_scheduler.in_yellow
    movement = fixed_scheduler.active_movements()[0]
    add(intersection, movement.approach, car(), movement.movement_type)
    assert not movement.can_serve()


class _ScriptedStrategy(BaseStrategy):
    """Always PHASE_1 for 10 s; on_green_end replies from a script."""

    def __init__(self, replies):
        super().__init__("scripted")
        self.replies = list(replies)
        self.green_end_calls = 0

    def decide_next_phase(self, intersection, current_phase, time):
        return PhaseType.PHASE_1, 10.0

    def on_green_end(self, intersection, current_phase, time):
        self.green_end_calls += 1
        return self.replies.pop(0) if self.replies else None


def test_default_on_green_end_never_extends():
    assert BaseStrategy.on_green_end(None, None, None, 0.0) is None


def test_green_extension_lengthens_the_green(intersection):
    strategy = _ScriptedStrategy([5.0, 5.0])
    scheduler = TrafficScheduler(intersection, strategy, yellow_duration=2.0)
    timeline = phase_timeline(scheduler, 60)
    assert timeline[0] == ("PHASE_1", "G", 40)   # 10 + 5 + 5 s
    assert timeline[1] == ("PHASE_1", "Y", 4)
    assert strategy.green_end_calls == 3


def test_hold_keeps_green_and_reasks_without_advancing(intersection):
    strategy = _ScriptedStrategy([HOLD, HOLD, None])
    scheduler = TrafficScheduler(intersection, strategy, yellow_duration=2.0)
    phase_timeline(scheduler, 1 + 20)            # start + 10 s green -> first ask: HOLD
    assert strategy.green_end_calls == 1 and not scheduler.in_yellow
    scheduler.update(0.0)                        # re-ask with no time passing -> HOLD
    assert strategy.green_end_calls == 2 and not scheduler.in_yellow
    scheduler.update(0.0)                        # -> None: yellow starts
    assert scheduler.in_yellow
