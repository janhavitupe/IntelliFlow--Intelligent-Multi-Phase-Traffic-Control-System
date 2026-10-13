"""Normal (non-emergency) scheduling: timing, clearance, gap-out and the green-end hook."""
from core.enums import MovementType, PhaseType, SignalState
from scheduler.traffic_scheduler import TrafficScheduler
from strategies.base_strategy import BaseStrategy, HOLD
from helpers import TICK, add, car, phase_timeline

CLEARANCE_TICKS = 8          # 2 s yellow + 2 s all-red, in 0.5 s ticks


def test_fixed_timer_green_then_clearance_then_next_phase(fixed_scheduler):
    timeline = phase_timeline(fixed_scheduler, 60)
    assert timeline[0] == ("PHASE_1", "G", 24)                  # 12 s green
    assert timeline[1] == ("PHASE_1", "Y", CLEARANCE_TICKS)     # yellow + all-red
    assert timeline[2][:2] == ("PHASE_2", "G")


def test_all_red_follows_every_yellow(fixed_scheduler, intersection):
    phase_timeline(fixed_scheduler, 25)                         # start + 12 s green: yellow begins
    states = []
    for _ in range(CLEARANCE_TICKS - 1):
        fixed_scheduler.update(TICK)
        states.append({m.signal.state for m in intersection.all_movements()})
    assert SignalState.YELLOW in states[0]                      # first 2 s: yellow
    assert states[-1] == {SignalState.RED}                       # last 2 s: everything red
    assert states[-4] == {SignalState.RED}


def test_fixed_timer_rotates_through_all_phases_and_wraps(fixed_scheduler):
    timeline = phase_timeline(fixed_scheduler, (24 + CLEARANCE_TICKS) * 7)
    greens = [name for name, state, _ in timeline if state == "G"]
    assert greens == [f"PHASE_{i}" for i in range(1, 7)] + ["PHASE_1"]


def test_only_active_phase_movements_are_green(fixed_scheduler, intersection):
    fixed_scheduler.update(TICK)
    active = set(fixed_scheduler.active_movements())
    for movement in intersection.all_movements():
        expected = SignalState.GREEN if movement in active else SignalState.RED
        assert movement.signal.state == expected


def test_clearance_movements_cannot_discharge(fixed_scheduler, intersection):
    phase_timeline(fixed_scheduler, 25)                          # yellow has begun
    assert fixed_scheduler.in_yellow
    movement = fixed_scheduler.current_phase.movements[0]
    add(intersection, movement.approach, car(), movement.movement_type)
    assert not movement.can_serve()


class _ScriptedStrategy(BaseStrategy):
    """Always PHASE_1 (North-South through) with a given green; scripted on_green_end."""

    def __init__(self, replies=(), green=10.0, gap_out=False):
        super().__init__("scripted")
        self.replies = list(replies)
        self.green = green
        self.gap_out = gap_out
        self.green_end_calls = 0

    def decide_next_phase(self, intersection, current_phase, time):
        return PhaseType.PHASE_1, self.green

    def on_green_end(self, intersection, current_phase, time):
        self.green_end_calls += 1
        return self.replies.pop(0) if self.replies else None


def test_default_on_green_end_never_extends():
    assert BaseStrategy.on_green_end(None, None, None, 0.0) is None


def test_green_extension_lengthens_the_green(intersection):
    strategy = _ScriptedStrategy([5.0, 5.0])
    scheduler = TrafficScheduler(intersection, strategy, yellow_duration=2.0)
    timeline = phase_timeline(scheduler, 60)
    assert timeline[0] == ("PHASE_1", "G", 40)                  # 10 + 5 + 5 s
    assert timeline[1] == ("PHASE_1", "Y", CLEARANCE_TICKS)
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


def test_gap_out_ends_an_empty_green_when_others_wait(intersection):
    strategy = _ScriptedStrategy(green=30.0, gap_out=True)
    scheduler = TrafficScheduler(intersection, strategy, yellow_duration=2.0)
    add(intersection, "East", car())             # East straight waits; PHASE_1 lanes are empty
    timeline = phase_timeline(scheduler, 40)
    assert timeline[0] == ("PHASE_1", "G", 10)   # 5 s minimum, lanes empty >= 2 s -> gap-out


def test_no_gap_out_while_the_phase_still_has_traffic_or_nobody_waits(intersection):
    strategy = _ScriptedStrategy(green=30.0, gap_out=True)
    scheduler = TrafficScheduler(intersection, strategy, yellow_duration=2.0)
    timeline = phase_timeline(scheduler, 70)     # nobody waiting anywhere: keep the green
    assert timeline[0] == ("PHASE_1", "G", 60)

    other = TrafficScheduler(type(intersection)(), _ScriptedStrategy(green=30.0, gap_out=True),
                             yellow_duration=2.0)
    add(other.intersection, "North", car())       # served lane occupied (nothing discharges here)
    add(other.intersection, "East", car())
    assert phase_timeline(other, 70)[0] == ("PHASE_1", "G", 60)


def test_fixed_timer_never_gaps_out(fixed_scheduler, intersection):
    add(intersection, "East", car(), MovementType.STRAIGHT)
    assert phase_timeline(fixed_scheduler, 30)[0] == ("PHASE_1", "G", 24)
