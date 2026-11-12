import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from offroad_vehicle_interface.dbw_adapter_base import LoggingDbwAdapter  # noqa: E402
from offroad_vehicle_interface.logic import WheelSpeed, ZeroClampWatchdog, steer_from_twist  # noqa: E402

L = 2.648


def test_negative_speed_clamped_to_zero():
    w = ZeroClampWatchdog()
    for v in (-5.0, -0.001, -1e-9):
        assert w.update(v, 0.3, 1.0)[0] == 0.0
        assert w.output(1.0)[0] == 0.0
    assert w.update(2.0, 0.3, 1.0)[0] == 2.0
    assert w.clamped == 3


def test_steer_exact_atan():
    for v, om in [(2.0, 0.3), (5.0, -0.2), (1.0, 0.1)]:
        assert steer_from_twist(v, om, L, 0.0, 10.0) == math.atan(L * om / v)


def test_steer_clamp_hold_and_rate_limit():
    assert steer_from_twist(1.0, 5.0, L, 0.0, 0.6) == 0.6
    assert steer_from_twist(1.0, -5.0, L, 0.0, 0.6) == -0.6
    assert steer_from_twist(0.05, 5.0, L, 0.2, 0.6) == 0.2          # standstill: hold
    assert abs(steer_from_twist(1.0, 5.0, L, 0.0, 0.6, max_rate=0.5, dt=0.1) - 0.05) < 1e-12


def test_watchdog_300ms():
    w = ZeroClampWatchdog(0.3)
    assert w.output(0.0)[2]                      # nothing received yet: stale -> zero
    w.update(3.0, 0.1, 10.0)
    assert w.output(10.29) == (3.0, 0.1, False)
    assert w.output(10.31) == (0.0, 0.0, True)
    w.update(3.0, 0.1, 10.4)
    assert not w.output(10.5)[2]


def test_wheel_speed():
    ws = WheelSpeed(0.315)
    assert ws.update([0.0, 0.0], 0.0) is None
    v = ws.update([1.0, 1.0], 0.1)               # 10 rad/s * 0.315
    assert abs(v - 3.15) < 1e-9


def test_engage_estop_stub():
    a = LoggingDbwAdapter()
    a.send(0.1, 2.0, 0.0)
    assert a.status()['last'] == (0.1, 2.0, 0.0) and not a.status()['estop']
    a.emergency_stop()
    a.send(0.1, 2.0, 0.0)
    assert a.status()['estop'] and a.status()['last'] == (0.0, 0.0, 0.0)
