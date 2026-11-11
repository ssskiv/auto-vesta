import math
import os
import re

import pytest

from webots_ros2_suv import referee as R

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WBT = open(os.path.join(PKG, 'worlds', 'robocross.wbt')).read()


@pytest.fixture(scope='module')
def zones():
    gates = {}
    for name in ('WinderStart', 'WinderFinish', 'WinderFreeDeal', 'WinderGaz'):
        gates[name] = [(float(x), float(y)) for x, y in
                       re.findall(name + r' \{\s*translation ([-\d.e]+) ([-\d.e]+)', WBT)]
    v = re.search(r'DEF Vesta1 LadaVesta \{\s*translation ([-\d.e]+) ([-\d.e]+)', WBT)
    return R.Zones(gates, (float(v[1]), float(v[2])), os.path.join(PKG, 'maps', R.MAP_YAML))


def test_zones(zones):
    assert zones.of_point((2.4, -2.3)) == 'S'           # spawn
    assert zones.of_point((60.0, 0.0)) == 'M'           # top lane
    assert zones.of_point((60.0, -45.0)) == 'A'         # bottom lane
    assert zones.of_point((162.0, -41.0)) == 'I'        # between the FreeDeal and Gaz gates


def drive(zones, xs, y=0.0, yaw=0.0, v=3.0, barrels=(), fence=()):
    s = R.Score(zones, list(fence), log=lambda m: None)
    t = 0.0
    for x in xs:
        s.step(t, 0.1, (x, y, yaw), v, list(barrels))
        t += 0.1
    return s


def upright(bid, x, y):
    return (bid, x, y, True, ((x, y), (0.7, 0.45), 0.0))


def test_barrier_pass_touch_and_reverse(zones):
    xs = [20 + 0.3 * k for k in range(100)]             # forward along the top lane, Main zone
    s = drive(zones, xs, barrels=[upright(0, 40.0, 2.0)])    # 2.0 - 0.865 - 0.45 = 0.69 m < W: credited
    assert [e['points'] for e in s.events] == [R.P_BARRIER_PASS]
    s = drive(zones, xs, barrels=[upright(0, 40.0, 4.0)])    # 2.69 m > W: not credited
    assert s.events == []
    s = drive(zones, xs, barrels=[upright(0, 40.0, 1.0)])    # overlaps the car: one touch, no credit
    assert [e['points'] for e in s.events] == [R.S_TOUCH]
    s = drive(zones, xs[::-1], yaw=0.0, v=-1.0)              # long reversing far from everything
    assert [e['points'] for e in s.events] == [R.S_REVERSE]


def test_speed(zones):
    xs = [20 + 0.6 * k for k in range(150)]
    s = drive(zones, xs, v=6.0)                              # 21.6 km/h for 15 s
    pts = [e['points'] for e in s.events]
    assert pts[0] == R.S_SPEED_FIRST and pts[1:] == [R.S_SPEED_LONG] * len(pts[1:]) and len(pts) >= 9


def test_obb_gap():
    a = ((0.0, 0.0), (1.0, 1.0), 0.0)
    assert R.obb_gap(a, ((3.0, 0.0), (1.0, 1.0), 0.0)) == pytest.approx(1.0)
    assert R.obb_gap(a, ((1.5, 0.0), (1.0, 1.0), math.pi / 4)) < 0


def test_speed_at_the_limit_is_one_exceeding(zones):
    s = R.Score(zones, [], log=lambda m: None)
    for k in range(300):                                     # 30 s jittering around 18 km/h
        v = R.SPEED_LIMIT + R.SPEED_TOLERANCE + (0.02 if k % 3 else -0.05)
        s.step(k * 0.1, 0.1, (20 + 0.5 * k, 0.0, 0.0), v, [])
    assert [e['points'] for e in s.events] == [R.S_SPEED_FIRST]


def test_uturn_route(zones):
    """Main zone -> Intermediate zone, a U-turn with reversing inside it, back through the Main zone."""
    s = R.Score(zones, [], log=lambda m: None)
    s.step(0.0, 0.1, (2.4, -2.3, 0.0), 1.0, [])             # starts moving in the start-finish zone
    s.left_s, s.visited = True, ['S', 'M']                   # (skip the drive along the top lane)
    ax, ay = zones.i_axis
    q = zones.quad
    cx, cy = sum(p[0] for p in q) / 4, sum(p[1] for p in q) / 4
    fwd = math.atan2(ay, ax)
    t = 10.0
    for k in range(60):                                      # turn 180 deg in place, partly in reverse
        yaw = fwd + math.pi * k / 59
        s.step(t, 0.1, (cx - 1.4 * math.cos(yaw), cy - 1.4 * math.sin(yaw), yaw), -0.5 if k < 20 else 0.5, [])
        t += 0.1
    pts = [e['points'] for e in s.events]
    assert pts == [R.P_TO_INTERMEDIATE, R.P_UTURN, R.P_UTURN_CLEAN, R.P_UTURN_REVERSE], s.events
    s.step(t, 0.1, (100.0, -5.0, math.pi), 3.0, [])          # back along the top lane (Main zone)
    s.step(t + 50, 0.1, (2.4, -2.3, math.pi), 1.0, [])       # back in the start-finish zone at 60 s
    assert [e['points'] for e in s.events][4:] == [R.P_BACK_MAIN, R.P_MAIN_FAST] and s.finished


def test_speed_tolerance(zones):
    s = drive(zones, [20 + 0.5 * k for k in range(100)], v=19.9 / 3.6)   # over 18, under 18 + 2
    assert s.events == []
