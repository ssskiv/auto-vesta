"""Automatic referee of the final run, «РобоКросс.Трасса-2026» (version 09.04.2026), sections 15 and 18.

An extern Webots controller of the Robot "referee" (supervisor) in robocross.wbt; robot.launch.py
starts it. It reads the poses of the car, the barrels and the fence every step and scores the run:

Premiums (18.1)                                                   Penalties (18.1)
  500  start-finish -> Main zone -> Intermediate zone               -200 each touch of a barrier or the fence
  500  Intermediate -> Main zone -> start-finish                    -200 reversing against 15.17
 1500  Intermediate -> only the Additional zone -> start-finish     speed above 18 km/h + 2 km/h tolerance (seat
  300  a U-turn in the Intermediate zone (once, 15.16)                  occupied): -20 the 1st time, -100 each
  200  ... the U-turn stays in the zone, no fence touch                 next time, -10 a second after 5 s over
  200  ... reversing used in the U-turn
  200  passing a barrier closer than the car width, no touch,      Total (18.3): floor(P * K + Ш)
       whole side projection, not reversing; once per barrier and direction (a U-turn route passes
       each barrier twice)
 1000  route S - Main - Intermediate - Main - S under 121 s
 3000  route S - Main - Intermediate - Additional - S under 181 s

Not scored (no meaning in the simulation): engineering book, technician on the track, signalling
devices, gearbox position, parking zone, operator roles, remote control.

Zones come from the gates in the world: WinderStart / WinderFinish close the start-finish corner,
WinderFreeDeal / WinderGaz bound the Intermediate zone; the rest of the track splits into the Main
zone (the spawn side) and the Additional zone (place_barrels.py uses the same zones). A zone holds
the car when all four corners of its collision box are inside (15.13).
Touches: the car's collision box (2D) against the barrel cylinders and the boxes of DEF Barrier and
SolidBox nodes, i.e. the collision frames seen from above (2D, no heights).

Results: HUD in the Webots window, a log line per event and REFEREE_OUT (default
/tmp/referee_results.json), rewritten on every event.
"""
import json
import math
import os
import sys

import numpy as np

# ---- rules (section 18.1) and run settings
P_TO_INTERMEDIATE = 500
P_BACK_MAIN = 500
P_BACK_ADDITIONAL = 1500
P_UTURN = 300
P_UTURN_CLEAN = 200
P_UTURN_REVERSE = 200
P_BARRIER_PASS = 200
P_MAIN_FAST, T_MAIN = 1000, 121.0
P_EXT_FAST, T_EXT = 3000, 181.0
S_TOUCH = -200
S_REVERSE = -200
SPEED_LIMIT = 18 / 3.6               # m/s; 9 km/h for vehicles without a seat (18.1)
SPEED_TOLERANCE = 2 / 3.6            # measurement tolerance: charged only above SPEED_LIMIT + this (20 km/h)
S_SPEED_FIRST, S_SPEED_NEXT, S_SPEED_LONG = -20, -100, -10
# one exceeding lasts until the speed stays SPEED_HYST below the limit for SPEED_CALM s, so a car
# cruising at the limit is not charged for every noise crossing
SPEED_HYST, SPEED_CALM = 1 / 3.6, 1.0
K = float(os.environ.get('REFEREE_K', '1.0'))   # 18.2: class MT 1.0, AT 0.9, free class 0.8
RUN_TIME = 600.0                     # 15.7

# Lada Vesta collision box (LadaVesta.proto boundingObject), node frame
CAR_X = (-0.8775, 3.6775)
CAR_Y = (-0.865, 0.865)
CAR_L, CAR_W = CAR_X[1] - CAR_X[0], CAR_Y[1] - CAR_Y[0]
BARREL_R = 0.45                      # PlasticBarrel.proto: Cylinder radius 0.45 ...
BARREL_OFFSET = (-0.05, -0.05)       # ... at translation -0.05 -0.05 in the barrel frame
BARREL_H = 1.4
PAIR_DIST = 1.2                      # barrels closer than this form one barrier
TOUCH = 0.03                         # m: collision boxes closer than this touch
REVERSE_V = -0.15                    # m/s along the heading: reversing
REVERSE_MIN_T = 0.5                  # s of reversing before it counts
MAP_YAML = 'robocross.yaml'
WORLD_FROM_MAP = (2.2, -1.6)         # world = map + this (place_barrels.py)


# ---- geometry (2D, world frame)
def car_corners(x, y, c, s):
    return [(x + c * a - s * b, y + s * a + c * b) for a, b in
            ((CAR_X[1], CAR_Y[1]), (CAR_X[1], CAR_Y[0]), (CAR_X[0], CAR_Y[0]), (CAR_X[0], CAR_Y[1]))]


def to_car(px, py, x, y, c, s):
    dx, dy = px - x, py - y
    return c * dx + s * dy, -s * dx + c * dy


def rect_point_dist(u, v):
    """Distance from a point (car frame) to the car box."""
    du = max(CAR_X[0] - u, 0.0, u - CAR_X[1])
    dv = max(CAR_Y[0] - v, 0.0, v - CAR_Y[1])
    return math.hypot(du, dv)


def obb_gap(a, b):
    """Separation of two oriented boxes ((cx, cy), (hx, hy), yaw) along the SAT axes (< 0: overlap)."""
    gap = -math.inf
    for (cx, cy), _, yaw in (a, b):
        for ang in (yaw, yaw + math.pi / 2):
            ax, ay = math.cos(ang), math.sin(ang)
            proj = []
            for (bx, by), (hx, hy), byaw in (a, b):
                ux, uy = math.cos(byaw), math.sin(byaw)
                r = hx * abs(ux * ax + uy * ay) + hy * abs(-uy * ax + ux * ay)
                m = bx * ax + by * ay
                proj.append((m - r, m + r))
            gap = max(gap, max(proj[1][0] - proj[0][1], proj[0][0] - proj[1][1]))
    return gap


def side(p, a, b):
    """> 0 left of the line a -> b."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def seg_dist(p, a, b):
    vx, vy = b[0] - a[0], b[1] - a[1]
    t = max(0.0, min(1.0, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / max(vx * vx + vy * vy, 1e-12)))
    return math.hypot(p[0] - a[0] - t * vx, p[1] - a[1] - t * vy)


# ---- zones
class Zones:
    """S (start-finish), I (Intermediate), M (Main), A (Additional) for world points."""

    def __init__(self, gates, spawn, map_yaml):
        self.start, self.finish = gates['WinderStart'], gates['WinderFinish']
        self.s_start = math.copysign(1, side(spawn, *self.start))
        self.s_finish = math.copysign(1, side(spawn, *self.finish))
        pts = self.start + self.finish
        self.s_xmax = max(p[0] for p in pts) + 1.0
        self.s_ymin = min(p[1] for p in pts) - 1.0
        a, b = gates['WinderFreeDeal']
        c, d = gates['WinderGaz']
        self.quad = [a, b, d, c] if math.dist(b, d) < math.dist(b, c) else [a, b, c, d]
        cen = (sum(p[0] for p in self.quad) / 4, sum(p[1] for p in self.quad) / 4)
        self.q_sign = [math.copysign(1, side(cen, self.quad[k], self.quad[(k + 1) % 4])) for k in range(4)]
        self.borders = [tuple(self.start), tuple(self.finish), (a, b), (c, d)]
        # Intermediate zone axis: from the FreeDeal gate (Main side) to the Gaz gate
        self.i_axis = ((c[0] + d[0] - a[0] - b[0]) / 2, (c[1] + d[1] - a[1] - b[1]) / 2)
        n = math.hypot(*self.i_axis)
        self.i_axis = (self.i_axis[0] / n, self.i_axis[1] / n)
        self._split(spawn, map_yaml)

    def in_s(self, p):
        return (math.copysign(1, side(p, *self.start)) == self.s_start
                and math.copysign(1, side(p, *self.finish)) == self.s_finish
                and p[0] <= self.s_xmax and p[1] >= self.s_ymin)

    def in_i(self, p):
        return all(math.copysign(1, side(p, self.quad[k], self.quad[(k + 1) % 4])) == self.q_sign[k]
                   for k in range(4))

    def _split(self, spawn, map_yaml):
        """Track cells outside S and I fall into two components: the one the car enters from
        the start line is the Main zone. Every map cell gets the label of the nearest track cell."""
        from scipy import ndimage
        occ, ox, oy, res = read_map(map_yaml)
        ox, oy = ox + WORLD_FROM_MAP[0], oy + WORLD_FROM_MAP[1]
        h, w = occ.shape
        X, Y = np.meshgrid(ox + (np.arange(w) + 0.5) * res, oy + (np.arange(h) + 0.5) * res)
        free = ndimage.distance_transform_edt(~occ) * res >= 2.0   # closes gaps in the fence
        in_s = ((np.sign(side((X, Y), *self.start)) == self.s_start)
                & (np.sign(side((X, Y), *self.finish)) == self.s_finish)
                & (X <= self.s_xmax) & (Y >= self.s_ymin))
        in_i = np.ones_like(free)
        for k in range(4):
            in_i &= np.sign(side((X, Y), self.quad[k], self.quad[(k + 1) % 4])) == self.q_sign[k]
        labels, n = ndimage.label(free & ~in_s & ~in_i)
        # the Main zone: the component just outside the start line, on the far side from the spawn
        mid = ((self.start[0][0] + self.start[1][0]) / 2, (self.start[0][1] + self.start[1][1]) / 2)
        nx, ny = self.start[1][1] - self.start[0][1], -(self.start[1][0] - self.start[0][0])
        if side((mid[0] + nx, mid[1] + ny), *self.start) * self.s_start > 0:
            nx, ny = -nx, -ny
        k = math.hypot(nx, ny)
        probe = (mid[0] + 4.0 * nx / k, mid[1] + 4.0 * ny / k)
        _, (iy, ix) = ndimage.distance_transform_edt(labels == 0, return_indices=True)
        nearest = labels[iy, ix]
        self.main_label = nearest[self._cell(probe, ox, oy, res, h, w)]
        self.labels, self.grid = nearest, (ox, oy, res, h, w)
        assert n >= 2 and self.main_label > 0, 'could not split the track into the Main / Additional zone'

    @staticmethod
    def _cell(p, ox, oy, res, h, w):
        return (min(h - 1, max(0, int((p[1] - oy) / res))), min(w - 1, max(0, int((p[0] - ox) / res))))

    def of_point(self, p):
        if self.in_s(p):
            return 'S'
        if self.in_i(p):
            return 'I'
        return 'M' if self.labels[self._cell(p, *self.grid)] == self.main_label else 'A'

    def border_dist(self, p):
        return min(seg_dist(p, a, b) for a, b in self.borders)


def read_map(yaml_path):
    meta = {}
    for line in open(yaml_path):
        if ':' in line:
            k, v = line.split(':', 1)
            meta[k.strip()] = v.strip()
    image = os.path.join(os.path.dirname(yaml_path), meta['image'])
    origin = [float(v) for v in meta['origin'].strip('[]').split(',')]
    data = open(image, 'rb').read()
    tokens, pos = [], 0
    while len(tokens) < 4:                       # P5 header
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b'#':
            pos = data.index(b'\n', pos) + 1
            continue
        start = pos
        while not data[pos:pos + 1].isspace():
            pos += 1
        tokens.append(data[start:pos])
    w, h = int(tokens[1]), int(tokens[2])
    pix = np.frombuffer(data, np.uint8, w * h, pos + 1).reshape(h, w)[::-1]   # row 0 = min y
    return (255.0 - pix) / 255.0 > float(meta['occupied_thresh']), origin[0], origin[1], float(meta['resolution'])


# ---- scoring (pure: fed with poses, testable without Webots)
class Score:
    def __init__(self, zones, fence, log=print):
        self.zones, self.fence, self.log = zones, fence, log
        self.events, self.premium, self.penalty = [], 0, 0
        self.t = 0.0
        self.t0 = None               # «Марш» (15.10): there is no judge here, so the moment the car starts moving
        self.finished = None
        self.left_s = False          # all of the car left the start-finish zone
        self.visited = []            # zones (by the car centre) since leaving S, consecutive duplicates removed
        self.reached_i = False
        self.uturn = None            # dict while turning in I
        self.uturn_done = False
        self.after_i = []            # zones of the car centre after leaving I
        self.touching = {}           # object id -> touching now
        self.reverse_t = 0.0
        self.reverse_charged = False
        self.speed_over_t = 0.0
        self.speed_calm_t = 0.0
        self.speed_active = False
        self.speed_count = 0
        self.speed_long_s = 0.0
        self.barriers = {}           # barrier id -> passage state

    def add(self, points, text):
        if points >= 0:
            self.premium += points
        else:
            self.penalty += points
        self.events.append(dict(t=round(self.t, 2), points=points, text=text))
        self.log(f'[{self.t:7.2f} s] {points:+5d}  {text}   (итог {self.total()})')

    def total(self):
        return math.floor(self.premium * K + self.penalty)

    def step(self, t, dt, pose, v_long, barrels):
        """pose = (x, y, yaw) of the car node; barrels = [(barrier id, cx, cy, upright, footprint)]."""
        if self.finished:
            return
        if self.t0 is None:
            if abs(v_long) < 0.1:
                return
            self.t0 = t
            self.log(f'старт заезда: ТС тронулось, время симуляции {t:.2f} с')
        t = self.t = t - self.t0
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        corners = car_corners(x, y, c, s)
        zc = [self.zones.of_point(p) for p in corners]
        centre = self.zones.of_point((x + c * 1.4, y + s * 1.4))   # middle of the box
        all_in = zc[0] if len(set(zc)) == 1 else None
        if not self.left_s and all(z != 'S' for z in zc):
            self.left_s = True
            self.log(f'[{t:7.2f} s] ТС покинуло Зону Старта-финиша')
        if self.left_s and (not self.visited or self.visited[-1] != centre):
            self.visited.append(centre)
        touches = self._touches(x, y, c, s, barrels)
        self._zones(t, all_in, centre, yaw, v_long, touches)
        self._reverse(dt, x, y, c, s, v_long, barrels)
        self._speed(dt, abs(v_long))
        self._passes(x, y, c, s, v_long, barrels, touches)
        if t >= RUN_TIME:
            self.finish('истекли 10 минут (п. 15.7)')

    # -- zones, U-turn, tasks
    def _zones(self, t, all_in, centre, yaw, v_long, touches):
        z = self.zones
        if not self.reached_i and self.left_s and all_in == 'I' and 'M' in self.visited:
            self.reached_i = True
            self.add(P_TO_INTERMEDIATE, 'Старт-финиш -> Основная зона -> Промежуточная зона')
        heading_i = math.cos(yaw) * z.i_axis[0] + math.sin(yaw) * z.i_axis[1]
        if self.reached_i and not self.uturn_done:
            if self.uturn is None and centre == 'I':
                self.uturn = dict(clean=True, reverse=False, sign=math.copysign(1, heading_i))
            if self.uturn is not None:
                u = self.uturn
                if all_in != 'I' or any(k[0] == 'fence' for k in touches):
                    u['clean'] = False
                if v_long < REVERSE_V:
                    u['reverse'] = True
                if heading_i * u['sign'] < -0.5 and centre == 'I':    # facing the other way
                    self.uturn_done = True
                    self.add(P_UTURN, 'разворот в Промежуточной зоне')
                    if u['clean']:
                        self.add(P_UTURN_CLEAN, 'разворот не выходя из Промежуточной зоны, без касания ограждений')
                    if u['reverse']:
                        self.add(P_UTURN_REVERSE, 'задний ход при развороте')
                elif centre != 'I':
                    self.uturn = None                                  # drove through, no U-turn
        if self.reached_i and centre != 'I' and (not self.after_i or self.after_i[-1] != centre):
            self.after_i.append(centre)
        if self.left_s and all_in == 'S' and len(self.visited) > 1:   # 15.11, 15.13: back in the start-finish zone
            route = [q for q in self.after_i if q != 'S']
            if self.reached_i and route == ['M']:
                self.add(P_BACK_MAIN, 'Промежуточная зона -> Основная зона -> Старт-финиш')
                if t < T_MAIN:
                    self.add(P_MAIN_FAST, f'маршрут через Основную зону за {t:.1f} с (< {T_MAIN:.0f} с)')
            elif self.reached_i and route == ['A']:
                self.add(P_BACK_ADDITIONAL, 'Промежуточная зона -> только Дополнительная зона -> Старт-финиш')
                if t < T_EXT:
                    self.add(P_EXT_FAST, f'маршрут через Дополнительную зону за {t:.1f} с (< {T_EXT:.0f} с)')
            self.finish('ТС вернулось в Зону Старта-финиша (п. 15.11)')

    # -- touches: one penalty per touch, i.e. per contact that starts
    def _touches(self, x, y, c, s, barrels):
        now = set()
        car_box = ((x + c * (CAR_X[0] + CAR_X[1]) / 2, y + s * (CAR_X[0] + CAR_X[1]) / 2),
                   (CAR_L / 2, CAR_W / 2), math.atan2(s, c))
        for k, (bid, bx, by, upright, box) in enumerate(barrels):
            if upright:
                d = rect_point_dist(*to_car(bx, by, x, y, c, s)) - BARREL_R
            else:
                d = obb_gap(car_box, box)
            if d < TOUCH:
                now.add(('barrel', k, bid))
        for k, box in enumerate(self.fence):
            if math.dist(box[0], car_box[0]) < 8.0 and obb_gap(car_box, box) < TOUCH:
                now.add(('fence', k))
        for key in now - set(self.touching):
            self.add(S_TOUCH, 'касание преграды (бочка)' if key[0] == 'barrel' else 'касание ограждения')
        self.touching = dict.fromkeys(now, True)
        return now

    # -- 15.17: reversing only near a barrier / zone border, or in the U-turn
    def _reverse(self, dt, x, y, c, s, v_long, barrels):
        if v_long >= REVERSE_V:
            self.reverse_t, self.reverse_charged = 0.0, False
            return
        self.reverse_t += dt
        if self.reverse_t < REVERSE_MIN_T or self.reverse_charged:
            return
        if self.uturn is not None and not self.uturn_done:
            return
        near = min([rect_point_dist(*to_car(bx, by, x, y, c, s)) - BARREL_R for _, bx, by, _, _ in barrels]
                   + [max(0.0, math.dist(b[0], (x, y)) - math.hypot(*b[1]) - CAR_L) for b in self.fence]
                   + [self.zones.border_dist((x + c * 1.4, y + s * 1.4)) - CAR_L / 2], default=math.inf)
        if near >= CAR_L:
            self.reverse_charged = True
            self.add(S_REVERSE, 'задний ход дальше длины ТС от преград и границ зон (п. 15.17)')

    # -- speed (18.1, vehicle with a seat: 18 km/h)
    def _speed(self, dt, v):
        if self.speed_active:
            self.speed_calm_t = self.speed_calm_t + dt if v < SPEED_LIMIT + SPEED_TOLERANCE - SPEED_HYST else 0.0
            if self.speed_calm_t >= SPEED_CALM:
                self.speed_active = False
        if v <= SPEED_LIMIT + SPEED_TOLERANCE:
            self.speed_over_t = 0.0
            return
        if not self.speed_active:
            self.speed_active, self.speed_calm_t = True, 0.0
            self.speed_count += 1
            self.speed_long_s = 5.0
            self.add(S_SPEED_FIRST if self.speed_count == 1 else S_SPEED_NEXT,
                     f'скорость {v * 3.6:.1f} км/ч > {SPEED_LIMIT * 3.6:.0f} км/ч + {SPEED_TOLERANCE * 3.6:.0f} допуск '
                     f'({self.speed_count}-й раз)')
        self.speed_over_t += dt
        while self.speed_over_t >= self.speed_long_s + 1.0:
            self.speed_long_s += 1.0
            self.add(S_SPEED_LONG, f'скорость выше {SPEED_LIMIT * 3.6:.0f} км/ч дольше 5 с')

    # -- barrier passes: whole side projection, closer than the car width, forward, no touch
    def _passes(self, x, y, c, s, v_long, barrels, touches):
        groups = {}
        for k, (bid, bx, by, _, _) in enumerate(barrels):
            groups.setdefault(bid, []).append((k, to_car(bx, by, x, y, c, s)))
        touched = {key[2] for key in touches if key[0] == 'barrel'}
        heading = math.atan2(s, c)
        for bid, pts in groups.items():
            us = [u for _, (u, _) in pts]
            gap = min(abs(v) for _, (_, v) in pts) - CAR_Y[1] - BARREL_R
            st = self.barriers.setdefault(bid, dict(state='far', credited=[]))
            if min(us) - BARREL_R > CAR_X[1]:                  # ahead of the front
                st.update(state='ahead', ok=True, gap=math.inf)
            elif max(us) + BARREL_R < CAR_X[0]:                # behind the rear
                if st['state'] == 'passing' and st['ok'] and st['gap'] < CAR_W \
                        and all(math.cos(heading - h) < 0 for h in st['credited']):
                    st['credited'].append(heading)
                    self.add(P_BARRIER_PASS, f'проезд мимо преграды {bid} в {st["gap"]:.2f} м без касания')
                st['state'] = 'behind'
            elif st['state'] in ('ahead', 'passing') and abs(gap) < 20:
                st['state'] = 'passing'
                st['gap'] = min(st['gap'], gap)
                if bid in touched or v_long < REVERSE_V:
                    st['ok'] = False

    def finish(self, reason):
        if not self.finished:
            self.finished = reason
            self.log(f'[{self.t:7.2f} s] заезд окончен: {reason}. П = {self.premium}, К = {K}, '
                     f'Ш = {self.penalty}, итог {self.total()}')

    def summary(self):
        return dict(rules='РобоКросс.Трасса-2026, 09.04.2026, sections 15, 18', time=round(self.t, 2),
                    finished=self.finished, premium=self.premium, k=K, penalty=self.penalty,
                    total=self.total(), events=self.events)


# ---- Webots side
def main():
    from controller import Supervisor
    sup = Supervisor()
    dt_ms = int(sup.getBasicTimeStep())
    root = sup.getRoot().getField('children')
    gates, fence = {}, []
    for k in range(root.getCount()):
        n = root.getMFNode(k)
        tn = n.getTypeName()
        if tn in ('WinderStart', 'WinderFinish', 'WinderFreeDeal', 'WinderGaz'):
            gates.setdefault(tn, []).append(tuple(n.getPosition()[:2]))
        elif tn == 'SolidBox':
            fence.append(box_of(n, n.getField('size').getSFVec3f()))
    barrier = sup.getFromDef('Barrier')
    ch = barrier.getField('children')
    for k in range(ch.getCount()):
        n = ch.getMFNode(k)
        bo = n.getField('boundingObject').getSFNode()
        if bo is not None and bo.getTypeName() == 'Box':
            fence.append(box_of(n, bo.getField('size').getSFVec3f()))
    car = sup.getFromDef('Vesta1')
    spawn = tuple(car.getPosition()[:2])
    map_yaml = os.environ.get('REFEREE_MAP') or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(sup.getWorldPath()))), 'maps', MAP_YAML)
    if not os.path.exists(map_yaml):
        from ament_index_python.packages import get_package_share_directory
        map_yaml = os.path.join(get_package_share_directory('webots_ros2_suv'), 'maps', MAP_YAML)
    zones = Zones(gates, spawn, map_yaml)
    out = os.environ.get('REFEREE_OUT', '/tmp/referee_results.json')
    score = Score(zones, fence, log=lambda m: print('referee:', m, flush=True))
    print(f'referee: {len(fence)} fence boxes, zones from {sorted(gates)}, results -> {out}', flush=True)

    barrels_id, barrels = None, []
    custom, stale_id = sup.getCustomData(), None
    n_events = -1
    while sup.step(dt_ms) != -1:
        t = sup.getTime()
        # vesta_driver reseeds by removing DEF BARRELS and importing a new one. It first writes
        # "barrels <seed> ..." into this robot's customData and removes a few steps later: from then on the
        # barrels are not read until a new DEF BARRELS is there. Reading removed nodes gives "node not found"
        # warnings and junk positions, which counted as touches of every barrel.
        if sup.getCustomData() != custom:
            custom = sup.getCustomData()
            if custom.startswith('barrels'):
                stale_id, barrels_id, barrels = barrels_id, None, []
        b = sup.getFromDef('BARRELS')
        b_id = b.getId() if b is not None else -1
        if b_id < 0 or b_id == stale_id:
            barrels_id, barrels = None, []
        elif b_id != barrels_id:
            barrels_id, barrels, stale_id = b_id, list_barrels(b), None     # new layout
        states = [barrel_state(bid, n) for bid, n in barrels]
        if not all(math.isfinite(v) for st in states for v in st[1:3]):
            states, barrels_id = [], None                            # removed after all: read the group again
        p, R = car.getPosition(), car.getOrientation()
        yaw = math.atan2(R[3], R[0])
        v = car.getVelocity()
        v_long = v[0] * R[0] + v[1] * R[3]
        score.step(t, dt_ms / 1000.0, (p[0], p[1], yaw), v_long, states)
        if len(score.events) != n_events or score.finished:
            n_events = len(score.events)
            with open(out, 'w') as f:
                json.dump(score.summary(), f, ensure_ascii=False, indent=1)
        hud(sup, score)
        if score.finished:
            break
    while sup.step(dt_ms) != -1:     # keep the final result on screen
        pass


def box_of(node, size):
    p, R = node.getPosition(), node.getOrientation()
    return ((p[0], p[1]), (size[0] / 2, size[1] / 2), math.atan2(R[3], R[0]))


def list_barrels(group_proto):
    ch = group_proto.getBaseNodeField('children') or group_proto.getField('children')
    nodes = [ch.getMFNode(k) for k in range(ch.getCount())]
    pos = [n.getPosition()[:2] for n in nodes]
    bid = list(range(len(nodes)))
    for i in range(len(nodes)):                                   # barrels touching side by side: one barrier
        for j in range(i):
            if math.dist(pos[i], pos[j]) < PAIR_DIST:
                bid[i] = bid[j]
    return list(zip(bid, nodes))


def barrel_state(bid, node):
    p, R = node.getPosition(), node.getOrientation()
    cx = p[0] + R[0] * BARREL_OFFSET[0] + R[1] * BARREL_OFFSET[1]
    cy = p[1] + R[3] * BARREL_OFFSET[0] + R[4] * BARREL_OFFSET[1]
    upright = R[8] > 0.9
    # lying: the cylinder from above is a 1.4 x 0.9 box along its axis (local z)
    box = ((cx, cy), (BARREL_H / 2, BARREL_R), math.atan2(R[5], R[2]))
    return bid, cx, cy, upright, box


def label(sup, k, text, y, size, color):
    """Label with a drop shadow: Webots labels have none, so a black copy goes underneath (lower id)."""
    sup.setLabel(k, text, 0.012, y + 0.003, size, 0x000000, 0.0, 'Arial')
    sup.setLabel(10 + k, text, 0.01, y, size, color, 0.0, 'Arial')


def hud(sup, score):  # bottom left: the top of the view holds the camera overlay
    label(sup, 0, f'РобоКросс.Трасса-2026, судья   t {score.t:6.1f} с', 0.74, 0.06, 0xffffff)
    label(sup, 1, f'П {score.premium}  x К {K}   Ш {score.penalty}   ИТОГ {score.total()}', 0.78, 0.08,
          0x30d158 if score.finished else 0xffffff)
    last = score.events[-3:]
    for k in range(3):   # one label per line: setLabel does not break lines at \n
        text = f"{last[k]['points']:+d} {last[k]['text']}" if k < len(last) else ''
        label(sup, 4 + k, text, 0.83 + 0.035 * k, 0.05, 0xffd60a)
    if score.finished:
        label(sup, 3, f'ЗАЕЗД ОКОНЧЕН: {score.finished}', 0.945, 0.06, 0x30d158)


if __name__ == '__main__':
    sys.exit(main())
