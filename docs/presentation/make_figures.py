"""Figures for the presentation. Run from ros2_ws/src/vesta_rl with the vesta_rl venv:
    python ../../../docs/presentation/make_figures.py
Training-history figures read the running tensorboard at http://localhost:8080.
"""
import json
import math
import os
import urllib.parse
import urllib.request

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vesta_rl import config as C  # noqa: E402
from vesta_rl.obs import PathTracker, speed_limit  # noqa: E402
from vesta_rl.sim2d import Vehicle, VehicleParams  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img')
plt.rcParams.update({'font.size': 13, 'axes.grid': True, 'grid.alpha': 0.3, 'font.family': 'DejaVu Sans'})
BLUE, ORANGE, GREEN, RED, GRAY = '#2563eb', '#ea580c', '#16a34a', '#dc2626', '#6b7280'


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=150)
    plt.close(fig)


# 1. inflation cost
d = np.linspace(0, 2.5, 500)
cost = np.where(d <= 0.0, 254, np.where(d <= C.INSCRIBED_RADIUS, 253,
                np.where(d <= C.INFLATION_RADIUS, 252 * np.exp(-C.COST_SCALING_FACTOR * (d - C.INSCRIBED_RADIUS)), 0)))
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(d, cost, color=BLUE, lw=2.5)
ax.axvline(C.INSCRIBED_RADIUS, color=GRAY, ls='--')
ax.axvline(C.INFLATION_RADIUS, color=GRAY, ls='--')
ax.text(C.INSCRIBED_RADIUS + 0.03, 200, 'r_ins = 0.9 м', color=GRAY)
ax.text(C.INFLATION_RADIUS + 0.03, 200, 'r_inf = 2.0 м', color=GRAY)
ax.set_xlabel('расстояние до ближайшего препятствия d, м')
ax.set_ylabel('стоимость клетки c(d)')
ax.set_title('Раздувание (inflation) Nav2, k = 2')
save(fig, 'fig_inflation.png')

# 2. speed limit: v_curve(kappa) and profile approaching a hairpin
fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
k = np.linspace(0.01, 0.26, 300)
axs[0].plot(1 / k, np.minimum(C.MAX_SPEED, np.sqrt(C.A_LAT_MAX / k)), color=BLUE, lw=2.5)
axs[0].set_xlabel('радиус поворота R = 1/κ, м')
axs[0].set_ylabel('v_curve, м/с')
axs[0].set_title('Допустимая скорость в повороте: v = √(a_lat·R)')
axs[0].axvline(3.9, color=RED, ls='--')
axs[0].text(5.0, 1.0, 'мин. радиус Весты 3.9 м', color=RED)
axs[0].set_xlim(0, 40)
axs[0].set_ylim(0, 5.5)
r = 4.5
th = np.arange(0, math.pi, 0.5 / r)
arc = np.stack([r * np.sin(th), r - r * np.cos(th)], axis=1)
approach = np.stack([np.arange(-20, 0, 0.5), np.zeros(40)], axis=1)
tr = PathTracker(np.concatenate([approach, arc]))
s = np.arange(0, tr.length, 0.25)
lim = [speed_limit(tr, x) for x in s]
axs[1].plot(s - 20, lim, color=ORANGE, lw=2.5, label='лимит (поворот + финиш)')
axs[1].axvspan(0, math.pi * r, color=BLUE, alpha=0.08, label='поворот R = 4.5 м')
axs[1].set_xlabel('положение вдоль пути, м (0 = начало поворота)')
axs[1].set_ylabel('v_limit, м/с')
axs[1].set_title('Профиль лимита: торможение заранее и к финишу')
axs[1].set_ylim(0, 5.5)
axs[1].legend(loc='lower left')
save(fig, 'fig_speedlimit.png')

# 3. saturation: plain gaussian mean vs tanh-bounded mean
fig, axs = plt.subplots(1, 2, figsize=(13, 4.3))
x = np.linspace(-1.5, 7.5, 600)
for ax, mu, sd, title in ((axs[0], 4.6, 0.7, 'Было: μ = 4.6 (линейный выход)'),
                          (axs[1], math.tanh(0.95), 0.5, 'Стало: μ = tanh(z) ∈ (−1, 1)')):
    pdf = np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
    ax.fill_between(x, pdf, where=x > 1, color=RED, alpha=0.35, label='обрезается до a = 1 (полный газ)')
    ax.fill_between(x, pdf, where=x <= 1, color=GREEN, alpha=0.35, label='реально разные скорости')
    ax.plot(x, pdf, color='k', lw=1.5)
    ax.axvline(1, color=RED, ls='--')
    ax.axvline(-1, color=GRAY, ls=':')
    share = np.mean(np.random.default_rng(0).normal(mu, sd, 200000) < 1)
    ax.set_title(f'{title}\nдоля проб ниже предела: {share:.1%}')
    ax.set_xlabel('действие a_v (до обрезки)')
    ax.legend(loc='upper right', fontsize=10)
save(fig, 'fig_saturation.png')

# 4. PPO clipped objective
eps = 0.2
rr = np.linspace(0.4, 1.6, 400)
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for ax, A, title in ((axs[0], 1.0, 'Â > 0 (действие лучше среднего)'), (axs[1], -1.0, 'Â < 0 (хуже среднего)')):
    ax.plot(rr, rr * A, color=GRAY, ls='--', label='r·Â')
    ax.plot(rr, np.minimum(rr * A, np.clip(rr, 1 - eps, 1 + eps) * A), color=BLUE, lw=2.5, label='L^CLIP')
    ax.axvline(1 - eps, color=RED, ls=':')
    ax.axvline(1 + eps, color=RED, ls=':')
    ax.set_xlabel('r = π_θ(a|s) / π_old(a|s)')
    ax.set_title(title)
    ax.legend()
save(fig, 'fig_ppo_clip.png')

# 5. vehicle lag: step response nominal vs randomized
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
rng = np.random.default_rng(1)
t = np.arange(0, 6, C.DT)
for i in range(12):
    p = VehicleParams.random(rng)
    car = Vehicle(p)
    v, dl = [], []
    for _ in t:
        car.step(3.0, 0.4)
        v.append(car.v)
        dl.append(car.delta)
    axs[0].plot(t, v, color=GRAY, alpha=0.5, lw=1)
    axs[1].plot(t, dl, color=GRAY, alpha=0.5, lw=1)
car = Vehicle(VehicleParams())
v, dl = [], []
for _ in t:
    car.step(3.0, 0.4)
    v.append(car.v)
    dl.append(car.delta)
axs[0].plot(t, v, color=BLUE, lw=2.5, label='номинал')
axs[1].plot(t, dl, color=BLUE, lw=2.5, label='номинал')
axs[0].set_title('Скорость при команде 3 м/с')
axs[1].set_title('Угол руля при команде 0.4 рад')
for ax in axs:
    ax.set_xlabel('t, с')
    ax.legend()
axs[0].set_ylabel('v, м/с')
axs[1].set_ylabel('δ, рад')
save(fig, 'fig_lag.png')

# 6. bicycle model sketch
fig, ax = plt.subplots(figsize=(7, 5))
L, yaw, delta = C.WHEELBASE, 0.35, 0.45
rear = np.array([0.0, 0.0])
front = rear + L * np.array([math.cos(yaw), math.sin(yaw)])
ax.plot(*zip(rear, front), color='k', lw=3)
for c0, ang, col in ((rear, yaw, 'k'), (front, yaw + delta, ORANGE)):
    dvec = 0.5 * np.array([math.cos(ang), math.sin(ang)])
    ax.plot(*zip(c0 - dvec, c0 + dvec), color=col, lw=8, solid_capstyle='butt')
R = L / math.tan(delta)
icr = rear + R * np.array([-math.sin(yaw), math.cos(yaw)])
ax.plot(*icr, 'o', color=RED)
ax.plot(*zip(rear, icr), color=RED, ls='--', lw=1)
ax.plot(*zip(front, icr), color=RED, ls='--', lw=1)
ax.text(*(icr + [0.15, 0]), 'мгновенный центр\nповорота (ICR)', color=RED)
ax.text(*(rear + [-0.2, -0.45]), 'задняя ось = base_link (x, y)', fontsize=11)
ax.text(*(front + [0.1, 0.1]), 'δ', color=ORANGE, fontsize=16)
ax.text(*((rear + front) / 2 + [-0.4, 0.2]), 'L = 2.648 м', fontsize=11)
ax.text(*((rear + icr) / 2 + [0.1, -0.2]), 'R = L / tan δ', color=RED, fontsize=11)
ax.annotate('', xy=rear + [1.2, 0], xytext=rear, arrowprops=dict(arrowstyle='->', color=GRAY))
ax.text(*(rear + [0.6, 0.12]), 'ψ', color=GRAY, fontsize=14)
ax.set_aspect('equal')
ax.set_xlim(-3.2, 4.0)
ax.set_ylim(-1, 6.2)
ax.axis('off')
save(fig, 'fig_bicycle.png')


# 7. training history from tensorboard
def tb(tag):
    url = 'http://localhost:8080/data/plugin/scalars/scalars?' + urllib.parse.urlencode({'run': 'sim2d_1', 'tag': tag})
    try:
        d = json.load(urllib.request.urlopen(url, timeout=5))
        return np.array([p[1] / 1e6 for p in d]), np.array([p[2] for p in d])
    except Exception:
        return np.array([]), np.array([])


events = [(1.27, '15 препятствий'), (2.245, 'процедурные\nтрассы'), (4.12, 'лимит скорости\nв награде'),
          (5.81, 'учебный\nплан'), (8.79, 'tanh'), (17.93, 'газ = доля\nлимита')]
st, sr = tb('rollout/success_rate')
if len(st):
    fig, axs = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    sm = np.convolve(sr, np.ones(15) / 15, mode='same')
    axs[0].plot(st, sr, color=BLUE, alpha=0.25, lw=1)
    axs[0].plot(st[7:-7], sm[7:-7], color=BLUE, lw=2.2, label='success_rate (скользящее среднее)')
    axs[0].set_ylabel('доля успешных\nэпизодов')
    lt, lv = tb('curriculum/level')
    axs[1].step(lt, lv, color=ORANGE, lw=2.2, where='post')
    axs[1].set_ylabel('уровень\nучебного плана')
    axs[1].set_xlabel('шаги среды, млн')
    for i, (x0, lab) in enumerate(events):
        for ax in axs:
            ax.axvline(x0, color=RED, ls='--', lw=0.9)
        axs[0].text(x0 + 0.1, 0.84 if i % 2 == 0 else 0.30, lab, color=RED, fontsize=9, va='top')
    axs[0].set_ylim(0.2, 0.86)
    axs[0].legend(loc='lower right')
    save(fig, 'fig_history.png')

# 8. evaluation of checkpoints (30 + 30 fixed episodes, deterministic policy)
ev = [('3.2 млн', 63, 20), ('5.1 млн', 53, 33), ('7.8 млн', 70, 37), ('10.8 млн', 60, 33), ('16.8 млн', 77, 23)]
fig, ax = plt.subplots(figsize=(10, 4))
xx = np.arange(len(ev))
ax.bar(xx - 0.2, [e[1] for e in ev], 0.4, color=BLUE, label='robocross')
ax.bar(xx + 0.2, [e[2] for e in ev], 0.4, color=ORANGE, label='процедурные')
for i, e in enumerate(ev):
    ax.text(i - 0.2, e[1] + 1, f'{e[1]}%', ha='center', fontsize=10)
    ax.text(i + 0.2, e[2] + 1, f'{e[2]}%', ha='center', fontsize=10)
ax.set_xticks(xx, [e[0] for e in ev])
ax.set_ylabel('доля доездов, %')
ax.set_ylim(0, 90)
ax.set_title('Оценка чекпоинтов: 30 + 30 одинаковых эпизодов, полная сложность')
ax.legend()
save(fig, 'fig_eval.png')
print('figures written to', OUT)
