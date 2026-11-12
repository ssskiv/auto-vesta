"""Pure logic (no ROS imports): zero-clamp + watchdog, Twist -> steering angle, wheel speed."""
import math


class ZeroClampWatchdog:
    """linear.x = max(linear.x, 0); if no command arrived for `timeout` s the output is a zero command."""

    def __init__(self, timeout=0.3):
        self.timeout = timeout
        self.last_in = None
        self.v, self.w = 0.0, 0.0
        self.clamped = 0

    def update(self, v, w, now):
        self.last_in = now
        if v < 0.0:
            v = 0.0
            self.clamped += 1
        self.v, self.w = v, w
        return v, w

    def output(self, now):
        """(v, omega, stale). Stale -> zero command."""
        if self.last_in is None or now - self.last_in > self.timeout:
            return 0.0, 0.0, True
        return self.v, self.w, False


def steer_from_twist(v, omega, wheelbase, prev_steer, max_steer, v_min=0.1, max_rate=None, dt=None):
    """delta = atan(L*omega/v) for v > v_min, else hold prev_steer. Clamp to +-max_steer, optional rate limit."""
    if v > v_min:
        delta = math.atan(wheelbase * omega / v)
    else:
        delta = prev_steer
    delta = max(-max_steer, min(max_steer, delta))
    if max_rate is not None and dt is not None:
        step = max_rate * dt
        delta = max(prev_steer - step, min(prev_steer + step, delta))
    return delta


class WheelSpeed:
    """Forward speed from cumulative wheel angles (rad): v = r * mean(d theta) / dt."""

    def __init__(self, radius, sign=1.0):
        self.r, self.sign = radius, sign
        self.prev = None

    def update(self, angles, t):
        v = None
        if self.prev is not None and t > self.prev[1]:
            d = [(a - b) for a, b in zip(angles, self.prev[0])]
            v = self.sign * self.r * (sum(d) / len(d)) / (t - self.prev[1])
        self.prev = (list(angles), t)
        return v
