"""Abstract drive-by-wire adapter. The real-car adapter (OSCC / ros2cco) waits for decision 7.6: TODO(user)."""
import abc
import logging


class DbwAdapter(abc.ABC):
    @abc.abstractmethod
    def send(self, steer_rad: float, speed_mps: float, accel: float) -> None: ...

    @abc.abstractmethod
    def status(self) -> dict: ...

    @abc.abstractmethod
    def emergency_stop(self) -> None: ...


class LoggingDbwAdapter(DbwAdapter):
    """Stub: logs what it would send. Safe default; never touches hardware."""

    def __init__(self):
        self.log = logging.getLogger('dbw_stub')
        self.last = None
        self.estop = False

    def send(self, steer_rad, speed_mps, accel):
        if self.estop:
            steer_rad, speed_mps, accel = 0.0, 0.0, 0.0
        self.last = (steer_rad, speed_mps, accel)
        self.log.info('DBW stub send steer=%.3f speed=%.2f accel=%.2f', *self.last)

    def status(self):
        return {'engaged': False, 'estop': self.estop, 'last': self.last}

    def emergency_stop(self):
        self.estop = True
        self.last = (0.0, 0.0, 0.0)
        self.log.warning('DBW stub EMERGENCY STOP')
