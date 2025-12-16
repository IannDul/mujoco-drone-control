from dataclasses import dataclass, field

import numpy as np


@dataclass
class Filter:
    omega: float
    ksi: float
    x: np.ndarray
    dx: np.ndarray

    def generate_signal(self, x_des: np.ndarray, dt: float):
        ddx = (self.omega ** 2) * (x_des - self.x) - 2 * self.ksi * self.omega * self.dx
        self.dx = self.dx + ddx * dt
        self.x = self.x + self.dx * dt

        return self.x, self.dx, ddx


@dataclass(frozen=True)
class Position:
    xyz: np.ndarray
    yaw: float


@dataclass(frozen=True)
class SMCGains:
    k: float
    eps: float
    kp: float
    kd: float
