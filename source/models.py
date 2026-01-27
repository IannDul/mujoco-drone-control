from dataclasses import dataclass, field
from typing import List, Tuple

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
    xyz: Tuple[float, float, float]
    yaw: float


@dataclass(frozen=True)
class SMCGains:
    k: float
    eps: float
    kp: float
    kd: float


@dataclass
class SimLogs:
    t: List[float] = field(default_factory=list)

    s_x: List[float] = field(default_factory=list)
    s_y: List[float] = field(default_factory=list)
    s_z: List[float] = field(default_factory=list)

    s_phi: List[float] = field(default_factory=list)
    s_theta: List[float] = field(default_factory=list)
    s_psi: List[float] = field(default_factory=list)

    e_pos: List[Tuple[float, float, float]] = field(default_factory=list)

    def append(self,
               cur_t: float,
               s_x: float, s_y: float, s_z: float,
               s_phi: float, s_theta: float, s_psi,
               e_pos: Tuple[float, float, float]
               ):
        self.t.append(cur_t)
        self.s_x.append(s_x)
        self.s_y.append(s_y)
        self.s_z.append(s_z)
        self.s_phi.append(s_phi)
        self.s_theta.append(s_theta)
        self.s_psi.append(s_psi)
        self.e_pos.append(e_pos)
