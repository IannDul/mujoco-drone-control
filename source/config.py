from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelNames:
    #  Path to \mujoco_menagerie\bitcraze_crazyflie_2\scene.xml
    model_path: str = r""

    quat_sensor: str = "body_quat"
    gyro_sensor: str = "body_gyro"

    camera: str = "track"

    body_thrust: str = "body_thrust"
    mx: str = "x_moment"
    my: str = "y_moment"
    mz: str = "z_moment"


@dataclass(frozen=True)
class PhysicsParams:
    mass: float = 0.027
    g: float = 9.81

    jx: float = 2.3951e-5
    jy: float = 2.3951e-5
    jz: float = 3.2347e-5

    gx: float = -0.00001
    gy: float = -0.00001
    gz: float = -0.00001


@dataclass(frozen=True)
class ControlLimits:
    max_tilt: float = float(np.deg2rad(20.0))
    max_u_xy: float = 1.0
