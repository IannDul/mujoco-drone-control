from dataclasses import dataclass
from typing import List

import mujoco
import numpy as np
from transforms3d.euler import quat2euler

from source.config import ModelNames, PhysicsParams, ControlLimits
from source.models import Position, Filter, SMCGains
from source.utils import wrap_to_pi, sat, clip_ctrl


@dataclass(frozen=True)
class ModelActuators:
    thrust: int
    mx: int
    my: int
    mz: int


class SMCController:
    def __init__(self,
                 model: mujoco.MjModel,
                 names: ModelNames,
                 physics: PhysicsParams,
                 limits: ControlLimits,
                 waypoints: List[Position],
                 stable_filter: Filter,
                 smc_xy: SMCGains,
                 smc_z: SMCGains,
                 smc_phi: SMCGains,
                 smc_theta: SMCGains,
                 smc_psi: SMCGains):
        self._model = model
        self._names = names
        self._physics = physics
        self._limits = limits

        self._waypoints = list(waypoints)
        if not self._waypoints:
            raise ValueError("Waypoints must not be empty")

        self._pos_idx = 0
        self._filter = stable_filter

        self._smc_xy = smc_xy
        self._smc_z = smc_z
        self._smc_phi = smc_phi
        self._smc_theta = smc_theta
        self._smc_psi = smc_psi

        self._phi_des_prev = 0.0
        self._theta_des_prev = 0.0
        self._psi_des_prev = self._waypoints[0].yaw

        self._actuators = self.__get_actuators()

        self._dt = float(model.opt.timestep)

    def __get_actuators(self):
        id_thrust = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, self._names.body_thrust)
        id_mx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, self._names.mx)
        id_my = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, self._names.my)
        id_mz = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, self._names.mz)

        return ModelActuators(id_thrust, id_mx, id_my, id_mz)

    def compute(self,
                data: mujoco.MjData) -> dict:
        # state
        x, y, z = data.qpos[0:3]
        v_x, v_y, v_z = data.qvel[0:3]

        quat_val = data.sensor(self._names.quat_sensor).data
        psi, theta, phi = quat2euler(quat_val, axes="szyx")

        w_phi, w_theta, w_psi = data.sensor(self._names.gyro_sensor).data

        pos = np.array([x, y, z], dtype=float)
        vel = np.array([v_x, v_y, v_z], dtype=float)

        # check if point has been achieved
        pos_ref = self._waypoints[self._pos_idx].xyz
        yaw_ref = self._waypoints[self._pos_idx].yaw

        if np.linalg.norm(pos_ref - pos) < 0.08 and \
                np.linalg.norm(vel) < 0.15 and \
                abs(wrap_to_pi(yaw_ref - psi)) < 0.2:
            self._pos_idx = (self._pos_idx + 1) % len(self._waypoints)

        p_d, v_d, a_d = self._filter.generate_signal(
            np.array(self._waypoints[self._pos_idx].xyz, dtype=float), self._dt
        )

        e_p = p_d - pos
        e_v = v_d - vel

        # SMC
        s_x = self._smc_xy.kd * e_v[0] + self._smc_xy.kp * e_p[0]
        s_y = self._smc_xy.kd * e_v[1] + self._smc_xy.kp * e_p[1]

        # virtual u
        ux = a_d[0] + self._smc_xy.kp * e_v[0] + self._smc_xy.k * sat(s_x, self._smc_xy.eps)
        uy = a_d[1] + self._smc_xy.kp * e_v[1] + self._smc_xy.k * sat(s_y, self._smc_xy.eps)

        ux = float(np.clip(ux, -self._limits.max_u_xy, self._limits.max_u_xy))
        uy = float(np.clip(uy, -self._limits.max_u_xy, self._limits.max_u_xy))

        cos_psi = np.cos(self._waypoints[self._pos_idx].yaw)
        sin_psi = np.sin(self._waypoints[self._pos_idx].yaw)
        theta_des = (ux * cos_psi + uy * sin_psi) / self._physics.g
        phi_des = (ux * sin_psi - uy * cos_psi) / self._physics.g

        theta_des = float(np.clip(theta_des, -self._limits.max_tilt, self._limits.max_tilt))
        phi_des = float(np.clip(phi_des, -self._limits.max_tilt, self._limits.max_tilt))
        psi_des = self._waypoints[self._pos_idx].yaw

        phi_des_dot = (phi_des - self._phi_des_prev) / self._dt
        theta_des_dot = (theta_des - self._theta_des_prev) / self._dt
        psi_des_dot = (psi_des - self._psi_des_prev) / self._dt

        self._phi_des_prev = phi_des
        self._theta_des_prev = theta_des
        self._psi_des_prev = psi_des

        e_z = p_d[2] - z
        e_vz = v_d[2] - v_z

        s_z = e_vz + self._smc_z.kp * e_z
        uz = a_d[2] + self._smc_z.kp * e_vz + self._smc_z.k * sat(s_z, self._smc_z.eps)

        e_phi = phi_des - phi
        e_theta = theta_des - theta
        e_psi = wrap_to_pi(psi_des - psi)

        e_phi_dot = phi_des_dot - w_phi
        e_theta_dot = theta_des_dot - w_theta
        e_psi_dot = psi_des_dot - w_psi

        # SMC
        s_phi = self._smc_phi.kd * e_phi_dot + self._smc_phi.kp * e_phi
        s_theta = self._smc_theta.kd * e_theta_dot + self._smc_theta.kp * e_theta
        s_psi = self._smc_psi.kd * e_psi_dot + self._smc_psi.kp * e_psi

        u_phi = self._smc_phi.kp * e_phi_dot + self._smc_phi.k * sat(s_phi, self._smc_phi.eps)
        u_theta = self._smc_theta.kp * e_theta_dot + self._smc_theta.k * sat(s_theta, self._smc_theta.eps)
        u_psi = self._smc_psi.kp * e_psi_dot + self._smc_psi.k * sat(s_psi, self._smc_psi.eps)

        # Real u
        tilt_comp = np.cos(phi) * np.cos(theta)
        tilt_comp = float(np.clip(tilt_comp, 0.2, 1))
        thrust = self._physics.mass * (self._physics.g + uz) / tilt_comp
        thrust = clip_ctrl(self._model, self._actuators.thrust, thrust)

        tau_x = self._physics.jx * u_phi
        tau_y = self._physics.jy * u_theta
        tau_z = self._physics.jz * u_psi

        # torque = ctrl * gear
        ctrl_mx = tau_x / self._physics.gx
        ctrl_my = tau_y / self._physics.gy
        ctrl_mz = tau_z / self._physics.gz

        ctrl_mx = clip_ctrl(self._model, self._actuators.mx, ctrl_mx)
        ctrl_my = clip_ctrl(self._model, self._actuators.my, ctrl_my)
        ctrl_mz = clip_ctrl(self._model, self._actuators.mz, ctrl_mz)

        # Control
        data.ctrl[self._actuators.thrust] = thrust
        data.ctrl[self._actuators.mx] = ctrl_mx
        data.ctrl[self._actuators.my] = ctrl_my
        data.ctrl[self._actuators.mz] = ctrl_mz

        log = {
            's_x': s_x,
            's_y': s_y,
            's_z': s_z,
            's_phi': s_phi,
            's_theta': s_theta,
            's_psi': s_psi,
            'e_pos': e_p
        }

        return log
