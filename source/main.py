import mujoco
import mujoco.viewer
import numpy as np
from transforms3d.euler import quat2euler

from source.data_classes import Filter, Position, SMCGains, SimLogs
from source.graphics import show_graphics
from source.utils import clip_ctrl, wrap_to_pi, sat

#  Path to \mujoco_menagerie\bitcraze_crazyflie_2\scene.xml
MODEL_PATH = r"D:\PythonProjects\mujoco_menagerie\bitcraze_crazyflie_2\scene.xml"

# sensors
QUAT_SENSOR = "body_quat"
GYRO_SENSOR = "body_gyro"

# camera
CAMERA = "track"

# actuators
BODY_THRUST = "body_thrust"
MX = "x_moment"
MY = "y_moment"
MZ = "z_moment"

# constants
MASS = 0.027
G = 9.81
JX, JY, JZ = 2.3951e-5, 2.3951e-5, 3.2347e-5
GX, GY, GZ = -0.00001, -0.00001, -0.00001

# limits
MAX_TILT = np.deg2rad(20.0)
MAX_U_XY = 1.0


def run() -> SimLogs:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    logs = SimLogs()

    dt = float(model.opt.timestep)

    id_thrust = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, BODY_THRUST)
    id_mx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, MX)
    id_my = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, MY)
    id_mz = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, MZ)

    desires_positions = [
        Position((4.0, 4.0, 1.0), 0),
        Position((4.0, -4.0, 4.0), np.pi / 4),
        Position((-4.0, -4.0, 3.0), 0),
        Position((-4.0, 4.0, 2.0), 0)
    ]

    pos_idx = 0

    stable_filter = Filter(omega=2,
                           ksi=1,
                           x=np.array(desires_positions[0].xyz, dtype=float),
                           dx=np.zeros(3, dtype=float)
                           )

    smc_xy = SMCGains(kp=1.2, kd=1, k=0.8, eps=0.10)
    smc_z = SMCGains(kp=1.8, kd=1, k=1.2, eps=0.08)
    smc_phi = SMCGains(kp=0.8, kd=1, k=0.20, eps=0.05)
    smc_theta = SMCGains(kp=0.8, kd=1, k=0.20, eps=0.05)
    smc_psi = SMCGains(kp=0.6, kd=1, k=0.15, eps=0.05)

    phi_des_prev = 0.0
    theta_des_prev = 0.0
    psi_des_prev = desires_positions[0].yaw

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "track")

        while viewer.is_running():
            # state
            x, y, z = data.qpos[0:3]
            v_x, v_y, v_z = data.qvel[0:3]

            quat_val = data.sensor(QUAT_SENSOR).data
            psi, theta, phi = quat2euler(quat_val, axes="szyx")

            w_phi, w_theta, w_psi = data.sensor(GYRO_SENSOR).data

            pos = np.array([x, y, z], dtype=float)
            vel = np.array([v_x, v_y, v_z], dtype=float)

            # check if point has been achieved
            pos_ref = desires_positions[pos_idx].xyz
            yaw_ref = desires_positions[pos_idx].yaw
            if np.linalg.norm(pos_ref - pos) < 0.08 and \
                    np.linalg.norm(vel) < 0.15 and \
                    abs(wrap_to_pi(yaw_ref - psi)) < 0.2:
                pos_idx = (pos_idx + 1) % len(desires_positions)

            p_d, v_d, a_d = stable_filter.generate_signal(
                np.array(desires_positions[pos_idx].xyz, dtype=float), dt
            )

            e_p = p_d - pos
            e_v = v_d - vel

            # SMC
            s_x = smc_xy.kd * e_v[0] + smc_xy.kp * e_p[0]
            s_y = smc_xy.kd * e_v[1] + smc_xy.kp * e_p[1]

            # virtual u
            ux = a_d[0] + smc_xy.kp * e_v[0] + smc_xy.k * sat(s_x, smc_xy.eps)
            uy = a_d[1] + smc_xy.kp * e_v[1] + smc_xy.k * sat(s_y, smc_xy.eps)

            ux = float(np.clip(ux, -MAX_U_XY, MAX_U_XY))
            uy = float(np.clip(uy, -MAX_U_XY, MAX_U_XY))

            cos_psi = np.cos(desires_positions[pos_idx].yaw)
            sin_psi = np.sin(desires_positions[pos_idx].yaw)
            theta_des = (ux * cos_psi + uy * sin_psi) / G
            phi_des = (ux * sin_psi - uy * cos_psi) / G

            theta_des = float(np.clip(theta_des, -MAX_TILT, MAX_TILT))
            phi_des = float(np.clip(phi_des, -MAX_TILT, MAX_TILT))
            psi_des = desires_positions[pos_idx].yaw

            phi_des_dot = (phi_des - phi_des_prev) / dt
            theta_des_dot = (theta_des - theta_des_prev) / dt
            psi_des_dot = (psi_des - psi_des_prev) / dt

            phi_des_prev = phi_des
            theta_des_prev = theta_des
            psi_des_prev = psi_des

            e_z = p_d[2] - z
            e_vz = v_d[2] - v_z

            s_z = e_vz + smc_z.kp * e_z
            uz = a_d[2] + smc_z.kp * e_vz + smc_z.k * sat(s_z, smc_z.eps)

            e_phi = phi_des - phi
            e_theta = theta_des - theta
            e_psi = wrap_to_pi(psi_des - psi)

            e_phi_dot = phi_des_dot - w_phi
            e_theta_dot = theta_des_dot - w_theta
            e_psi_dot = psi_des_dot - w_psi

            # SMC
            s_phi = smc_phi.kd * e_phi_dot + smc_phi.kp * e_phi
            s_theta = smc_theta.kd * e_theta_dot + smc_theta.kp * e_theta
            s_psi = smc_psi.kd * e_psi_dot + smc_psi.kp * e_psi

            u_phi = smc_phi.kp * e_phi_dot + smc_phi.k * sat(s_phi, smc_phi.eps)
            u_theta = smc_theta.kp * e_theta_dot + smc_theta.k * sat(s_theta, smc_theta.eps)
            u_psi = smc_psi.kp * e_psi_dot + smc_psi.k * sat(s_psi, smc_psi.eps)

            # Real u
            tilt_comp = np.cos(phi) * np.cos(theta)
            tilt_comp = float(np.clip(tilt_comp, 0.2, 1))
            thrust = MASS * (G + uz) / tilt_comp
            thrust = clip_ctrl(model, id_thrust, thrust)

            tau_x = JX * u_phi
            tau_y = JY * u_theta
            tau_z = JZ * u_psi

            # torque = ctrl * gear
            ctrl_mx = tau_x / GX
            ctrl_my = tau_y / GY
            ctrl_mz = tau_z / GZ

            ctrl_mx = clip_ctrl(model, id_mx, ctrl_mx)
            ctrl_my = clip_ctrl(model, id_my, ctrl_my)
            ctrl_mz = clip_ctrl(model, id_mz, ctrl_mz)

            # Control
            data.ctrl[id_thrust] = thrust
            data.ctrl[id_mx] = ctrl_mx
            data.ctrl[id_my] = ctrl_my
            data.ctrl[id_mz] = ctrl_mz

            logs.append(data.time,
                        s_x, s_y, s_z,
                        s_phi, s_theta, s_psi,
                        e_p)

            mujoco.mj_step(model, data)
            viewer.sync()

    return logs


if __name__ == "__main__":
    simLogs: SimLogs = run()
    show_graphics(simLogs)
