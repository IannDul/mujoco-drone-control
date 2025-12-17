import mujoco
import mujoco.viewer
import numpy as np
from matplotlib import pyplot as plt
from transforms3d.euler import quat2euler

from source.data_classes import Filter, Position, SMCGains
from source.utils import clip_ctrl, wrap_to_pi, sat

#  Path to \mujoco_menagerie\bitcraze_crazyflie_2\scene.xml
MODEL_PATH = r""

# sensors
QUAT_SENSOR = "body_quat"
GYRO_SENSOR = "body_gyro"

# actuators
BODY_THRUST = "body_thrust"
MX = "x_moment"
MY = "y_moment"
MZ = "z_moment"


# TODO: kp -> lambda
# TODO: compress loop
def main() -> None:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    m = 0.027
    g = 9.81
    jx, jy, jz = 2.3951e-5, 2.3951e-5, 3.2347e-5
    dt = float(model.opt.timestep)
    gx, gy, gz = -0.00001, -0.00001, -0.00001

    id_thrust = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, BODY_THRUST)
    id_mx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, MX)
    id_my = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, MY)
    id_mz = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, MZ)

    desires_positions = [
        Position(np.array([4.0, 4.0, 1.0]), 0),
        Position(np.array([4.0, -4.0, 4.0]), np.pi / 4),
        Position(np.array([-4.0, -4.0, 3.0]), 0),
        Position(np.array([-4.0, 4.0, 2.0]), 0)
    ]

    pos_idx = 0

    stable_filter = Filter(omega=2, ksi=1, x=desires_positions[0].xyz, dx=np.zeros(3, dtype=float))

    max_tilt = np.deg2rad(20.0)
    max_u_xy = 1.0

    smc_xy = SMCGains(kp=1.2, kd=1, k=0.8, eps=0.10)
    smc_z = SMCGains(kp=1.8, kd=1, k=1.2, eps=0.08)
    smc_phi = SMCGains(kp=0.8, kd=1, k=0.20, eps=0.05)
    smc_theta = SMCGains(kp=0.8, kd=1, k=0.20, eps=0.05)
    smc_psi = SMCGains(kp=0.6, kd=1, k=0.15, eps=0.05)

    phi_des_prev = 0.0
    theta_des_prev = 0.0
    psi_des_prev = desires_positions[0].yaw

    log_sx = []
    log_sy = []
    log_sz = []
    log_s_phi = []
    log_s_theta = []
    log_s_psi = []
    log_ep = []
    log_t = []

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

            p_d, v_d, a_d = stable_filter.generate_signal(desires_positions[pos_idx].xyz, dt)

            e_p = p_d - pos
            e_v = v_d - vel

            # SMC
            s_x = smc_xy.kd * e_v[0] + smc_xy.kp * e_p[0]
            s_y = smc_xy.kd * e_v[1] + smc_xy.kp * e_p[1]

            # virtual u
            ux = a_d[0] + smc_xy.kp * e_v[0] + smc_xy.k * sat(s_x, smc_xy.eps)
            uy = a_d[1] + smc_xy.kp * e_v[1] + smc_xy.k * sat(s_y, smc_xy.eps)

            ux = float(np.clip(ux, -max_u_xy, max_u_xy))
            uy = float(np.clip(uy, -max_u_xy, max_u_xy))

            cos_psi = np.cos(desires_positions[pos_idx].yaw)
            sin_psi = np.sin(desires_positions[pos_idx].yaw)
            theta_des = (ux * cos_psi + uy * sin_psi) / g
            phi_des = (ux * sin_psi - uy * cos_psi) / g

            theta_des = float(np.clip(theta_des, -max_tilt, max_tilt))
            phi_des = float(np.clip(phi_des, -max_tilt, max_tilt))
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
            thrust = m * (g + uz) / tilt_comp
            thrust = clip_ctrl(model, id_thrust, thrust)

            tau_x = jx * u_phi
            tau_y = jy * u_theta
            tau_z = jz * u_psi

            # torque = ctrl * gear
            ctrl_mx = tau_x / gx
            ctrl_my = tau_y / gy
            ctrl_mz = tau_z / gz

            ctrl_mx = clip_ctrl(model, id_mx, ctrl_mx)
            ctrl_my = clip_ctrl(model, id_my, ctrl_my)
            ctrl_mz = clip_ctrl(model, id_mz, ctrl_mz)

            # Control
            data.ctrl[id_thrust] = thrust
            data.ctrl[id_mx] = ctrl_mx
            data.ctrl[id_my] = ctrl_my
            data.ctrl[id_mz] = ctrl_mz

            log_t.append(len(log_t) * dt)
            log_ep.append(e_p)
            log_sx.append(s_x)
            log_sy.append(s_y)
            log_sz.append(s_z)
            log_s_psi.append(s_psi)
            log_s_theta.append(s_theta)
            log_s_phi.append(s_phi)

            mujoco.mj_step(model, data)
            viewer.sync()

    ep_arr = np.asarray(log_ep)
    sx_arr = np.asarray(log_sx)
    sy_arr = np.asarray(log_sy)
    sz_arr = np.asarray(log_sz)
    s_psi_arr = np.asarray(log_s_psi)
    s_theta_arr = np.asarray(log_s_theta)
    s_phi_arr = np.asarray(log_s_phi)

    plt.figure()
    plt.plot(log_t, ep_arr[:, 0], label="e_x")
    plt.plot(log_t, ep_arr[:, 1], label="e_y")
    plt.plot(log_t, ep_arr[:, 2], label="e_z")
    plt.title("Position errors")
    plt.xlabel("t, s")
    plt.ylabel("m")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(log_t, sx_arr, label="s_x")
    plt.plot(log_t, sy_arr, label="s_y")
    plt.plot(log_t, sz_arr, label="s_z")
    plt.title("Sliding surfaces (position)")
    plt.xlabel("t, s")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(log_t, s_psi_arr, label="s_psi")
    plt.plot(log_t, s_theta_arr, label="s_theta")
    plt.plot(log_t, s_phi_arr, label="s_phi")
    plt.title("Sliding surfaces (orientation)")
    plt.xlabel("t, s")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
