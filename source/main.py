import mujoco.viewer
from transforms3d.euler import quat2euler

from source import utils

QUAT_SENSOR = 'body_quat'
MODEL_PATH = 'D:\\PythonProjects\\mujoco_menagerie\\bitcraze_crazyflie_2\\scene.xml'

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    utils.look_at_xz(viewer)

    while viewer.is_running():
        x_des, y_des, z_des, psi_des, theta_des, phi_des = (1, 1, 1, 0, 0, 0)

        x, y, z = data.qpos[0:3]
        quat_val = data.sensor(QUAT_SENSOR).data
        psi, theta, phi = quat2euler(quat_val, axes='szyx')

        e_x = x_des - x
        e_y = y_des - y
        e_z = z_des - z
        e_psi = psi_des - psi
        e_theta = theta_des - theta
        e_phi = phi_des - phi

        mujoco.mj_step(model, data)

        viewer.sync()
