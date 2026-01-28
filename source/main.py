import mujoco
import mujoco.viewer
import numpy as np

from source.config import ModelNames, PhysicsParams, ControlLimits
from source.console_reader import start_console_reader_thread
from source.controller import SMCController
from source.graphics import show_graphics
from source.models import Filter, Position, SMCGains, SimLogs
from source.wp_queue import WayPointsQueue


def run() -> SimLogs:
    names = ModelNames()
    physics = PhysicsParams()
    limits = ControlLimits()
    model = mujoco.MjModel.from_xml_path(names.model_path)
    data = mujoco.MjData(model)
    logs = SimLogs()

    initial_position = Position((0.0, 0.0, 1.0), 0)

    stable_filter = Filter(omega=2,
                           ksi=1,
                           x=np.array(initial_position.xyz, dtype=float),
                           dx=np.zeros(3, dtype=float)
                           )

    smc_xy = SMCGains(kp=1.2, kd=1, k=0.8, eps=0.10)
    smc_z = SMCGains(kp=1.8, kd=1, k=1.2, eps=0.08)
    smc_phi = SMCGains(kp=0.8, kd=1, k=0.20, eps=0.05)
    smc_theta = SMCGains(kp=0.8, kd=1, k=0.20, eps=0.05)
    smc_psi = SMCGains(kp=0.6, kd=1, k=0.15, eps=0.05)

    wp_queue = WayPointsQueue(initial_position)

    start_console_reader_thread(wp_queue)

    controller = SMCController(model, names, physics,
                               limits, wp_queue, stable_filter,
                               smc_xy, smc_z, smc_phi, smc_theta, smc_psi)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, names.camera)

        while viewer.is_running():
            log = controller.compute(data)
            logs.append(data.time,
                        log.get('s_x', 0), log.get('s_z', 0), log.get('s_z', 0),
                        log.get('s_phi', 0), log.get('s_theta', 0), log.get('s_psi', 0),
                        log.get('e_pos', (0, 0, 0))
                        )

            mujoco.mj_step(model, data)
            viewer.sync()

    return logs


if __name__ == "__main__":
    simLogs: SimLogs = run()
    show_graphics(simLogs)
