import mujoco.viewer
import numpy as np


def wrap_to_pi(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def clip_ctrl(model: mujoco.MjModel, act_id: int, u: float) -> float:
    min_v, max_v = model.actuator_ctrlrange[act_id]
    return float(np.clip(u, min_v, max_v))


def sat(s: float, eps: float) -> float:
    if s > eps:
        return 1.0
    if s < -eps:
        return -1.0
    return s / eps


if __name__ == '__main__':
    pass
