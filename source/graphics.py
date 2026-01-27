import numpy as np
from matplotlib import pyplot as plt

from source.data_classes import SimLogs


def show_graphics(logs: SimLogs):
    e_pos = np.asarray(logs.e_pos)

    plt.figure()
    plt.plot(logs.t, e_pos[:, 0], label="e_x")
    plt.plot(logs.t, e_pos[:, 1], label="e_y")
    plt.plot(logs.t, e_pos[:, 2], label="e_z")
    plt.title("Position errors")
    plt.xlabel("t, s")
    plt.ylabel("m")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(logs.t, logs.s_x, label="s_x")
    plt.plot(logs.t, logs.s_y, label="s_y")
    plt.plot(logs.t, logs.s_z, label="s_z")
    plt.title("Sliding surfaces (position)")
    plt.xlabel("t, s")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(logs.t, logs.s_psi, label="s_psi")
    plt.plot(logs.t, logs.s_theta, label="s_theta")
    plt.plot(logs.t, logs.s_phi, label="s_phi")
    plt.title("Sliding surfaces (orientation)")
    plt.xlabel("t, s")
    plt.grid(True)
    plt.legend()

    plt.show()
