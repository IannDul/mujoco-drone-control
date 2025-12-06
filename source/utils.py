import mujoco.viewer


def look_at_xz(v: mujoco.viewer.Handle):
    v.cam.lookat = [0, 0, 0.1]
    v.cam.distance = 0.75
    v.cam.elevation = 0


if __name__ == '__main__':
    pass
