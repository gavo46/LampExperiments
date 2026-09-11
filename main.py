import mujoco
import mujoco.viewer

from control.poses import poses
from control.controller import move_toward_target


def main():
    model = mujoco.MjModel.from_xml_path("robot/dummy_lamp_5dof.urdf")
    data = mujoco.MjData(model)

    target = poses["neutral"]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():

            move_toward_target(data, target)

            mujoco.mj_forward(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()