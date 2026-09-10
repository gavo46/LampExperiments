import mujoco
import mujoco.viewer

from robot_control.poses import poses
from robot_control.controller import move_toward_target
from vision.camera import Camera


MODEL_PATH = "robot/dummy_lamp_6dof.urdf"


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    camera = Camera()

    target = poses["neutral"]

    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():

            frame = camera.read()

            if frame is None:
                break

            # Eventually:
            # person = detect_person(frame)
            # objects = detect_objects(frame)
            # speech = listen()
            # target = choose_behavior(...)

            move_toward_target(data, target)

            mujoco.mj_forward(model, data)
            viewer.sync()

    camera.release()


if __name__ == "__main__":
    main()