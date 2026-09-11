import mujoco
import mujoco.viewer
import threading
import time
from control.controller import move_toward_target

model = mujoco.MjModel.from_xml_path("robot/dummy_lamp_5dof.urdf")
data = mujoco.MjData(model)

poses = {
    "neutral": [0.0, 0.4, 1.6, 0.0, -2.0],
    "alert":   [0.0, 0.2, 0.8, 0.0, -1.5],
    "left":    [0.8, 0.4, 1.6, 0.0, -2.0],
    "right":   [-0.8, 0.4, 1.6, 0.0, -2.0],
}

target = poses["neutral"]


def choose_pose():
    global target

    while True:
        choice = input("Pose (neutral/alert/left/right): ")

        if choice in poses:
            target = poses[choice]
        else:
            print("Unknown pose.")


threading.Thread(target=choose_pose, daemon=True).start()


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        move_toward_target(data, target)

        mujoco.mj_forward(model, data)
        viewer.sync()

        time.sleep(0.01)