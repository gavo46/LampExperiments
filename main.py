import mujoco
import mujoco.viewer
import math

model = mujoco.MjModel.from_xml_path("robot/dummy_lamp_5dof.urdf")
data = mujoco.MjData(model)

poses = {
    # Relaxed baseline
    "neutral": [0.0, 0.4, 1.6, 0.0, -2.0],

    # Base/head yaw
    "look_left":  [0.6, 0.4, 1.6, 0.0, -2.0],
    "look_right": [-0.6, 0.4, 1.6, 0.0, -2.0],

    # Head yaw
    "head_left":  [0.0, 0.4, 1.6, 0.7, -2.0],
    "head_right": [0.0, 0.4, 1.6, -0.7, -2.0],

    # Head pitch
    "head_up":   [0.0, 0.4, 1.6, 0.0, -1.3],
    "head_down": [0.0, 0.4, 1.6, 0.0, -2.7],

    # Arm configurations
    "arm_up":       [0.0, 0.0, 1.0, 0.0, -2.0],
    "arm_down":     [0.0, 0.8, 2.0, 0.0, -2.0],
    "arm_extended": [0.0, 0.8, 0.3, 0.0, -2.0],
    "arm_bent":     [0.0, 0.2, 2.2, 0.0, -2.0],

    # Coordinated poses
    "alert": [
        0.0, 0.1, 0.9, 0.0, -1.5
    ],

    "look_left_up": [
        0.5, 0.4, 1.6, 0.5, -1.4
    ],

    "look_right_up": [
        -0.5, 0.4, 1.6, -0.5, -1.4
    ],

    "look_left_down": [
        0.5, 0.4, 1.6, 0.5, -2.5
    ],

    "look_right_down": [
        -0.5, 0.4, 1.6, -0.5, -2.5
    ],
}

# for i in range(model.njnt):
#     print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))

#t = 0

target = poses["neutral"]
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        #t += .001

        data.qpos[0] = 0.0    
        data.qpos[1] = 0.4    
        data.qpos[2] = 1.6    
        data.qpos[3] = 0    
        data.qpos[4] = -2
        mujoco.mj_forward(model, data)
        viewer.sync()