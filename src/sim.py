import mujoco
import mujoco.viewer
import math

model = mujoco.MjModel.from_xml_path("robot/dummy_lamp_5dof.urdf")
data = mujoco.MjData(model)

for i in range(model.njnt):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))

t = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t += .0001
        data.qpos[4] = math.sin(t)
        mujoco.mj_forward(model, data)
        viewer.sync()