import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("robot/dummy_lamp_5dof.urdf")
data = mujoco.MjData(model)

for i in range(model.njnt):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()