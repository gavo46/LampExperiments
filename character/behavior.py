from control.poses import poses
import math
import time
from character.state import MODE_IDLE, MODE_ENGAGED, MODE_LISTENING, MODE_THINKING, MODE_SPEAKING

def choose_pose(mode, face_present, face_x):
    if mode == MODE_IDLE:
        return list(poses["neutral"])
    if mode == MODE_ENGAGED:
        target = list(poses["alert"])
        target[0] = (face_x - 0.5) * -2.0
        return target
    if mode == MODE_LISTENING:
        target = [0.0, 0.1, 0.9, 1.0, -1.5]
        target[0] = (face_x - 0.5) * -2.0
        return target
    if mode == MODE_THINKING:
        return [0.0, 0.1, 0.9, 1.0, -1.2]
    if mode == MODE_SPEAKING:
        target = list(poses["alert"])
        target[0] = (face_x - 0.5) * -2.0
        target[4] = poses["alert"][4] + 0.15 * math.sin(time.time() * 6)
        return target
    return list(poses["neutral"])
