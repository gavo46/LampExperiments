from control.poses import poses
import math
import time
from character.state import MODE_IDLE, MODE_ENGAGED, MODE_LISTENING, MODE_THINKING, MODE_SPEAKING, MODE_DANCING

def choose_pose(mode, face_present, face_x, beat_phase=0.0):
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
    if mode == MODE_DANCING:
        target = list(poses["neutral"])

        if face_present:
            target[0] = (face_x - 0.5) * -2.0

        beat_number = int(beat_phase)
        beat_position = beat_phase - beat_number

        # Asymmetric nod: a short fast drop, then a longer eased rise
        # back to neutral - not a symmetric sine. drop_fraction is how
        # much of the beat is spent going down; the rest is the rise.
        drop_fraction = 0.2
        if beat_position < drop_fraction:
            envelope = beat_position / drop_fraction  # quick linear drop
        else:
            rise_t = (beat_position - drop_fraction) / (1.0 - drop_fraction)
            envelope = (1.0 - rise_t) ** 2  # slower, easing rise back to neutral

        # Mark every 8th beat (the top of a two-bar phrase at 4/4) with
        # a bigger nod so the motion has shape instead of being uniform.
        nod_amplitude = 0.06
        accent_multiplier = 1.6
        if beat_number % 8 == 0:
            nod_amplitude *= accent_multiplier

        target[4] = poses["neutral"][4] - nod_amplitude * envelope
        return target
    return list(poses["neutral"])
