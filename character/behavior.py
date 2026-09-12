from control.poses import poses


def choose_pose(mode, face_present, face_x):
    """
    STUB - fill this in.

    Decide what pose/gesture the lamp should move toward for the
    current mode (one of the MODE_* constants in character/state.py:
    idle, engaged, listening, thinking, speaking). `face_present` and
    `face_x` are the latest vision readings, in case a mode's pose
    should react to where the face is.

    Return a 5-element joint target list - see control/poses.py for
    examples, or build your own. All timing/easing/expressiveness is
    up to you and control/controller.py's move_toward_target(); this
    function only needs to decide *what* to move toward.

    For now every mode maps to "neutral" so the sim loop always has a
    valid target while you build this out.
    """
    return list(poses["neutral"])
