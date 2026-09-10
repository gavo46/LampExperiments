def choose_behavior(state):
    """
    Decide what the character should do based
    on its current state.

    This is intentionally simple for now.
    """

    if not state.person_present:
        return "neutral"

    if state.attention:
        return "alert"

    return "neutral"