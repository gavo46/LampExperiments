def clamp(value, minimum, maximum):
    """
    Restrict value to the range [minimum, maximum].
    """

    return max(minimum, min(value, maximum))