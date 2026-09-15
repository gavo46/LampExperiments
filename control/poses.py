poses = {
    "neutral": [0.0, 0.4, 1.6, 0.0, -2.0],
    "alert": [0.0, 0.1, 0.9, 0.0, -1.5],
    # Same standing posture as "alert" - character/behavior.py animates
    # the head joint (index 4) on top of this baseline to nod in time
    # with the music, only while it's actually playing.
    "music-alert": [0.0, 0.1, 0.9, 0.0, -1.5],
}
