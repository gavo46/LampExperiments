import subprocess


def speak(text):
    """
    Convert text to speech and play it through the computer's speakers
    via macOS's built-in `say`. Blocks until playback finishes, so
    callers can treat "speak() returned" as "done talking" (e.g. to
    drop back out of a speaking mode).
    """
    print("CHARACTER:", text)
    try:
        subprocess.run(["say", text], check=False)
    except FileNotFoundError:
        # `say` isn't available (not macOS) - text is already printed above.
        pass
