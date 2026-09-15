"""
Turns whatever the camera is currently looking at into a remembered
object: grabs the latest frame the vision thread has captured, asks a
local Ollama vision model to describe it, and stores the description
in shared state's object memory so later conversation turns can refer
back to it (see speech/think.py, which folds the memory snapshot into
the LLM prompt).
"""

import cv2
import ollama

VISION_MODEL = "llama3.2-vision"
DESCRIBE_PROMPT = (
    "A person is holding an object up to the camera. Identify only the object they are holding, in one short phrase. Do not describe the person, their face, or the background."
)


def describe_frame(frame, prompt=DESCRIBE_PROMPT):
    """
    Ask the local Ollama vision model to describe `frame` (a BGR image,
    e.g. straight from vision/camera.py). Returns a short text
    description, or None if the call fails or the model has nothing to
    say.
    """
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    cv2.imwrite("debug_frame.jpg", frame)
    try:
        response = ollama.chat(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [buffer.tobytes()],
            }],
        )
    except Exception as exc:
        print(f"[scene] vision model call failed: {exc}")
        return None

    description = response["message"]["content"].strip()
    return description or None


def remember_current_view(state, name=None):
    """
    Capture whatever the vision thread last saw (state.get_last_frame()),
    describe it, and store it in `state`'s object memory.

    Returns (name, description) on success, or None if there's no
    frame yet or the description failed. `name` is the memory key; if
    omitted, one is generated so repeated calls don't overwrite each
    other.
    """
    frame = state.get_last_frame()
    if frame is None:
        return None

    description = describe_frame(frame)
    if not description:
        return None

    if name is None:
        name = f"object_{len(state.get_memory_snapshot()) + 1}"

    state.remember_object(name, description)
    return name, description
