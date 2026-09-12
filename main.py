import threading
import time

import mujoco
import mujoco.viewer

import config
from character.behavior import choose_pose
from character.state import SharedState, MODE_LISTENING, MODE_THINKING, MODE_SPEAKING
from control.controller import move_toward_target
from speech import think
from speech.listen import listen_loop
from speech.speak import speak
from vision import scene, tracking

# Simple keyword heuristic for "go look at what's in front of you and
# remember it." Replace/extend however you like later.
SCENE_TRIGGERS = ("what is this", "what's this", "look at this", "remember this")

# Guards against a second utterance kicking off a new
# Ollama/speak pipeline while one is already in flight.
_pipeline_lock = threading.Lock()


def _on_speech_start(state):
    if _pipeline_lock.locked():
        return  # already mid-conversation; don't interrupt the pipeline's own mode changes
    state.set_mode(MODE_LISTENING)


def _on_speech_end(state):
    if _pipeline_lock.locked():
        return
    state.return_to_resting()


def _on_utterance(state, text):
    if not _pipeline_lock.acquire(blocking=False):
        print("[main] still handling the previous utterance, ignoring:", text)
        return
    state.set_mode(MODE_THINKING)
    threading.Thread(target=_run_pipeline, args=(state, text), daemon=True).start()


def _run_pipeline(state, text):
    """text -> (optional scene lookup) -> Ollama -> speech.

    Runs entirely off the sim thread so Whisper (already off-thread in
    speech/listen.py) and the LLM/TTS calls here never block the sim loop.
    """
    try:
        if any(trigger in text.lower() for trigger in SCENE_TRIGGERS):
            scene.remember_current_view(state)

        memory_snapshot = state.get_memory_snapshot()
        reply = think.respond(text, memory_snapshot)

        state.set_mode(MODE_SPEAKING)
        speak(reply)
    finally:
        state.return_to_resting()
        _pipeline_lock.release()


def main():
    state = SharedState()

    model = mujoco.MjModel.from_xml_path("robot/dummy_lamp_5dof.urdf")
    data = mujoco.MjData(model)

    threading.Thread(
        target=tracking.watch, args=(state, config.CAMERA_ID), daemon=True
    ).start()

    threading.Thread(
        target=listen_loop,
        args=(lambda text: _on_utterance(state, text),),
        kwargs={
            "on_speech_start": lambda: _on_speech_start(state),
            "on_speech_end": lambda: _on_speech_end(state),
        },
        daemon=True,
    ).start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mode = state.get_mode()
            face_present, face_x = state.get_face()

            # --- POSE STUB: character/behavior.py:choose_pose() ---
            # Mode -> actual pose/gesture, and all timing/expressiveness,
            # is intentionally left to you to fill in there.
            target = choose_pose(mode, face_present, face_x)

            move_toward_target(data, target)
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
