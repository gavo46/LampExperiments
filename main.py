import threading
import time

import mujoco
import mujoco.viewer

import config
from character.behavior import choose_pose
from character.light import LampLight, build_model
from character.music import MusicPlayer
from character.sound import SoundCues
from character.state import (
    SharedState,
    MODE_LISTENING,
    MODE_THINKING,
    MODE_SPEAKING,
    MOOD_PLEASANT,
    MOOD_CONFUSED,
    MOOD_PASSIONATE,
    MOOD_REASSURING,
)
from control.controller import move_toward_target
from speech import think
from speech.listen import listen_loop
from speech.speak import speak
from utils import metrics
from vision import scene, tracking

# Simple keyword heuristic for "go look at what's in front of you and
# remember it." Replace/extend however you like later.
SCENE_TRIGGERS = ("what is this", "what's this", "look at this", "remember this")

# Same pattern, for starting/stopping the music (see character/music.py).
MUSIC_START_TRIGGERS = ("play music", "play some music", "let's jam", "lets jam")
MUSIC_STOP_TRIGGERS = ("stop the music", "stop playing", "turn off the music")

# mood -> light color. Pick your own values; these are just a starting point.
MOOD_COLORS = {
    MOOD_PLEASANT: (1.0, 0.85, 0.4),     # warm yellow
    MOOD_CONFUSED: (0.3, 0.5, 1.0),      # blue
    MOOD_PASSIONATE: (1.0, 0.15, 0.15),  # red
    MOOD_REASSURING: (0.25, 0.85, 0.4),  # green
}

# Guards against a second utterance kicking off a new
# Ollama/speak pipeline while one is already in flight.
_pipeline_lock = threading.Lock()


def _on_speech_start(state, music_player):
    if _pipeline_lock.locked():
        return  # already mid-conversation; don't interrupt the pipeline's own mode changes
    if music_player.is_playing():
        # The music itself is loud and continuous enough to cross
        # listen.py's amplitude threshold on its own - don't let that
        # false trigger visually knock it out of dancing. A genuine
        # command (e.g. "stop the music") still reaches _on_utterance
        # regardless, since that path doesn't go through here.
        return
    state.set_mode(MODE_LISTENING)


def _on_speech_end(state, music_player):
    if _pipeline_lock.locked():
        return
    if music_player.is_playing():
        return
    state.return_to_resting()


def _on_utterance(state, music_player, text, trace=None):
    if not _pipeline_lock.acquire(blocking=False):
        print("[main] still handling the previous utterance, ignoring:", text)
        return
    state.set_mode(MODE_THINKING)
    threading.Thread(
        target=_run_pipeline, args=(state, music_player, text, trace), daemon=True
    ).start()


def _run_pipeline(state, music_player, text, trace=None):
    """text -> (optional scene lookup / music control) -> Ollama -> speech.

    Runs entirely off the sim thread so Whisper (already off-thread in
    speech/listen.py) and the LLM/TTS calls here never block the sim loop.
    Speech recognition and responses keep working normally regardless of
    whether music is playing - MODE_DANCING doesn't gate anything in
    speech/listen.py or here. Getting to MODE_DANCING itself doesn't wait
    on this pipeline either - state.sync_music_mode() in main()'s loop
    flips to it as soon as music_player.start() above makes is_playing()
    true, on the very next frame.

    `trace` (utils.metrics.UtteranceTrace, or None if config.MEASURE is
    False) is marked at the two stage boundaries that happen here -
    LLM response ready and speech start - then printed/recorded.
    """
    try:
        lowered = text.lower()

        if any(trigger in lowered for trigger in MUSIC_START_TRIGGERS):
            music_player.start()
        elif any(trigger in lowered for trigger in MUSIC_STOP_TRIGGERS):
            music_player.stop()

        if any(trigger in lowered for trigger in SCENE_TRIGGERS):
            scene.remember_current_view(state)

        reply = think.respond(state, text)
        if trace:
            trace.mark("llm_done")

        state.set_mode(MODE_SPEAKING)
        if trace:
            trace.mark("speech_start")
            trace.finish()
        speak(reply)
    finally:
        state.return_to_resting()
        _pipeline_lock.release()


def main():
    metrics.start()

    state = SharedState()

    # build_model() (not MjModel.from_xml_path) - it adds the lamp's
    # light to the model before compiling. See character/light.py.
    model = build_model()
    data = mujoco.MjData(model)
    lamp_light = LampLight(model)
    sound_cues = SoundCues()
    music_player = MusicPlayer()

    threading.Thread(
        target=tracking.watch, args=(state, config.CAMERA_ID), daemon=True
    ).start()

    threading.Thread(
        target=listen_loop,
        args=(state, lambda text, trace: _on_utterance(state, music_player, text, trace)),
        kwargs={
            "on_speech_start": lambda: _on_speech_start(state, music_player),
            "on_speech_end": lambda: _on_speech_end(state, music_player),
        },
        daemon=True,
    ).start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            music_playing = music_player.is_playing()
            state.sync_music_mode(music_playing)

            mode = state.get_mode()
            face_present, face_x = state.get_face()
            beat_phase = music_player.get_beat_phase()

            sound_cues.update(mode)

            # --- POSE STUB: character/behavior.py:choose_pose() ---
            # Mode -> actual pose/gesture, and all timing/expressiveness,
            # is intentionally left to you to fill in there. beat_phase
            # (0.0 when music isn't playing) is what the MODE_DANCING
            # stub has to work with - see character/music.py.
            target = choose_pose(mode, face_present, face_x, beat_phase, music_playing)

            # mood -> light color/intensity (see MOOD_COLORS above). intensity
            # is fixed at 1.0 for every mood here; vary it per-mood too if
            # you want some moods to glow brighter than others.
            color = MOOD_COLORS.get(state.get_mood(), MOOD_COLORS[MOOD_PLEASANT])
            lamp_light.set_target(color, intensity=1.0)
            lamp_light.update(model)

            move_toward_target(data, target)
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
