import queue
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import whisper

from character.state import MODE_SPEAKING
from utils import metrics

SAMPLE_RATE = 16000
CHUNK = 1600
THRESHOLD = 0.02
SILENCE_CHUNKS = 10
MIN_SPEECH_CHUNKS = 5

# Cap on one continuous "speech" stretch before it's flushed and
# transcribed regardless of silence. Without this, sound that stays
# above THRESHOLD indefinitely - music playing through the room being
# the main case - would grow the buffer forever and never reach the
# silence check below, so commands like "stop the music" spoken over
# it would never get transcribed at all.
MAX_SPEECH_SECONDS = 4.0

# Extra mute time (seconds) kept after MODE_SPEAKING ends, since `say`
# can finish playing just before the mode flips back and its last word
# or two would otherwise leak into the mic.
SPEAKING_COOLDOWN_SECONDS = 0.6

model = whisper.load_model("base")
audio_q = queue.Queue()


def _callback(indata, frames, time_info, status):
    audio_q.put(indata.copy())


def _transcribe(buffer):
    """chunks -> (text, trace) via Whisper. text is "" if it heard nothing."""
    trace = metrics.start_utterance()  # marks "speech_end" now
    audio = np.concatenate(buffer)
    wav.write("temp.wav", SAMPLE_RATE, audio)
    result = model.transcribe("temp.wav")
    text = result["text"].strip()
    if trace:
        trace.mark("transcription_done")
    return text, trace


def listen_loop(state, on_utterance, on_speech_start=None, on_speech_end=None,
                 cooldown_seconds=SPEAKING_COOLDOWN_SECONDS):
    """
    Blocks forever. Calls on_utterance(text, trace) each time you
    finish speaking a recognizable phrase. `trace` is a
    utils.metrics.UtteranceTrace already marked with "speech_end" and
    "transcription_done" (or None if config.MEASURE is False) - pass
    it along so downstream stages can keep marking it.

    `state` (character.state.SharedState) is used to stop the lamp
    hearing itself: while state.get_mode() is MODE_SPEAKING - and for
    `cooldown_seconds` afterward - incoming audio is discarded and any
    in-progress buffer is dropped rather than fed to Whisper. The
    cooldown covers `say` still finishing playback just as the mode
    flips back to a listening-eligible one.

    on_speech_start() fires as soon as the mic first crosses the
    amplitude threshold - useful for driving a "listening" state.
    on_speech_end() fires when the mic goes quiet again *without* a
    usable utterance coming out of it (too short, or Whisper heard
    nothing); on_utterance already implies speech ended, so this only
    covers that "false alarm" case.

    Note: this module doesn't know about music playback at all, and
    deliberately keeps listening while it plays (see main.py's
    _on_speech_start) so "stop the music" still works. That means the
    music itself can cross THRESHOLD and trigger on_speech_start - the
    caller's job to not let that visually look like it stopped dancing.
    """
    buffer = []
    silent_for = 0
    speaking = False

    chunk_seconds = CHUNK / SAMPLE_RATE
    cooldown_chunks = max(0, round(cooldown_seconds / chunk_seconds))
    cooldown_remaining = 0
    max_speech_chunks = max(1, round(MAX_SPEECH_SECONDS / chunk_seconds))

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        blocksize=CHUNK, callback=_callback):
        while True:
            chunk = audio_q.get()

            if state.get_mode() == MODE_SPEAKING:
                # the lamp is talking right now - never listen to its own voice
                buffer = []
                silent_for = 0
                speaking = False
                cooldown_remaining = cooldown_chunks
                continue

            if cooldown_remaining > 0:
                # still inside the post-speech mute window
                cooldown_remaining -= 1
                buffer = []
                silent_for = 0
                speaking = False
                continue

            level = np.abs(chunk).max()

            if level > THRESHOLD:
                if not speaking and on_speech_start:
                    on_speech_start()
                speaking = True
                silent_for = 0
                buffer.append(chunk)

                if len(buffer) >= max_speech_chunks:
                    # Been "speaking" this long with no silence gap at
                    # all - flush and check now rather than waiting
                    # indefinitely for one. speaking stays True: as far
                    # as the caller's concerned this is still one
                    # ongoing utterance, and on_speech_end still only
                    # fires later on a real silence-without-text.
                    text, trace = _transcribe(buffer)
                    buffer = []
                    if text:
                        on_utterance(text, trace)
            elif speaking:
                # still capture the trailing quiet so words don't get clipped
                buffer.append(chunk)
                silent_for += 1

                if silent_for >= SILENCE_CHUNKS:
                    text = None
                    trace = None
                    if len(buffer) >= MIN_SPEECH_CHUNKS:
                        text, trace = _transcribe(buffer)

                    buffer = []
                    silent_for = 0
                    speaking = False

                    if text:
                        on_utterance(text, trace)
                    elif on_speech_end:
                        on_speech_end()


if __name__ == "__main__":
    from character.state import SharedState
    listen_loop(SharedState(), lambda t, trace: print("Heard:", t))
