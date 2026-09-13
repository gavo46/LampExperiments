import queue
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import whisper

from character.state import MODE_SPEAKING

SAMPLE_RATE = 16000
CHUNK = 1600
THRESHOLD = 0.02
SILENCE_CHUNKS = 10
MIN_SPEECH_CHUNKS = 5

# Extra mute time (seconds) kept after MODE_SPEAKING ends, since `say`
# can finish playing just before the mode flips back and its last word
# or two would otherwise leak into the mic.
SPEAKING_COOLDOWN_SECONDS = 0.6

model = whisper.load_model("base")
audio_q = queue.Queue()


def _callback(indata, frames, time_info, status):
    audio_q.put(indata.copy())


def listen_loop(state, on_utterance, on_speech_start=None, on_speech_end=None,
                 cooldown_seconds=SPEAKING_COOLDOWN_SECONDS):
    """
    Blocks forever. Calls on_utterance(text) each time you finish
    speaking a recognizable phrase.

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
    """
    buffer = []
    silent_for = 0
    speaking = False

    chunk_seconds = CHUNK / SAMPLE_RATE
    cooldown_chunks = max(0, round(cooldown_seconds / chunk_seconds))
    cooldown_remaining = 0

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
            elif speaking:
                # still capture the trailing quiet so words don't get clipped
                buffer.append(chunk)
                silent_for += 1

                if silent_for >= SILENCE_CHUNKS:
                    text = None
                    if len(buffer) >= MIN_SPEECH_CHUNKS:
                        audio = np.concatenate(buffer)
                        wav.write("temp.wav", SAMPLE_RATE, audio)
                        result = model.transcribe("temp.wav")
                        text = result["text"].strip()

                    buffer = []
                    silent_for = 0
                    speaking = False

                    if text:
                        on_utterance(text)
                    elif on_speech_end:
                        on_speech_end()


if __name__ == "__main__":
    from character.state import SharedState
    listen_loop(SharedState(), lambda t: print("Heard:", t))
