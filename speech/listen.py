import queue
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import whisper
 
SAMPLE_RATE = 16000
CHUNK = 1600                 
THRESHOLD = 0.02             
SILENCE_CHUNKS = 10          
MIN_SPEECH_CHUNKS = 5        
 
model = whisper.load_model("base")
audio_q = queue.Queue()
 
 
def _callback(indata, frames, time_info, status):
    audio_q.put(indata.copy())
 
 
def listen_loop(on_utterance):
    """Blocks forever. Calls on_utterance(text) each time you finish speaking."""
    buffer = []
    silent_for = 0
    speaking = False
 
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        blocksize=CHUNK, callback=_callback):
        while True:
            chunk = audio_q.get()
            level = np.abs(chunk).max()
 
            if level > THRESHOLD:
                speaking = True
                silent_for = 0
                buffer.append(chunk)
            elif speaking:
                # still capture the trailing quiet so words don't get clipped
                buffer.append(chunk)
                silent_for += 1
 
                if silent_for >= SILENCE_CHUNKS:
                    if len(buffer) >= MIN_SPEECH_CHUNKS:
                        audio = np.concatenate(buffer)
                        wav.write("temp.wav", SAMPLE_RATE, audio)
                        result = model.transcribe("temp.wav")
                        text = result["text"].strip()
                        if text:
                            on_utterance(text)
 
                    buffer = []
                    silent_for = 0
                    speaking = False
 
 
if __name__ == "__main__":
    listen_loop(lambda t: print("Heard:", t))