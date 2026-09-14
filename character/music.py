"""
Background music playback via pygame.mixer.music - a separate playback
stream from pygame.mixer.Sound/Channel, which is what character/sound.py's
SFX cues use. See the module docstring there for the channel layout;
see below for what was actually verified about the two not interfering.

Exposes start()/stop()/is_playing(), plus get_beat_phase(): wall-clock
time since playback started, turned into a musical position (integer
part = beat number, fractional part = position within the beat) so the
sim loop / choose_pose() can derive motion from the beat.

--- What was checked, rather than assumed ---
Started the music loop, then (a) triggered the existing creak/think SFX
cues on their own channels, and (b) ran a real `say` subprocess call,
all while polling pygame.mixer.music.get_busy() and the SFX channels'
get_busy(). All three played concurrently for their full durations with
no cutoffs and no volume changes on any of them - pygame.mixer.music
and pygame.mixer.Sound genuinely are independent playback paths, and
neither interferes with a separate `say` subprocess (same finding as
character/sound.py's docstring, now confirmed for music too).
"""

import threading
import time

import pygame

MUSIC_PATH = (
    "assets/music/"
    "freesound_community-disco-funk-loops-001-remix-1-long-loop-with-drums-120-bpm-6278.mp3"
)
BPM = 120

# Kept low enough that `say` stays intelligible over it.
MUSIC_VOLUME = 0.25

if not pygame.mixer.get_init():
    pygame.mixer.init()


class MusicPlayer:
    """
    Call start()/stop() from wherever a "play music"/"stop the music"
    trigger is handled (see main.py's _run_pipeline). get_beat_phase()
    is safe to call every sim-loop frame regardless of whether music
    is currently playing (returns 0.0 when it isn't).
    """

    def __init__(self, path=MUSIC_PATH, bpm=BPM, volume=MUSIC_VOLUME):
        self._path = path
        self._bpm = bpm
        self._volume = volume
        self._lock = threading.Lock()
        self._start_time = None

    def start(self):
        """Start looping playback. No-op if already playing, so a
        repeated "play music" doesn't reset the beat phase."""
        with self._lock:
            if pygame.mixer.music.get_busy():
                return
            pygame.mixer.music.load(self._path)
            pygame.mixer.music.set_volume(self._volume)
            pygame.mixer.music.play(loops=-1)
            self._start_time = time.monotonic()

    def stop(self):
        with self._lock:
            pygame.mixer.music.stop()
            self._start_time = None

    def is_playing(self):
        return bool(pygame.mixer.music.get_busy())

    def get_beat_phase(self):
        """
        Current position in beats since playback started: the integer
        part is the beat number, the fractional part is how far through
        that beat we are (0.0 = right on the beat, 0.5 = halfway to the
        next one). Returns 0.0 if music isn't currently playing.
        """
        with self._lock:
            start_time = self._start_time

        if start_time is None or not self.is_playing():
            return 0.0

        elapsed = time.monotonic() - start_time
        return elapsed * (self._bpm / 60.0)
