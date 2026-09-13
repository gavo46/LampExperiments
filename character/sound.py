"""
Sound-effect cues driven by mode transitions: a "creak" one-shot when
the lamp first notices someone (idle -> engaged), and a soft "think"
loop for as long as it stays in MODE_THINKING.

Both clips are loaded once, here, at import time via pygame.mixer.
SoundCues exposes a single update(mode) method for the sim loop to
call once per frame; it tracks the previous mode internally and only
reacts on an actual transition, never on every frame.

--- Asset note ---
The files this module expects (assets/audio/creak.aiff,
assets/audio/think.aiff) didn't quite match what was described:
creak.aiff existed but at audio/assets/creak.aiff, and think.aiff
didn't exist at all. Moved creak.aiff to the path below, and copied
in /System/Library/Sounds/Purr.aiff as a think.aiff placeholder (a
soft, loopable ~0.76s clip) - swap it for whatever you actually want
the thinking loop to sound like. The pre-existing audio/ package
(audio/sound.py, audio/music.py) is unrelated - unimported stub
functions left over from earlier scaffolding, not wired to this.
"""

import time

import pygame

from character.state import MODE_ENGAGED, MODE_IDLE, MODE_SPEAKING, MODE_THINKING

CREAK_PATH = "assets/audio/creak.aiff"
THINK_PATH = "assets/audio/think.aiff"

# Kept low enough to sit under speech rather than over it.
CREAK_VOLUME = 0.4
THINK_VOLUME = 0.22

# A brief face-detection dropout (idle -> engaged -> idle -> engaged
# within this window) plays the creak once, not on every re-entry.
CREAK_DEBOUNCE_SECONDS = 1.0

if not pygame.mixer.get_init():
    pygame.mixer.init()

_creak_sound = pygame.mixer.Sound(CREAK_PATH)
_creak_sound.set_volume(CREAK_VOLUME)

_think_sound = pygame.mixer.Sound(THINK_PATH)
_think_sound.set_volume(THINK_VOLUME)

# Dedicated channels so these two cues (and any more added later) never
# steal playback from each other the way Sound.play()'s automatic
# "pick any free channel" can - in particular so a new creak can't
# stomp the think loop's channel or vice versa.
_creak_channel = pygame.mixer.Channel(0)
_think_channel = pygame.mixer.Channel(1)


class SoundCues:
    """
    Call update(mode) once per sim-loop frame with the current
    character.state mode.

      - creak: one-shot, only on an idle -> engaged transition,
        debounced (see CREAK_DEBOUNCE_SECONDS).
      - think: starts looping the instant mode becomes MODE_THINKING,
        stops the instant it becomes anything else.

    Both are force-stopped the instant mode becomes MODE_SPEAKING, so
    a cue's tail (or a loop that hasn't been told to stop yet) can
    never overlap `say`. This is a deliberate belt-and-suspenders
    choice, not a workaround for an observed bug - separate channels
    plus a real subprocess `say` call were tested together and neither
    cuts the other off or ducks it (see the conversation this was
    built from); it just guarantees speech is never competed with even
    in an edge case like a creak still tailing off when a fast
    engaged -> listening -> thinking -> speaking sequence happens.
    """

    def __init__(self):
        self._prev_mode = None
        self._last_creak_at = float("-inf")

    def update(self, mode):
        if mode == self._prev_mode:
            return  # nothing changed - cues fire on transitions only

        previous = self._prev_mode
        self._prev_mode = mode

        if mode == MODE_SPEAKING:
            _creak_channel.stop()
            _think_channel.stop()
            return

        if previous == MODE_IDLE and mode == MODE_ENGAGED:
            now = time.monotonic()
            if now - self._last_creak_at >= CREAK_DEBOUNCE_SECONDS:
                _creak_channel.play(_creak_sound)
                self._last_creak_at = now

        if previous == MODE_THINKING and mode != MODE_THINKING:
            _think_channel.stop()

        if mode == MODE_THINKING:
            _think_channel.play(_think_sound, loops=-1)
