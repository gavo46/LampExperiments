import threading

from character.memory import CharacterMemory

MODE_IDLE = "idle"
MODE_ENGAGED = "engaged"
MODE_LISTENING = "listening"
MODE_THINKING = "thinking"
MODE_SPEAKING = "speaking"
MODE_DANCING = "dancing"

MODES = (MODE_IDLE, MODE_ENGAGED, MODE_LISTENING, MODE_THINKING, MODE_SPEAKING, MODE_DANCING)

# Modes where nothing conversational is happening yet - just passively
# reacting to whether a face is in frame.
_RESTING_MODES = (MODE_IDLE, MODE_ENGAGED)

MOOD_PLEASANT = "pleasant"
MOOD_CONFUSED = "confused"
MOOD_PASSIONATE = "passionate"
MOOD_REASSURING = "reassuring"

MOODS = (MOOD_PLEASANT, MOOD_CONFUSED, MOOD_PASSIONATE, MOOD_REASSURING)
DEFAULT_MOOD = MOOD_PLEASANT


class SharedState:
    """
    Single thread-safe container for everything the vision thread, the
    speech thread, the LLM/TTS pipeline threads, and the sim loop need
    to read or write. Replaces the old module-level globals in
    main.py - every access goes through the lock so no thread ever
    sees a half-written update.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._face_present = False
        self._face_x = 0.5
        self._last_frame = None
        self._mode = MODE_IDLE
        self._mood = DEFAULT_MOOD
        self._memory = CharacterMemory()

    # --- vision: face presence/position ---

    def set_face(self, present, x=None):
        """
        Update face presence (and, if a face is present, its normalized
        horizontal position). Also drives the idle<->engaged transition:
        outside of a conversation (listening/thinking/speaking), mode
        follows face presence directly.
        """
        with self._lock:
            self._face_present = present
            if x is not None:
                self._face_x = x
            if self._mode in _RESTING_MODES:
                self._mode = MODE_ENGAGED if present else MODE_IDLE

    def get_face(self):
        """Returns (face_present, face_x)."""
        with self._lock:
            return self._face_present, self._face_x

    # --- vision: latest camera frame, for on-demand scene description ---

    def set_last_frame(self, frame):
        with self._lock:
            self._last_frame = None if frame is None else frame.copy()

    def get_last_frame(self):
        with self._lock:
            return self._last_frame

    # --- mode ---

    def get_mode(self):
        with self._lock:
            return self._mode

    def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode!r}")
        with self._lock:
            self._mode = mode

    def return_to_resting(self):
        """
        Drop back to idle/engaged (based on current face presence). Call
        this once a conversation turn (listening/thinking/speaking) has
        finished.
        """
        with self._lock:
            self._mode = MODE_ENGAGED if self._face_present else MODE_IDLE

    # --- mood ---

    def get_mood(self):
        with self._lock:
            return self._mood

    def set_mood(self, mood):
        if mood not in MOODS:
            raise ValueError(f"Unknown mood: {mood!r}")
        with self._lock:
            self._mood = mood

    # --- object memory ---

    def remember_object(self, name, description):
        with self._lock:
            self._memory.remember_object(name, description)

    def get_object(self, name):
        with self._lock:
            return self._memory.get_object(name)

    def get_memory_snapshot(self):
        """A plain-dict copy of everything remembered so far, safe to read
        or format outside the lock (e.g. to build an LLM prompt)."""
        with self._lock:
            return dict(self._memory.objects)

    # --- conversation history ---

    def add_exchange(self, user_text, assistant_text):
        with self._lock:
            self._memory.add_exchange(user_text, assistant_text)

    def get_history_messages(self):
        """A copy of the conversation history in Ollama's messages format,
        oldest first - safe to read outside the lock."""
        with self._lock:
            return self._memory.get_history_messages()
