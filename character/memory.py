from collections import deque

# How many user/assistant exchanges to keep for prompt context. Capped so
# the prompt sent to the LLM doesn't grow unbounded over a long conversation.
MAX_HISTORY_EXCHANGES = 6


class CharacterMemory:
    def __init__(self, max_history_exchanges=MAX_HISTORY_EXCHANGES):
        self.objects = {}
        self._history = deque(maxlen=max_history_exchanges)

    # --- object memory ---

    def remember_object(self, name, description):
        self.objects[name] = description

    def get_object(self, name):
        return self.objects.get(name)

    # --- conversation history ---

    def add_exchange(self, user_text, assistant_text):
        """
        Record one user/assistant turn. Oldest exchanges are dropped once
        more than `max_history_exchanges` have been recorded.
        """
        self._history.append((user_text, assistant_text))

    def get_history_messages(self):
        """
        History as a flat list of Ollama-style {role, content} dicts,
        oldest first - ready to splice into a `messages` list ahead of
        the new turn.
        """
        messages = []
        for user_text, assistant_text in self._history:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})
        return messages
