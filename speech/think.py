"""
Text -> local Ollama (llama3.2) -> reply text + mood. Folds in the
object-memory dict (see vision/scene.py) and prior conversation turns
(see character/memory.py) so the character doesn't reintroduce itself
or repeat questions every turn.

The model is asked to return JSON with a "text" field (what to say)
and a "mood" field (what color/intensity character/light.py's
LampLight should aim for - see main.py's sim loop). Only "text" is
ever spoken or recorded in conversation history; the raw JSON never
leaks out of this module.
"""

import json

import ollama

from character.state import DEFAULT_MOOD, MOODS

CHAT_MODEL = "llama3.2"
SYSTEM_PROMPT = (
    "You are a small desk lamp character with a camera and a voice. "
    'Respond with a JSON object with two fields: "text" - your spoken '
    "reply, at most one or two short sentences, like something you'd "
    'say out loud, never a paragraph; and "mood" - exactly one of: '
    f"{', '.join(MOODS)}, chosen to match the emotional tone of your "
    "reply. You remember the conversation so far, so don't reintroduce "
    "yourself or repeat earlier questions."
)

# Constrains ollama's output to this shape directly (most models still
# need the SYSTEM_PROMPT's plain-English version of the same rules to
# produce good *content*, but this keeps the JSON itself well-formed).
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "mood": {"type": "string", "enum": list(MOODS)},
    },
    "required": ["text", "mood"],
}


def _format_memory(memory_snapshot):
    if not memory_snapshot:
        return ""
    lines = "\n".join(
        f"- {name}: {description}" for name, description in memory_snapshot.items()
    )
    return f"Things you remember seeing:\n{lines}\n\n"


def _parse_reply(raw):
    """
    Parse the model's {"text": ..., "mood": ...} JSON reply. Falls back
    to treating the whole raw string as the spoken text (with
    DEFAULT_MOOD) if it's missing, malformed, or the mood isn't one of
    the known values - so a bad response degrades to "say something"
    rather than crashing the pipeline.
    """
    try:
        data = json.loads(raw)
        text = str(data["text"]).strip()
        if not text:
            raise ValueError("empty text field")
        mood = data.get("mood", DEFAULT_MOOD)
        if mood not in MOODS:
            mood = DEFAULT_MOOD
        return text, mood
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        print(f"[think] malformed reply ({exc}), falling back to raw text: {raw!r}")
        return raw.strip(), DEFAULT_MOOD


def respond(state, text):
    """
    Send `text` to the local LLM, using `state` for both object-memory
    context and conversation history, and return the reply string
    (text only - never the raw JSON). Also updates state's mood and
    conversation history.

    `state` is read and written through its thread-safe methods, so
    this is safe to call from a pipeline thread while other threads
    read/write `state` concurrently.
    """
    memory_context = _format_memory(state.get_memory_snapshot())
    history = state.get_history_messages()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": f"{memory_context}{text}"})

    try:
        response = ollama.chat(model=CHAT_MODEL, messages=messages, format=RESPONSE_SCHEMA)
        reply_text, mood = _parse_reply(response["message"]["content"].strip())
    except Exception as exc:
        print(f"[think] LLM call failed: {exc}")
        reply_text, mood = "Sorry, I'm having trouble thinking right now.", DEFAULT_MOOD

    state.set_mood(mood)

    # Record the raw exchange (not the memory-injected prompt, and never
    # the JSON) so history doesn't re-duplicate the object list or leak
    # the mood field on every replay.
    state.add_exchange(text, reply_text)

    return reply_text
