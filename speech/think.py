"""
Text -> local Ollama (llama3.2) -> reply text. Folds in the object-memory
dict (see vision/scene.py) and prior conversation turns (see
character/memory.py) so the character doesn't reintroduce itself or
repeat questions every turn.
"""

import ollama

CHAT_MODEL = "llama3.2"
SYSTEM_PROMPT = (
    "You are a small desk lamp character with a camera and a voice. "
    "Respond in at most one or two short sentences - a quick spoken "
    "reaction, never a paragraph. You remember the conversation so far, "
    "so don't reintroduce yourself or repeat earlier questions."
)


def _format_memory(memory_snapshot):
    if not memory_snapshot:
        return ""
    lines = "\n".join(
        f"- {name}: {description}" for name, description in memory_snapshot.items()
    )
    return f"Things you remember seeing:\n{lines}\n\n"


def respond(state, text):
    """
    Send `text` to the local LLM, using `state` for both object-memory
    context and conversation history, and return the reply string.

    The history in `state` is read and appended to through its
    thread-safe methods, so this is safe to call from a pipeline
    thread while other threads read/write `state` concurrently.
    """
    memory_context = _format_memory(state.get_memory_snapshot())
    history = state.get_history_messages()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": f"{memory_context}{text}"})

    try:
        response = ollama.chat(model=CHAT_MODEL, messages=messages)
        reply = response["message"]["content"].strip()
    except Exception as exc:
        print(f"[think] LLM call failed: {exc}")
        reply = "Sorry, I'm having trouble thinking right now."

    # Record the raw exchange (not the memory-injected prompt) so history
    # doesn't re-duplicate the object list on every replay.
    state.add_exchange(text, reply)

    return reply
