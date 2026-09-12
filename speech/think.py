"""
Text -> local Ollama (llama3.2) -> reply text. Folds in whatever the
object-memory dict holds so the character can refer back to things
it's seen (see vision/scene.py).
"""

import ollama

CHAT_MODEL = "llama3.2"
SYSTEM_PROMPT = (
    "You are a small desk lamp character with a camera and a voice. "
    "Keep replies short and conversational, like something you'd say "
    "out loud, not written text."
)


def _format_memory(memory_snapshot):
    if not memory_snapshot:
        return ""
    lines = "\n".join(
        f"- {name}: {description}" for name, description in memory_snapshot.items()
    )
    return f"Things you remember seeing:\n{lines}\n\n"


def respond(text, memory_snapshot=None):
    """
    Send `text` to the local LLM and return its reply as a string.

    `memory_snapshot` is a dict of remembered objects (see
    character/state.py's get_memory_snapshot()); it's included in the
    prompt so the model can reference what's been seen, even on
    utterances that aren't themselves about the camera.
    """
    memory_context = _format_memory(memory_snapshot or {})
    user_prompt = f"{memory_context}{text}"

    try:
        response = ollama.chat(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as exc:
        print(f"[think] LLM call failed: {exc}")
        return "Sorry, I'm having trouble thinking right now."

    return response["message"]["content"].strip()
