---
name: continuity
description: Re-establish context at session start or when the user asks a memory question like "do you remember?", "what were we doing?", or "where did we leave off?".
allowed-tools:
  - read
  - memory_recall
  - memory_retain
  - memory_promote
triggers:
  - "continuity"
  - "/continuity"
  - "do you remember"
  - "what were we doing"
  - "where did we leave off"
  - "what did we"
  - "where were we"
---

# /continuity

Do not invent continuity. Sessions end, but the work does not have to start over. Use this skill to recall context, surface what matters, and capture what should survive.

## At session start or after a long silence

1. Read `{profile_root}/MEMORY.md`.
2. Read `{sessions_root}/{chat_id}/chat_MEMORY.md` if the chat id is known.
3. If the user has already asked something, call `memory_recall(query=<paraphrase>, tags=[...])` before answering.
4. Surface the most relevant threads in ≤3 bullets.

## When the user asks a memory question

1. Paraphrase the question into a `query` for `memory_recall`.
2. If the Hindsight bank or `diploid-memory` is unreachable, fall back to the markdown files above.
3. Answer from recalled facts. If nothing is found, say so plainly.

## When a durable fact appears

If the user states a preference, decision, fact, or emotional context that should outlast this session, capture it:

- Prefer writing a ` ```memory ` block in your reply (the harness auto-promotes when `persistent_memory` is active).
- Otherwise call `memory_retain(content=..., tags=..., context=...)` for chat-level facts, or `memory_promote(fact=...)` for persona-level facts.

After editing any memory file, append a changelog entry with the date and a short reason.

## What to avoid

- Do not rely on the conversation transcript alone.
- Do not guess when `memory_recall` is available.
- Do not overload the user with more than three surfaced threads.
