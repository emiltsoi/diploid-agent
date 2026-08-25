---
name: memory
description: Recall, retain, or promote facts across sessions.
allowed-tools:
  - memory_recall
  - memory_retain
  - memory_promote
triggers:
  - "remember that"
  - "what did we"
  - "promote to memory"
  - "recall"
  - "retain"
---

Use the memory tools to:
- `memory_recall(query, tags)` when the user asks about past turns or facts.
- `memory_retain(content, tags, context)` to save a useful observation for this chat.
- `memory_promote(fact)` to add a fact to your persona memory.

Prefer `memory_recall` before inventing answers. Keep tags short and specific, e.g. `state`, `plan`, `project`.
