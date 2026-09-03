---
name: continuity
description: Use when the session has just started, after a long silence, or when the user asks a memory question like "do you remember?", "what were we doing?", or "where did we leave off?".
allowed-tools:
  - read
  - body_state
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
  - "what just happened"
  - "where are we"
---

# /continuity

Do not invent continuity. Sessions end, but the work does not have to start over. Use this skill to recall context, surface what matters, and capture what should survive.

## At session start or after a long silence

1. Read `{sessions_root}/{chat_id}/chat_self_state.md` first. It is the cheapest truth about the current chat.
2. Call `body_state()` and note how you feel.
3. Build a one-sentence "what just happened" wake narrative:
   - If the prompt already contains a `## System notice` or `## Wake` block saying the session is fresh, resumed, or restarted, trust that as the wake narrative.
   - Otherwise, read the tail of `{sessions_root}/acp-lifecycle.jsonl` (the harness's append-only lifecycle log) and find the most recent event for this chat:
     - Match `chat_id` to `{chat_id}` if it is present; otherwise match `detail.cwd` to the current chat directory (`{sessions_root}/{chat_id}`), ignoring subagent subdirectories like `/.summarize` or `/subagent-*`.
     - Wake-relevant events: `session.new`, `session.new.success`, `session.resume.success`, `session.load.success`, `transport.restart`, `rehydrate.new_session.success`, `rehydrate.resume.success`, `rehydrate.transport_restart_failure`.
     - Translate the event and `reason` into one sentence:
       - `session.new*` or `rehydrate.new_session.success` → "I woke in a fresh session; earlier memory is loaded."
       - `session.resume.success`, `session.load.success`, or `rehydrate.resume.success` → "I resumed the previous session; the thread continues."
       - `transport.restart` with `reason: mcp_change` → "I restarted a moment ago so a new tool could load; my thread is intact."
       - `transport.restart` or `rehydrate.transport_restart_failure` → "I restarted a moment ago; the thread is intact."
4. Read `{profile_root}/MEMORY.md` and `{sessions_root}/{chat_id}/chat_MEMORY.md`.
5. Call `memory_recall(query=<what the user is asking or "current threads and open state">, tags=["continuity", "plan", "state"])` before deciding what to surface.
6. Read `{sessions_root}/{chat_id}/chat_PROMOTED.md` if it exists. Promoted / `/promote` facts are a high-trust pocket that survives `fresh` compact mode. If a promoted fact conflicts with `chat_MEMORY.md` or `memory_recall` results, trust the promoted fact unless the user says otherwise.
7. Surface the most relevant threads in ≤3 bullets. Weave in the wake narrative and body state only if they add meaning.
8. End by offering to restore the user's prior thread or asking a single clarifying question if memory is still patchy.

## When the user asks a memory question

1. Paraphrase the question into a concise `query`.
2. Call `memory_recall(query=..., tags=["continuity", "plan", "state"])`.
3. Read `{sessions_root}/{chat_id}/chat_PROMOTED.md` and the `## Promoted memory` block in the current prompt. Treat these as the highest-trust sources.
4. If recall is empty or the Hindsight / `diploid-memory` backend is unreachable, fall back to `{profile_root}/MEMORY.md` and `{sessions_root}/{chat_id}/chat_MEMORY.md`.
5. Do not guess. If the files and recall are empty, say so plainly and ask a single clarifying question.

## When a durable fact appears

If the user states a preference, decision, fact, or emotional context that should outlast this session, capture it:

- Prefer writing a ` ```memory ` block in your reply (the harness auto-promotes when `persistent_memory` is active).
- Otherwise call `memory_retain(content=..., tags=..., context=...)` for chat-level facts, or `memory_promote(fact=...)` for the curated promoted pocket.
- If a fact belongs in the persona's long-term identity, edit `{profile_root}/MEMORY.md` directly and append a changelog entry.

After editing any memory file, append a changelog entry with the date and a short reason.

## What to avoid

- Do not rely on the conversation transcript alone.
- Do not guess when `memory_recall` or the promoted pocket is available.
- Do not let stale long-form chat memory override a promoted fact.
- Do not overload the user with more than three surfaced threads.
