# Model switching

## Why a switch creates a new session

An ACP session's model is set at creation and can be changed via
`session/set_config_option`, but the harness prefers to start a fresh session
when the model changes. This keeps the transcript boundary clean and lets the
new model see the full persona + memory context in a single first prompt.

## What the harness does

`ConversationHarness.switch_model(chat_id, model)`:

1. Checks if the chat is already on the requested model.
2. Builds a first prompt with:
   - the full persona identity
   - the current chat memory (recalled from the active backend)
   - a system-style instruction to continue the conversation
3. Calls `session/new` through the long-lived ACP engine process and sets the
   `model` config option to `<new_model>`.
4. Updates `sessions.jsonl` with the new `session_id` and `model`.
5. Records the switch as a transcript turn so the new session has continuity.

The `session_id` in `/status` changes because it now points to the new agent
session. The old session still exists on the provider's servers but is no longer used by
the harness.

## Context preservation

A model switch does **not** mean the conversation is lost. The harness preserves
context in two ways:

1. **Short-term transcript** — the last `short_term_turns` are re-injected into
   the new session's prompt.
2. **Long-term memory** — file keyword recall or Hindsight semantic recall adds
   older relevant context.

This is why a follow-up like "What is my name?" still works after switching from
`swe-1-7` to `glm-5-2`.

## Commands

- Telegram: `/model <name>`
- HTTP: `POST /switch-model`

```bash
curl -X POST http://127.0.0.1:4003/switch-model \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "test-1", "model": "glm-5-2"}'
```

The harness prepends a clear note to the reply:

```text
Now running on model `glm-5-2`.

<agent reply>
```

The agent itself is not asked to state the model, because it cannot reliably
know.

## Model name normalization

The ACP server expects model IDs with dashes (e.g. `swe-1-7`, `glm-5-2`). The
harness normalizes dotted names like `swe-1.7` or `glm-5.2` by replacing `.` with
`-` before sending them to the ACP engine, so either form can be used when switching
models.
