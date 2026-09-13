# Model switching

## Why a switch creates a new session

An ACP session's model is set at creation and can be changed via
`session/set_config_option`, but the harness prefers to start a fresh session
when the model changes. This keeps the transcript boundary clean and lets the
new model see the full persona + memory context in a single first prompt.

An in-place switch is also available (see below) for cases where keeping the
live session's context is worth more than a clean boundary.

## What the harness does

`AgentRuntime.switch_model(chat_id, model)`:

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

## In-place switch

`/model --in-place <name>` (Telegram) or `POST /switch-model` with
`"in_place": true` changes the model on the **live** ACP session instead of
starting a new one:

```bash
curl -X POST http://127.0.0.1:4003/switch-model \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "test-1", "model": "glm-5-2", "in_place": true}'
```

Under the hood `AcpClient.set_session_model` issues a single
`session/set_config_option` (`configId: "model"`) on the existing session, so
the `session_id`, transcript, and in-context history all survive. The applied
model is written back to `SessionRecord.model` so follow-up turns keep it.

Differences from the default fresh-session switch:

- No persona + memory re-injection — the new model only sees what is already
  in the session's context plus whatever the next prompt adds.
- No plugin gate — `before_session_start` does not fire (no session starts), so
  a policy plugin cannot veto an in-place switch.
- All `/model` switches (in-place and fresh-session) are refused while a turn
  or another session op (`/new`, `/resume`, `/branch`, another `/model`) is in
  progress for the chat — the op marker (`runtime._session_ops`) stays held
  across the ACP call so nothing can slip into the unlocked window and
  overwrite the session record mid-switch.
- If the session is stale (e.g. after a transport restart) the call fails with
  the ACP error; retry with plain `/model` to switch via a fresh session.
- With no active session at all, `--in-place` falls back to the fresh-session
  path.

## Model name normalization

The ACP server expects model IDs with dashes (e.g. `swe-1-7`, `glm-5-2`). The
harness normalizes dotted names like `swe-1.7` or `glm-5.2` by replacing `.` with
`-` before sending them to the ACP engine, so either form can be used when switching
models.
