# Architecture and data flow

## Components

```
┌─────────────────┐     ┌──────────────────────┐
│ Telegram bot    │────▶│ telegram_poll.py     │
│ (user messages) │     │ (long-polling loop)  │
└─────────────────┘     └──────────┬───────────┘
                                   │ HTTP
                                   ▼
┌─────────────────────────────────────────────────┐
│ telegram_ingress.py (FastAPI)                           │
│  /chat /turn /stop /switch-model /status /memory ...     │
└────────┬────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│ ConversationHarness                               │
│  - sessions.jsonl (registry)                      │
│  - sessions/<chat>/.archive/ (session history)    │
│  - AcpClient                                      │
│  - MemoryManager                                  │
│  - McpManager                                     │
│  - SkillManager                                   │
└────────┬──────────────────────┬──────────────────┘
         │                      │
         ▼                      ▼
┌─────────────────┐     ┌──────────────────────┐
│ ACP engine      │     │ Memory backend       │
│ (default devin) │     │ (file or Hindsight)  │
└────────┬────────┘     └──────────────────────┘
         │
         ▼
┌─────────────────┐
│ Agent cloud     │
│ session         │
└─────────────────┘
```

### `ConversationHarness`

The core state machine.

- Owns `sessions.jsonl`, the chat registry.
- Creates or resumes ACP engine sessions.
- Calls `MemoryManager` to retain turns and recall context.
- Handles model switching by starting a new engine session.

### `AcpClient`

Synchronous wrapper around the configured ACP v1 JSON-RPC agent binary.

- `create_session(prompt, cwd, model)` calls `session/new`, sets `mode`/`model`
  config options, and streams the first prompt.
- `send_message(session_id, prompt, cwd, model)` calls `session/prompt` on an
  existing session.
- `list_models()` returns the model options advertised by the ACP server.
- Spawns one long-lived ACP agent subprocess and multiplexes all chats
  through it.

### `MemoryManager`

Coordinates transcript, retention, summarization, and recall.

- Keeps `sessions/<chat_id>/chat_transcript.jsonl` (durable chat ledger).
- Picks the configured `MemoryBackend` (`file` or `hindsight`).
- Builds the memory block that is injected into new agent sessions.

### `MemoryBackend` implementations

- `FileMemoryBackend` — local JSONL transcript + `MEMORY.md` summaries, keyword recall.
- `HindsightMemoryBackend` — HTTP client for the Hindsight memory server with local spool and file fallback.

### `McpManager`

Resolves configured MCP servers into the ACP `session/new` payload.

- Stores MCP server definitions from `config/harness.yaml` under `harness.mcp.servers`.
- Computes per-chat enabled set from the active `SessionRecord` or `harness.mcp.default_enabled`.
- Provides `mcp_list`, `mcp_enable`, `mcp_disable` used by Telegram and HTTP commands.

### `SkillManager`

Discovers and syncs `SKILL.md` files into the chat working directory.

- Searches `personas/<persona>/skills/`, `personas/shared/skills/`, and `sessions/<chat_id>/.devin/skills/`.
- Loads YAML frontmatter (`name`, `description`, `allowed-tools`, `triggers`, `permissions`) and the prompt body.
- Copies enabled skills into `sessions/<chat_id>/.devin/skills/` before `session/new` so the ACP process discovers them.
- Supports chat-scoped skill creation via `/skill create <name> <markdown>`.

### `persona_composer`

Reads the canonical persona identity files from `personas/<persona>`:

- `SOUL.md`
- `AGENTS.md`
- `personas/shared/AGENTS.md` (or `<fleet_root>/shared/AGENTS.md`)

Other persona-adjacent layers (state map, working memory, continuity, identity,
and `MEMORY.md`) are owned by the runtime or by plugins. `MEMORY.md` is loaded
and capped by `MemoryManager` during prompt assembly. Optional reference docs
such as `references/*.md` are loaded by the plugin or skill that needs them, not
by the composer.

### `telegram_ingress.py` and `telegram_poll.py`

- `telegram_ingress.py` is the FastAPI service.
- `telegram_poll.py` is a long-polling Telegram client that calls the ingress
  endpoints and sends replies back. Each active chat gets a `TurnWorker` thread
  that starts a turn, polls `GET /turn/{chat_id}`, and edits the reply
  placeholder in place.
- If `intermediate_messages` is enabled, the worker commits the currently
  streamed text as a real message when it pauses after a complete sentence, then
  starts a fresh placeholder below it. This keeps tool-call gaps from mashing
  the pre-tool and post-tool text into one message.
- New messages while a turn is running cancel the active turn and start the next
  one (steering).
- `/stream_thoughts on|off` toggles an optional second placeholder that edits
  with `agent_thought_chunk` text.
- `systemd/diploid-agent-run.sh` starts both as a pair under one systemd unit.

## Session lifecycle

The harness can keep a history of sessions per chat. The active session is
stored in `sessions/<chat_id>/`; older sessions are archived in
`sessions/<chat_id>/.archive/<n>/`. See [Session management](session-management.md)
for details on `/new`, `/resume`, `/branch`, `/sessions`, and auto-recovery.

## Data flow for a normal message

1. Telegram/caller sends message for `chat_id`.
2. `ConversationHarness.process(chat_id, message, reply_to=..., reply_to_is_bot=...)`:
   - Loads the active record from `sessions.jsonl`.
   - Determines model.
   - If new, model changed, or the previous turn ended with a hard `timeout`:
     builds `_first_prompt` = persona + chat memory + continuation anchor (for
     `Continue` triggers) + message.
   - If existing: builds `_follow_up_prompt` = identity anchor + continuation
     anchor (for `Continue` triggers) + message.
   - If `reply_to` is present, the quoted text is inserted with a clear
     "In reply to ..." label and capped at `max_reply_quote_chars`.
   - Sends to the ACP engine and receives an `AcpPromptResult` with `reply`,
     `stop_reason`, `cancelled`, and `partial`. If the ACP session is stale, the
     harness reuses the existing transport, creates a new ACP session from the
     local transcript and memory, and continues the turn. If the transport itself
     is unresponsive or the new session also fails, the ACP process is restarted
     and the turn is retried once. Unrecoverable ACP errors (e.g. an unknown
     model or rejected MCP server config) return a graceful `ChatResult` instead
     of crashing the harness.
   - If `engine.soft_timeout` elapses before the ACP turn completes,
     `AcpClient` sends a `session/cancel` notification and the result is
     `partial=True`. The reply includes a notice prompting the user to send
     `Continue`. The same mechanism is used when the user sends `/stop`.
   - `MemoryManager.record_turn(message, reply, model, turn_number, session_number)` appends
     the (possibly partial) assistant reply to the transcript and includes `session_number`
     in the document ID and Hindsight tags.
   - Appends the updated record, including `last_stop_reason`, to `sessions.jsonl`.
3. Reply and any system notice are returned to the caller.

## Data flow for `/stop`

1. Caller sends `POST /stop` with `chat_id`.
2. `ConversationHarness.stop(chat_id)`:
   - Acquires the lock, reads the active `session_id` from `ActiveTurn`.
   - Calls `AcpClient.cancel(session_id)`, which schedules a `session/cancel`
     notification on the ACP background loop.
   - Returns an acknowledgement immediately; the in-flight `/chat` call
     continues and returns a partial reply.

## Data flow for a model switch

1. User calls `POST /switch-model` or Telegram `/model <name>`.
2. `ConversationHarness.switch_model(chat_id, model)`:
   - Checks if already on that model.
   - Builds `_first_prompt` with persona + current memory + a system turn to introduce the switch.
   - Creates a new ACP session and sets the `model` config option to
     `<new_model>`.
   - Updates `sessions.jsonl` with the new `session_id` and `model`.
   - Records the switch as a turn in the transcript/memory.
3. The agent acknowledges the new model.

## Files and directories

|| Path | Purpose | Tracked? |
|---|---|---|---|
|| `config/harness.yaml` | Main config | No (copy from `.example`) |
|| `config/secrets.env` | Telegram token / API keys | No (gitignored) |
|| `config/harness.yaml.example` | Generic config template | Yes |
|| `systemd/diploid-agent.service` | Local systemd unit | No (copy from `.example`) |
|| `systemd/diploid-agent.service.example` | Generic unit template | Yes |
|| `systemd/diploid-agent-run.sh` | Supervisor that runs ingress + poller | Yes |
|| `sessions/<chat_id>/` | Per-chat working directory | No (runtime) |
|| `sessions/<chat_id>/.devin/skills/` | Synced skills for the ACP process | No (runtime) |
|| `sessions/<chat_id>/chat_transcript.jsonl` | Durable turn log | No (runtime) |
|| `sessions/<chat_id>/chat_MEMORY.md` | Durable file-backend summaries | No (runtime) |
|| `sessions/<chat_id>/hindsight-pending-retain.jsonl` | Durable Hindsight spool | No (runtime) |
|| `sessions/<chat_id>/.archive/<n>/` | Archived session `n` | No (runtime) |
|| `sessions.jsonl` | Session registry | No (runtime) |
