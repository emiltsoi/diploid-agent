# Architecture and data flow

## Components

```
┌─────────────────┐     ┌────────────────────────────────────────────┐
│ Telegram bot    │────▶│ transport/telegram/                        │
│ (user messages) │     │  poller.py (long-polling loop)             │
└─────────────────┘     │  workers.py (TurnWorker, DeliveryWorker)   │
                      └──────────────────────┬───────────────────────┘
                                             │ HTTP
                                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ transport/http/ (FastAPI)                                                │
│  app.py — create_app / HttpTransport / main                              │
│  routes/*.py — /chat /turn /stop /switch-model /status /memory ...       │
│  /outbox /subagents /subagent                                            │
└──────────────────────┬───────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ runtime/agent_runtime.py (thin orchestrator)                            │
│  - sessions.jsonl (registry)                                             │
│  - sessions/<chat>/.archive/ (session history)                           │
│  - wake_queue.jsonl (pending wakes)                                      │
│  - outbox (per-chat ChatResult queue)                                    │
│  - AcpClient                                                             │
│  - MemoryManager                                                         │
│  - McpManager                                                            │
│  - SkillManager                                                          │
└──────────┬───────────────────────────────────────────────────────────────┘
           │                    │
           ▼                    ▼
┌───────────────────────┐    ┌──────────────────────────────────────────┐
│ runtime/*.py          │    │ turn/ (per-turn ACP engine)              │
│  store.py             │    │  controller.py — turn coordinator        │
│  metrics.py           │    │  process.py — main process() turn loop   │
│  config_manager.py    │    │  session.py — new/resume/branch          │
│  outbox.py            │    │  rehydrate.py — stale-session recovery   │
│  mcp_skills.py        │    │  dispatch.py — background continue-turn  │
│  plugins.py           │    │  notifier.py — stream/outbox heartbeat   │
│  prompts.py           │    └──────────────────────────────────────────┘
│  subagent.py          │
│  planning.py          │
│  actions.py           │
└──────────┬────────────┘
           │
           ▼
┌──────────────────────┐     ┌──────────────────────────────────────────┐
│ acp_client/          │     │ memory.py / memory_mcp.py                │
│  client.py           │     │ (file or Hindsight backend)              │
│  transport.py        │     └──────────────────────────────────────────┘
│  watchdog.py         │
└────────┬─────────────┘
         │
         ▼
┌─────────────────┐
│ ACP engine      │
│ (default devin) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Agent cloud     │
│ session         │
└─────────────────┘
```

### `AgentRuntime` (`runtime/agent_runtime.py`)

The thin core orchestrator.

- Owns `sessions.jsonl` through `runtime/store.py` (`ChatSessionStore`).
- Creates or resumes ACP engine sessions by delegating to `turn/`.
- Calls `MemoryManager` to retain turns and recall context.
- Delegates metrics, config, outbox, MCP/skills, plugins, prompts, subagents, plans, and public actions to focused `runtime/*.py` collaborators.

### `AcpClient` (`acp_client/`)

Synchronous wrapper around the configured ACP v1 JSON-RPC agent binary.

- `create_session(prompt, cwd, model)` calls `session/new`, sets `mode`/`model`
  config options, and streams the first prompt.
- `send_message(session_id, prompt, cwd, model)` calls `session/prompt` on an
  existing session.
- `list_models()` returns the model options advertised by the ACP server.
- Spawns one long-lived ACP agent subprocess and multiplexes all chats
  through it.
- `acp_client/transport.py` owns the JSON-RPC stdio transport, process
  lifecycle, and backoff/recovery (`AcpTransport`).
- `acp_client/watchdog.py` monitors in-flight calls and kills stuck children
  (`PromptWatchdog`).
- `acp_client/control.py` listens for graceful-restart requests from the ACP
  child over a private Unix socket.
- `acp_client/sandbox.py` sets up the isolated `HOME`, fake `systemctl`
  wrappers, and MCP configuration so the child cannot touch the host system
  directly.

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

### `turn/`

The ACP per-turn engine, extracted from the former `runtime/turn_controller.py`.

- `turn/controller.py` — thin turn coordinator; owns `TurnProcess`, `TurnSession`, `TurnRehydrate`, and `TurnDispatch`.
- `turn/process.py` — main `process()` ACP turn loop, including `_NotifyStream` streaming and placeholder editing.
- `turn/session.py` — `switch_model`, `new_session`, `resume_session`, `branch_session`, and session activation helpers.
- `turn/rehydrate.py` — stale session recovery, ACP `session/resume` fallback, and prompt rehydration.
- `turn/dispatch.py` — background dispatch `dispatch()` and `continue_turn()` when a subagent or background task completes.
- `turn/notifier.py` — `_NotifyStream` and `_OutboxHeartbeat` used by `process` and `continue_turn`.

### `transport/http/` and `transport/telegram/`

- `transport/http/app.py` creates the FastAPI service and keeps `HttpTransport`
  and `main`. The route handlers are grouped by domain in
  `transport/http/routes/*.py` (health, chat, sessions, models, skills,
  plugins, plans, runtime, state, config, mesh, webhook).
- `transport/telegram/poller.py` is the long-polling Telegram client. It
  composes `TelegramCommandMixin` (slash handlers), `TelegramSenderMixin`
  (Bot API calls and throttling), and `TelegramStateMixin` (per-chat
  persistence). It can connect either to a local `RuntimeAPI` or to the HTTP
  harness.
- `transport/command_handler.py` is the single dispatch layer used by the
  Telegram poller's slash-command handlers. It calls `RuntimeAPI` methods when
  the poller has a direct runtime, otherwise it calls the matching HTTP
  endpoint, so command implementation no longer duplicates the
  `if self.runtime is not None` / HTTP branching logic.
- Each active chat gets a `TurnWorker` thread that starts a turn, polls
  `GET /turn/{chat_id}`, and edits the reply placeholder in place.
- When `harness.notifications.outbox_delivery` is enabled, a `DeliveryWorker`
  per chat consumes `GET /outbox/{chat_id}` and delivers enqueued `ChatResult`s
  (queue acknowledgements, background subagent completions, wake continuations,
  and liveness heartbeats) to Telegram.
- If `intermediate_messages` is enabled, the worker commits the currently
  streamed text as a real message when it pauses after a complete sentence, then
  starts a fresh placeholder below it. This keeps tool-call gaps from mashing
  the pre-tool and post-tool text into one message.
- New messages while a turn is running are queued as `user_request` wakes when
  `outbox_delivery` is enabled, or cancel the active turn and start the next one
  (steering) otherwise.
- `/stream_thoughts on|off` toggles an optional second placeholder that edits
  with `agent_thought_chunk` text.
- `systemd/diploid-agent-run.sh` starts both as a pair under one systemd unit.

## Startup behavior

On startup the harness:

1. Loads the session registry and wake queue from disk.
2. Drops `auto_continue` wakes whose `created_at` is older than the current
   process start. Queued user messages and other system wakes are kept.
3. Sends a direct `System: service was restarted.` notice to each chat whose
   latest session was updated in the last 24 hours.
4. Starts the `TimerService`, which consumes due wakes and routes them through
   the appropriate `AgentRuntime.wake` handler.

## Session lifecycle

The harness can keep a history of sessions per chat. The active session is
stored in `sessions/<chat_id>/`; older sessions are archived in
`sessions/<chat_id>/.archive/<n>/`. See [Session management](session-management.md)
for details on `/new`, `/resume`, `/branch`, `/sessions`, and auto-recovery.

## Data flow for a normal message

1. Telegram/caller sends message for `chat_id`.
2. `AgentRuntime.process(chat_id, message, reply_to=..., reply_to_is_bot=...)`:
   - If a turn is already running for `chat_id`, the message is queued as a
     high-priority `user_request` wake and the user sees an immediate
     acknowledgement (`I'll get back to you in a moment.`). It runs when the
     current turn finishes.
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
3. Reply and any system notice are returned to the caller. When
   `harness.notifications.outbox_delivery` is enabled, the `ChatResult` is
   pushed to the per-chat outbox and the transport's `DeliveryWorker` sends it.
   For wake-driven and background turns, the harness also sends periodic
   liveness heartbeats (`⏳ Still thinking...`) if the model is only thinking
   and has produced no visible reply text.

## Data flow for `/stop`

1. Caller sends `POST /stop` with `chat_id`.
2. `AgentRuntime.stop(chat_id)`:
   - Acquires the lock, reads the active `session_id` from `ActiveTurn`.
   - Calls `AcpClient.cancel(session_id)`, which schedules a `session/cancel`
     notification on the ACP background loop.
   - Returns an acknowledgement immediately; the in-flight `/chat` call
     continues and returns a partial reply.

## Data flow for a model switch

1. User calls `POST /switch-model` or Telegram `/model <name>`.
2. `AgentRuntime.switch_model(chat_id, model)`:
   - Checks if already on that model.
   - Builds `_first_prompt` with persona + current memory + a system turn to introduce the switch.
   - Creates a new ACP session and sets the `model` config option to
     `<new_model>`.
   - Updates `sessions.jsonl` with the new `session_id` and `model`.
   - Records the switch as a turn in the transcript/memory.
3. The agent acknowledges the new model.

## Data flow for a background subagent

1. User calls `POST /subagent` or Telegram `/subagent <prompt>`, or the ACP
   child invokes the `harness_subagent` MCP tool.
2. `AgentRuntime.subagent_start(chat_id, prompt, ...)`:
   - Creates a `Dispatch` record in `dispatch_store.jsonl`.
   - Creates a one-task `Plan` with `TaskType.SUBAGENT`.
   - Enqueues a `dispatch` wake with the `dispatch_id`.
   - Starts the task via `TaskEngine.start_task`.
3. `TaskEngine` runs the `SUBAGENT` task on a fresh `AcpEngine` built by
   `build_engine`. Because the subagent has its own engine, it survives the
   parent turn being stopped or the transport being killed.
4. When the subagent finishes, `AgentRuntime._complete_subagent_task` stores
   the result in the dispatch and marks the wake ready.
5. The `TimerService` picks up the ready wake and calls
   `AgentRuntime.wake(chat_id, event_id)`.
6. `wake` sees a `dispatch` wake with a completed result and calls
   `continue_turn(dispatch_id, result)`. The harness builds a follow-up prompt
   with the subagent result as a continuation anchor and starts a real new turn
   for the chat, so Telegram receives the subagent output as a normal message.

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
