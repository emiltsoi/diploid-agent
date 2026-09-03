# HTTP API reference

Base URL: `http://127.0.0.1:4003` (configurable).

These HTTP endpoints mirror the methods on `RuntimeAPI` and are also used by the
Telegram poller's `CommandHandler` when the poller does not have a direct
runtime reference. Endpoints that return `ChatResponse` are implemented by
calling the matching `RuntimeAPI` method and serializing the `ChatResult`.

## `GET /health`

Health check.

```bash
curl http://127.0.0.1:4003/health
```

Response:

```json
{"status": "ok"}
```

## `POST /chat`

Send a message and get a reply.

```bash
curl -X POST http://127.0.0.1:4003/chat \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "test-1", "message": "Hello"}'
```

Response:

```json
{
  "reply": "Hello.",
  "notice": null
}
```

`notice` is set on the first turn of a session or when a memory file crosses
its context budget.

Optional `model` starts a new Devin session if it differs from the current one:

```bash
curl -X POST http://127.0.0.1:4003/chat \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "test-1", "message": "Hello", "model": "glm-5-2"}'
```

Optional `reply_to` and `reply_to_is_bot` inject a quoted earlier message into
the prompt with a clear label (e.g. when the user is replying to a previous
message in Telegram). Optional `reply_to_message_id` lets the harness resolve
the reference from the Telegram message registry instead of quoting the full
replied-to text:

```bash
curl -X POST http://127.0.0.1:4003/chat \
  -H "Content-Type: application/json" \
  -d '{
    "chat_id": "test-1",
    "message": "Can you explain this?",
    "reply_to": "The earlier message.",
    "reply_to_is_bot": true,
    "reply_to_message_id": 123
  }'
```

Long `reply_to` text is trimmed to `harness.memory.max_reply_quote_chars`
(default 2048 characters) to avoid bloating the prompt. When `reply_to_is_bot`
is true, the quote is capped at `harness.memory.max_bot_reply_quote_chars`
(default 240 characters) so the assistant does not see its own long output
echoed back.

## `POST /switch-model`

Switch the model for a chat.

```bash
curl -X POST http://127.0.0.1:4003/switch-model \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "test-1", "model": "glm-5-2"}'
```

Response:

```json
{
  "reply": "Now running on model `glm-5-2`.\n\nReady to continue.",
  "notice": null
}
```

## `POST /stop`

Cancel an in-flight ACP turn for a chat and return an immediate
acknowledgement. The active turn will return a partial reply when it finishes
aborting.

```bash
curl -X POST http://127.0.0.1:4003/stop \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "test-1"}'
```

Response:

```json
{
  "reply": "Stopping the current turn...",
  "notice": "The agent will return a partial summary when it aborts."
}
```

## `POST /restart`

Kill the ACP subprocess transport for a chat and start a fresh one. This does **not**
restart the systemd service; it only restarts the `devin acp` process.

```bash
curl -X POST http://127.0.0.1:4003/restart \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "test-1"}'
```

Response:

```json
{
  "reply": "ACP transport restarted.",
  "notice": "The next message will start a fresh Devin session."
}
```

## `POST /graceful-restart`

Schedule a graceful systemd restart of a service. The harness sends an
acknowledgement first, then uses `systemd-run` to restart the unit after a short
delay so the HTTP response can be delivered before the process goes down. If
`service` is omitted, it defaults to the current persona's `.service` unit
(self-service restart).

```bash
curl -X POST http://127.0.0.1:4003/graceful-restart \
  -H "Content-Type: application/json" \
  -d '{
    "chat_id": "test-1",
    "service": "diploid-agent.service",
    "reason": "user-requested"
  }'
```

Self-service restart (omit `service`):

```bash
curl -X POST http://127.0.0.1:4003/graceful-restart \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "test-1"}'
```

Response:

```json
{
  "reply": "Restarting diploid-agent.service. I'll be back in a moment.",
  "notice": "The service will restart in a few seconds."
}
```

## `GET /turn/{chat_id}`

Return the partial state of an active turn. Polling clients (like the Telegram
poller) use this to update a streaming placeholder message.

```bash
curl http://127.0.0.1:4003/turn/test-1
```

Response while a turn is running:

```json
{
  "chat_id": "test-1",
  "status": "running",
  "session_id": "...",
  "user_message": "Write a long story...",
  "message_text": "So far the story is about...",
  "thought_text": "",
  "stopped": false
}
```

Response when idle:

```json
{
  "chat_id": "test-1",
  "status": "idle"
}
```

## `GET /outbox`

Long-poll the outbound message outbox for *any* chat. Used by the global
`DeliveryWorker` in the Telegram poller to pick up the next `ChatResult` that
was enqueued by background turns, wake events, mesh messages, or subagent
completions, regardless of which chat it belongs to.

```bash
curl "http://127.0.0.1:4003/outbox?wait=5.0"
```

- `wait` (float, 0–60 seconds, default `0.0`) — how long to block before
  returning if the outbox is empty.

Response when a result is available:

```json
{
  "chat_id": "test-1",
  "result": {
    "reply": "The subagent finished.",
    "notice": null,
    "continuation": false,
    "dispatch_id": "dispatch-abc123",
    "session_id": null,
    "session_number": null,
    "turn_number": null,
    "metrics": null
  }
}
```

## `GET /outbox/{chat_id}`

Long-poll the outbound message outbox for a specific chat. Still supported for
per-chat `DeliveryWorker`s and backwards compatibility.

```bash
curl "http://127.0.0.1:4003/outbox/test-1?wait=5.0"
```

- `wait` (float, 0–60 seconds, default `0.0`) — how long to block before
  returning if the outbox is empty.

Response when the outbox is empty (or the wait expired):

```json
{
  "chat_id": "test-1",
  "result": null
}
```

## `POST /new/{chat_id}`

Archive the current active session and start a fresh Devin session with an
empty active directory.

```bash
curl -X POST http://127.0.0.1:4003/new/test-1
```

Response:

```json
{
  "reply": "New session started.\n\nReady to continue.",
  "notice": null
}
```

## Session management

These endpoints require the multi-session feature. See [Session management](session-management.md) for the full design.

### `GET /sessions/{chat_id}`

List the per-chat sessions. The active session is marked.

```bash
curl http://127.0.0.1:4003/sessions/test-1
```

Response:

```json
{
  "chat_id": "test-1",
  "active": 2,
  "sessions": [
    {
      "number": 1,
      "label": "2026-08-19 hello",
      "model": "swe-1-7",
      "turn_number": 3,
      "updated_at": 1724000000.0,
      "parent": null,
      "is_active": false
    },
    {
      "number": 2,
      "label": "2026-08-19 switched to glm-5-2",
      "model": "glm-5-2",
      "turn_number": 1,
      "updated_at": 1724000100.0,
      "parent": null,
      "is_active": true
    }
  ]
}
```

### `POST /resume`

Resume a previous session as the active one.

```bash
curl -X POST http://127.0.0.1:4003/resume \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "test-1", "session_number": 1}'
```

### `POST /branch`

Branch from a previous session and make the copy the active session.

```bash
curl -X POST http://127.0.0.1:4003/branch \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "test-1", "session_number": 1}'
```

## `GET /models`

List the model names accepted by the ACP engine.

```bash
curl http://127.0.0.1:4003/models
```

## `GET /metrics`

Return global aggregate metrics across all chats.

```bash
curl http://127.0.0.1:4003/metrics
```

Response:

```json
{
  "global": {
    "turns": 10,
    "input_tokens": 5000,
    "output_tokens": 3000,
    "total_tokens": 8000,
    "cached_tokens": 0,
    "latency_seconds": 45.0
  },
  "recent_turns": [...]
}
```

## `GET /metrics/{chat_id}`

Return per-chat cumulative and last-turn metrics.

```bash
curl http://127.0.0.1:4003/metrics/test-1
```

Response:

```json
{
  "cumulative": {
    "turns": 5,
    "input_tokens": 2500,
    "output_tokens": 1500,
    "total_tokens": 4000,
    "cached_tokens": 0,
    "latency_seconds": 22.5
  },
  "last_turn": {
    "turn_number": 5,
    "model": "swe-1-7",
    "input_tokens": 500,
    "output_tokens": 300,
    "total_tokens": 800,
    "cached_tokens": 0,
    "latency_seconds": 4.5
  }
}
```

## `GET /status/{chat_id}`

Show the chat's harness record.

```bash
curl http://127.0.0.1:4003/status/test-1
```

Response:

```json
{
  "chat_id": "test-1",
  "active": true,
  "persona": "test-pilot",
  "model": "swe-1-7",
  "session_number": 2,
  "session_id": "nickel-tango",
  "cwd": "/path/to/diploid-agent/sessions/test-1",
  "turn_number": 3,
  "persona_memory_exceeded": false,
  "chat_memory_exceeded": false,
  "memory": {...},
  "enabled_mcp_servers": ["github"],
  "enabled_skills": ["review"],
  "disabled_skills": null,
  "last_turn_metrics": {...},
  "cumulative_metrics": {...},
  "context_usage": {
    "model": "swe-1-7",
    "context_window": 262144,
    "last_turn": {...},
    "cumulative": {...},
    "memory_budgets": {...},
    "memory_exceeded": {...}
  },
  "continuity": {
    "resume_enabled": true,
    "current_session_id": "nickel-tango",
    "state": "resumed",
    "state_reason": null,
    "last_restart_at": "2026-01-15T10:23:45+00:00",
    "last_restart_reason": "transport timeout",
    "restart_count_in_window": 1
  },
  "active_turn": {
    "chat_id": "test-1",
    "status": "running",
    "session_id": "nickel-tango",
    "user_message": "hello",
    "message_text": "",
    "thought_text": "",
    "stopped": false,
    "start_time": 1788201234.5,
    "elapsed_seconds": 12.3
  }
}
```

When no turn is running, `active_turn.status` is `idle`. The `last_turn` block also includes `stop_reason` when the previous turn stopped with a non-empty reason (e.g. `timeout`).

The `continuity` block is populated from the per-harness `acp-lifecycle.jsonl` and `acp_restart_history.jsonl` files. It reports how the current ACP session was established (`resumed`, `rebuilt`, `new` or `unknown`), the most recent transport restart timestamp and reason, and the number of restarts within the configured backoff window.

## `GET /memory/{chat_id}`

Return the per-chat memory content.

```bash
curl http://127.0.0.1:4003/memory/test-1
```

## `POST /summarize/{chat_id}`

Manually trigger file-backend summarization.

```bash
curl -X POST http://127.0.0.1:4003/summarize/test-1
```

## `POST /recall`

Search the active memory backend.

```bash
curl -X POST http://127.0.0.1:4003/recall \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "test-1", "query": "what is my name", "tags": ["memory"]}'
```

## `POST /retain`

Retain an observation in the active memory backend.

```bash
curl -X POST http://127.0.0.1:4003/retain \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "test-1", "content": "User prefers tea in the morning", "tags": ["preference"], "context": "drink"}'
```

## `POST /promote`

Append a fact to the persona's global memory.

```bash
curl -X POST http://127.0.0.1:4003/promote \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "test-1", "message": "I prefer Rust over Python"}'
```

## Plugin sandbox

- `POST /plugin/sandbox` — validate a plugin module in an isolated subprocess.
  - Body: `{ "module": "my.module", "plugin": { "name": "my_plugin" } }`
  - Returns: `{ "ok": true }` or `{ "ok": false, "error": "..." }`

## Plugin incidents

- `GET /plugin-incidents` — return recent plugin failures and recovery actions.
- `GET /plugin-incidents/{plugin_name}` — incidents for one plugin.
- `POST /plugin-incidents` — record a plugin incident (used by the watchdog).
  - Body: `{ "plugin": "name", "phase": "start", "error": "...", "action": "..." }`

## Plugin lifecycle

Add, remove, update, toggle, and roll back plugins at runtime. All endpoints
require the `X-API-Key` header when `HARNESS_API_KEY` is configured and return a
`ChatResponse` shape.

The `module` field, when provided, must be a valid Python file module name
matching `^[A-Za-z_][A-Za-z0-9_.]*$`, must not contain `..`, and must expose a
`Plugin` class. Built-in and frozen modules are rejected.

### `POST /plugins`

Add a new plugin. Pass `dry_run: true` to validate the module without applying
the change.

Request body:

- `plugin` (object, required) — full plugin config; same fields as
  `harness.plugins` entries.
- `dry_run` (bool, default `false`) — if `true`, attempt to import
  `plugin.module` and return `Dry run OK for <name>`.

```bash
curl -X POST http://127.0.0.1:4003/plugins \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $HARNESS_API_KEY" \
  -d '{
    "plugin": {
      "name": "planner",
      "enabled": true,
      "module": "diploid_plugins.planner"
    },
    "dry_run": true
  }'
```

Response (dry run):

```json
{"reply": "Dry run OK for planner", "notice": null}
```

Response (applied):

```json
{"reply": "Plugin planner added", "notice": null}
```

### `DELETE /plugins/{name}`

Remove a plugin. Its instances are stopped and its MCP server is no longer
offered to new sessions.

```bash
curl -X DELETE http://127.0.0.1:4003/plugins/continuity \
  -H "X-API-Key: $HARNESS_API_KEY"
```

Response:

```json
{"reply": "Plugin continuity removed", "notice": null}
```

### `PATCH /plugins/{name}`

Update a plugin's config or swap its module. Pass `dry_run: true` to validate
the new module.

Request body:

- `name` (string, required) — must match the path parameter.
- `plugin` (object, required) — fields to overlay onto the existing config.
- `dry_run` (bool, default `false`) — if `true`, attempt to import the new
  module and return `Dry run OK for <name>`.

```bash
curl -X PATCH http://127.0.0.1:4003/plugins/continuity \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $HARNESS_API_KEY" \
  -d '{
    "name": "continuity",
    "plugin": {
      "module": "diploid_plugins.continuity"
    },
    "dry_run": true
  }'
```

Response (dry run):

```json
{"reply": "Dry run OK for continuity", "notice": null}
```

Response (applied):

```json
{"reply": "Plugin continuity updated", "notice": null}
```

### `POST /plugins/{name}/toggle`

Enable or disable a plugin globally, or for a specific chat when `chat_id` is
provided.

Request body:

- `chat_id` (string, optional) — per-chat toggle; omit for a global toggle.
- `enabled` (bool, required) — target state.

```bash
curl -X POST http://127.0.0.1:4003/plugins/continuity/toggle \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $HARNESS_API_KEY" \
  -d '{"chat_id": "test-1", "enabled": false}'
```

Response:

```json
{"reply": "Plugin continuity disabled", "notice": null}
```

### `POST /config/rollback`

Roll back the last `steps` plugin configuration changes.

Request body:

- `steps` (int, default `1`) — number of snapshots to roll back.

```bash
curl -X POST http://127.0.0.1:4003/config/rollback \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $HARNESS_API_KEY" \
  -d '{"steps": 1}'
```

Response:

```json
{"reply": "Rolled back 1 plugin configuration(s)", "notice": null}
```

## `POST /state`

Dispatch an event to a state plugin for a chat.

```bash
curl -X POST http://127.0.0.1:4003/state \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "test-1", "plugin": "curriculum", "event": "add_word", "params": {"word": "hola", "translation": "hello"}}'
```

The `params` object is passed as keyword arguments to the plugin's event handler.

## `GET /mcp/{chat_id}`

List configured MCP servers and whether each is enabled for the chat.

```bash
curl http://127.0.0.1:4003/mcp/test-1
```

## `POST /mcp`

Enable or disable an MCP server for the chat.

```bash
curl -X POST http://127.0.0.1:4003/mcp \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "test-1", "command": "enable", "name": "github"}'
```

Valid commands: `list`, `enable`, `disable`. `name` is required for `enable` and `disable`.

Configured plugin MCP servers are listed automatically per chat. Notable servers:

- `diploid-self-management` — list, sandbox, add, remove, toggle, and roll back plugins. Mutations require a per-chat approval token.
- `acp-harness-watchdog` — systemd user service that polls `/health`, rolls back on failure, and restarts the harness only if rollback does not restore health.

## `GET /skill/{chat_id}`

List available skills and whether each is enabled for the chat.

```bash
curl http://127.0.0.1:4003/skill/test-1
```

## `POST /skill`

Enable, disable, or create a chat-scoped skill.

```bash
# Enable a skill
curl -X POST http://127.0.0.1:4003/skill \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "test-1", "command": "enable", "name": "review"}'

# Create a chat-scoped skill
curl -X POST http://127.0.0.1:4003/skill \
  -H "Content-Type: application/json" \
  -d '{
    "chat_id": "test-1",
    "command": "create",
    "name": "oncall",
    "content": "---\nname: oncall\n---\n\nAcknowledge the page and check dashboards."
  }'
```

Valid commands: `list`, `enable`, `disable`, `create`. `name` is required for `enable`, `disable`, and `create`; `content` is required for `create`.

## `POST /dispatch`

Start a new dispatch for a chat. The external worker receives the returned `dispatch_id` and calls `POST /continue` when it finishes.

```bash
curl -X POST http://127.0.0.1:4003/dispatch \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "test-1", "context": "run tests"}'
```

Response:

```json
{
  "reply": "Dispatched.",
  "dispatch_id": "dispatch-abc123"
}
```

## `POST /continue`

Report a dispatch completion and trigger the next agent turn.

```bash
curl -X POST http://127.0.0.1:4003/continue \
  -H "Content-Type: application/json" \
  -d '{"dispatch_id": "dispatch-abc123", "result": "All tests passed."}'
```

Response:

```json
{
  "reply": "All tests passed.",
  "dispatch_id": "dispatch-abc123"
}
```

## `POST /wake`

Trigger or re-trigger a wake for a chat. This is primarily used by the wake
queue's `TimerService`; it can also be used to test a wake or force an
`auto_continue`.

```bash
curl -X POST http://127.0.0.1:4003/wake \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $HARNESS_API_KEY" \
  -d '{
    "chat_id": "test-1",
    "reason": "user_request",
    "event_id": null,
    "silent": false
  }'
```

## `POST /subagent`

Start a background ACP subagent for a chat. The subagent runs in a fresh
AcpEngine, so it survives the parent turn being stopped or killed. When it
finishes, the harness continues the chat and sends the result.

Optional fields:

- `model` — ACP model for the subagent; defaults to the chat's current model.
- `cwd` — working directory for the subagent; defaults to the chat's session
  directory.
- `acp_timeout` — hard prompt deadline in seconds for the subagent.
- `context` — a short label stored in the dispatch record.

```bash
curl -X POST http://127.0.0.1:4003/subagent \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $HARNESS_API_KEY" \
  -d '{
    "chat_id": "test-1",
    "prompt": "Research the latest Python release notes and summarize them",
    "context": "Python release research"
  }'
```

Response:

```json
{
  "reply": "Subagent started. I'll report back when it finishes.",
  "dispatch_id": "dispatch-abc123"
}
```

## `GET /subagents/{chat_id}`

List the background subagent dispatches for a chat, including their status,
 prompt snippet, duration, summary, and timestamps.

```bash
curl http://127.0.0.1:4003/subagents/test-1
```

Response:

```json
{
  "chat_id": "test-1",
  "subagents": [
    {
      "dispatch_id": "dispatch-abc123",
      "status": "completed",
      "type": "subagent",
      "started_at": 1724000000.0,
      "finished_at": 1724000300.0,
      "summary": "Summary of the subagent result",
      "prompt_snippet": "Research the latest Python release notes..."
    }
  ]
}
```

## `POST /webhook`

Telegram webhook. Expects a Telegram `Update` JSON payload and returns
`{"ok": True, "reply": "..."}`. If the update contains a `reply_to_message`,
its text is extracted and injected into the prompt as a quote.

## Runtime configuration

These endpoints let you inspect and mutate the live `task`, `waker`, `timer`, and `notifications` configuration without restarting the harness. `GET` and `POST` are available for each section. They require the `X-API-Key` header when `HARNESS_API_KEY` is configured. Partial updates are supported: only the fields present in the request body are changed. Invalid values return `422`. Successful updates are persisted to `runtime-overrides.yaml` in the project root; a persistence failure returns `503`.

The `acp_model` in `/task/config` is the default for ACP tasks. You can override it per ACP task with the `acp_model` field in `POST /plan/create` or in the planner's task JSON (`!plan`, `Plan:`, or `/plan` triggers); per-task values take precedence.

### `GET /task/config`

Return the current task configuration.

```bash
curl http://127.0.0.1:4003/task/config
```

Response:

```json
{"workers":4,"shell_timeout":60.0,"enabled_types":["shell","noop","acp"],"acp_timeout":null,"acp_model":null}
```

### `POST /task/config`

Update the task configuration. All fields are optional; the worker pool is resized at runtime.

```bash
curl -X POST http://127.0.0.1:4003/task/config \
  -H 'Content-Type: application/json' \
  -d '{"workers":6}'
```

Response has the same shape as the `GET` response and reflects the updated values.

### `GET /waker/config`

Return the current waker configuration.

```bash
curl http://127.0.0.1:4003/waker/config
```

Response:

```json
{"enabled":false,"interval_seconds":5.0,"max_retries":3,"retry_after":30.0,"lease_seconds":300.0}
```

### `POST /waker/config`

Update the waker configuration.

```bash
curl -X POST http://127.0.0.1:4003/waker/config \
  -H 'Content-Type: application/json' \
  -d '{"max_retries":4}'
```

### `GET /timer/config`

Return the current timer configuration.

```bash
curl http://127.0.0.1:4003/timer/config
```

Response:

```json
{"enabled":true,"interval_seconds":5.0,"lease_seconds":300.0,"max_retries":5,"retry_after_seconds":30.0}
```

### `POST /timer/config`

Update the timer configuration.

```bash
curl -X POST http://127.0.0.1:4003/timer/config \
  -H 'Content-Type: application/json' \
  -d '{"interval_seconds":10.0}'
```

### `GET /notifications/config`

Return the current notifications configuration.

```bash
curl http://127.0.0.1:4003/notifications/config
```

Response:

```json
{"enabled":true,"webhook_url":null}
```

### `POST /notifications/config`

Update the notifications configuration. `webhook_url` may be `null` or a non-empty string; the notifier is recreated at runtime.

```bash
curl -X POST http://127.0.0.1:4003/notifications/config \
  -H 'Content-Type: application/json' \
  -d '{"webhook_url":"https://example.com/notify"}'
```

