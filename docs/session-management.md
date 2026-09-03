# Session management

This page describes the multi-session, resume, branch, and pruning features
for `diploid-agent`.

For the current implementation, see:

- `AgentRuntime` in `src/diploid_agent/runtime/agent_runtime.py`
- `AcpClient` in `src/diploid_agent/acp_client/client.py`
- `transport/telegram/` for the Telegram command surface.

## Background

The harness keeps one active ACP session per Telegram chat. The Devin ACP binary
persists session state in `~/.local/share/devin/cli/sessions.db`, so a new ACP
process can continue a previously saved session with `session/resume` or
`session/load`.

When the harness wants to continue an existing diploid session, it asks the
current `devin acp` subprocess to resume the stored `session_id`. If that succeeds,
the ACP message chain continues and the next user message is sent as a
rehydrated follow-up, so first-prompt-only plugins (e.g. `continuity`) still
appear at the session boundary. Only explicit `/new` or a chat that fails
consistency checks starts a fresh `session/new` with a full first prompt.

## Concepts

- **Active session** — the one currently serving a chat. Its files live in
  `sessions/<chat_id>/`.
- **Archived session** — a previous session kept for later resume or branch.
  Its files are copied to `sessions/<chat_id>/.archive/<n>/`.
- **Session number** — a per-chat, 1-based integer shown to the user (`/sessions`,
  `/resume 1`).
- **Session record** — a JSON line in `sessions.jsonl` with `chat_id`,
  `session_number`, `session_id`, `model`, `cwd`, `created_at`, `updated_at`,
  `turn_number`, `label`, and `parent`.

## Commands

### Telegram / HTTP

| Telegram | HTTP | Effect |
|---|---|---|
| `/new` | `POST /new/{chat_id}` | Archive the current active session and start a fresh one. The active workspace is cleared, but the chat-level ledger (`chat_transcript.jsonl`, `chat_MEMORY.md`, `hindsight-pending-retain.jsonl`) is preserved. |
| `/stop` | `POST /stop` | Cancel the in-flight ACP turn and return the partial reply. |
| `/stream_thoughts on\|off` | — | Toggle the optional real-time thought stream in Telegram. |
| `/sessions` | `GET /sessions/{chat_id}` | List numbered sessions; active one is marked. |
| `/resume 1` | `POST /resume` | Make session `1` the active session again. |
| `/branch 1` | `POST /branch` | Copy session `1` to the active slot and start a new ACP session from it. |

## Resume and rehydration

`/resume 1` works in two steps:

1. Attempt `AcpClient.resume_session` for the stored `session_id`. The client
   tries ACP `session/resume` first and falls back to the stable `session/load`
   if the ACP server does not advertise the unstable `session/resume` method.
   After a successful resume, the session `mode` and `model` are re-applied.
   The resumed follow-up is built with `rehydrated=True`, so the prompt includes
   first-prompt-only plugins and a rehydration notice.
2. If the ACP session cannot be resumed (or the feature is disabled), fall back
   to building a fresh first prompt from the archived transcript and memory,
   calling `AcpClient.create_session`, and updating the record with the new
   `session_id`. This prompt is also built with `rehydrated=True`.

This means a service restart or transport restart does not lose the conversation:
the ACP state is reloaded from `sessions.db` and the next message is sent as a
follow-up on the resumed session.

## Branching

`/branch 1` starts a new diploid session number. By default it first attempts to
resume the source ACP `session_id` so the new branch continues the same ACP
message chain. If the source session cannot be resumed (incompatible model,
MCP list, skills, or a previous timeout), it falls back to creating a new ACP
session seeded with the archived transcript and memory.

## ACP resume configuration

ACP session resume is controlled by `engine.acp_resume_enabled` (also writable as `diploid.acp_resume_enabled`; default `true`). When enabled, the harness tries ACP `session/resume` (falling back to `session/load`) before rehydrating from a rebuilt prompt.

Resume is only attempted when the active configuration matches the stored
session:

- The requested model must match the session's model.
- The active MCP server list must match `record.enabled_mcp_servers`.
- The active skill list must match `record.enabled_skills`.
- The previous turn must not have stopped with `timeout` (timed-out sessions are
  restarted from scratch).

If any of these checks fail, or if ACP resume raises an error, the harness falls
back to `session/new` with a full `build_first` prompt.

## Auto-pruning

The harness keeps the active session and any session whose `updated_at` is
within a configurable window (default 14 days). Older sessions are deleted from
`.archive/` and removed from `sessions.jsonl` during compaction. The active
session is never pruned.

## Data layout

```
sessions/
  <chat_id>/                 # active session workspace + durable chat ledger
    chat_transcript.jsonl    # durable turn log (survives /new, /resume, /branch)
    chat_MEMORY.md           # durable file-backend summaries
    hindsight-pending-retain.jsonl  # durable Hindsight spool
    ...
  <chat_id>/.archive/
    1/                       # archived session 1 workspace
      ...
    2/                       # archived session 2
      ...
sessions.jsonl               # append-only session registry
```

## Stop and soft timeout

Long-running ACP turns can be interrupted mid-flight:

- **Soft timeout** — `devin.soft_timeout` is passed to every `session/prompt`
  when it is set to a positive value. The default is `600.0`, which auto-cancels
  after 10 minutes. `0.0` or `null` disables auto-cancel. When a soft timeout
  fires, `AcpClient` sends a `session/cancel` *notification* and returns the
  streamed text collected so far with `partial=True` and `cancelled=True`.
  The reply includes a notice prompting the user to send `Continue` to resume.
- **Manual stop** — the user can send `/stop` in Telegram or `POST /stop` at
  any time. The harness looks up the active turn, calls `AcpClient.cancel()`,
  and the running `process()` returns a partial reply with a notice.

## Auto-recovery

During a normal turn, if `AcpClient.send_message` fails because the ACP session
is stale, the harness first attempts `AcpClient.resume_session` to reload the
stored `session_id` from `sessions.db`. If that succeeds, the turn continues as a
rehydrated follow-up, so first-prompt-only plugins still run and a rehydration
notice is injected. If resume fails or is disabled, the harness falls back to
building a first prompt from the current local state and creating a new ACP
session on the existing transport. The user does not need to manually type `/new`
after a service restart.

If the ACP transport itself is unresponsive (a `TimeoutError` or transport
failure), the harness calls `AcpClient.restart_transport()` to kill and restart
the `devin acp` process, then attempts to resume the last active session before
falling back to rehydration. Restart attempts are rate-limited with
`acp_max_restarts` / `acp_restart_backoff_window` (default 3 per 300 s) to
avoid tight kill/restart loops. The interrupted turn's `last_stop_reason` is
stored on the session record, and a continuation trigger (`Continue`, `Go on`,
`Proceed`, or `Resume`) inserts a prompt anchor telling the model to pick up
where it left off.

Unrecoverable ACP configuration errors (e.g. an unknown model or a rejected MCP
server definition) are classified as `AcpModelError` / `AcpMcpError` and return
a `ChatResult` with `last_stop_reason` set to `error` instead of crashing the
harness.

## ACP lifecycle log

The harness writes an append-only JSONL audit log next to the session store:

```
<harness-data>/
  acp-lifecycle.jsonl
  acp_restart_history.jsonl
```

`acp-lifecycle.jsonl` records transport start/stop/restart, session
new/resume/load attempts and outcomes, and rehydration events. `acp_restart_history.jsonl`
persists the in-process ACP restart backoff counter so the limiter survives
harness restarts.

## `/status` command

Both the Telegram bot (`/status`) and the HTTP endpoint `GET /status/{chat_id}`
return a `continuity` block that reports:

- `resume_enabled` — whether `engine.acp_resume_enabled` is on.
- `current_session_id` — the active ACP session id.
- `state` — how the session was established: `resumed`, `rebuilt`, `new` or `unknown`.
- `last_restart_at` / `last_restart_reason` — the most recent `transport.restart` event.
- `restart_count_in_window` — number of restarts inside `acp_restart_backoff_window`.

## Context pressure and `fresh` compact soul mode

When the next prompt is projected to exceed the context window, `ContextBuilder`
switches `soul_mode` to `fresh` and asks for a new ACP session on the following
turn. A `fresh` prompt:

- Uses the identity anchor instead of the full persona memory.
- Forces only the cheap `SOUL_SLOTS` (`self_narrative`, `self_state`, `body`, `wake`, `mesh`).
- Loads on-disk chat memory but skips long-term `recall_context`.
- Keeps the most recent `min_short_term_turns` raw and loads a pre-computed
  short-term compaction summary for the older turns.
- Injects a `Fresh ACP session for context pressure` notice.

The proactive trigger is controlled by `harness.proactive_new_session_threshold`
(default `0.85`) and `harness.proactive_input_buffer_factor` (default `1.2`).

## Smart short-term context

`memory.short_term_strategy` defaults to `smart` and `max_short_term_chars` to
`6144`. In `smart` mode the oldest `short_term_turns - min_short_term_turns`
pairs are summarized into a cached `.cache/short-term-summary-*.md` file. The
summary is pre-computed after each completed turn and after each `recall_context`
call so it is ready when a `fresh` reset happens.

## Deferred design notes

### Resend after hard timeout

When a prompt hits the hard deadline, the harness currently cancels the in-flight
request and returns a partial reply with a "Continue" notice.  Resending the
interrupted message on the user's behalf is intentionally deferred: it needs a
careful design that avoids double sends, respects user consent, and integrates
with the auto-continue / wake queue.  The open questions are:

- Should the resend preserve the exact prompt or rebuild it with a
  "you were interrupted" prefix?
- How does the resend interact with the `InstanceManager` queue and the
  `other_instance_running` check?
- Should it be gated by `acp_resume_enabled` and the `mcp_change` restart cause?

A future PR should add a dedicated `resend_interrupted_turn` action behind a
feature flag and pair it with an integration test that simulates a hard timeout
followed by a successful retry.
