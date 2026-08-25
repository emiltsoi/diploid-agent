# Session management

This page describes the multi-session, resume, branch, and pruning features
for `acp-fleet-harness`.

For the current implementation, see:

- `ConversationHarness` in `src/acp_fleet_harness/harness.py`
- `AcpClient` in `src/acp_fleet_harness/acp_client.py`
- `telegram_poll.py` for the Telegram command surface.

## Background

The harness keeps one active Devin ACP session per Telegram chat. ACP session
IDs are only valid for as long as the `devin acp` subprocess that created them
is alive. If the service restarts or the subprocess crashes, the stored ACP
`session_id` becomes stale. The harness therefore treats the ACP session as
ephemeral and the **local on-disk state** (transcript, memory, session record)
as the authoritative conversation history.

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

1. Try to reuse the stored ACP `session_id` by sending a low-cost
   `session/set_config_option` probe.
2. If the session is stale, build a fresh first prompt from the archived
   transcript and memory, call `AcpClient.create_session`, and update the record
   with the new `session_id`.

This means a service restart does not lose the conversation; the next message
automatically rehydrates the ACP session from local context.

## Branching

`/branch 1` always creates a **new** ACP session seeded with the context of
session `1`. It does not reuse the original ACP session, so the new branch can
be edited without affecting the source session.

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
is stale, the harness catches the error, builds a first prompt from the current
local state, creates a new ACP session, and continues the turn. The user does
not need to manually type `/new` after a service restart.

If a hard `devin.timeout` elapses and the ACP process becomes unresponsive, the
next turn detects the `TimeoutError`, calls `AcpClient.restart_transport()` to
kill and restart the `devin acp` process, and then rehydrates a fresh session
from the durable transcript. The interrupted turn's `last_stop_reason` is
stored on the session record, and a continuation trigger (`Continue`, `Go on`,
`Proceed`, or `Resume`) inserts a prompt anchor telling the model to pick up
where it left off.
