# Wake queue and proactive wake

The harness keeps a persistent `wake_queue.jsonl` of events that should
start a new turn for a chat. Events are created when:

- a background dispatch is registered (`POST /dispatch`);
- a timer or external worker enqueues one via `POST /wake`;
- a user explicitly requests a wake.

A separate `acp-waker` process polls the queue and calls `POST /wake`.
`/wake` checks the per-chat `InstanceManager` lock, so two instances cannot
process the same chat at once. A wake with `silent=True` runs the turn but
does not send an outbound message.

## HTTP endpoints

- `POST /wake` — body `{"chat_id": "...", "reason": "...", "event_id": "...", "silent": ...}`. Returns the turn result and consumes the queued event on success.
- `POST /dispatch` — creates a background dispatch and a pending wake event.
- `POST /continue` — completes a dispatch, marks the wake event ready, and
  immediately runs the continuation turn (consuming the wake event).

## Instance guard

`sessions/<chat_id>/instance.lock` contains the PID, instance id, and
heartbeat. A new instance will steal the lock only if the old PID is dead
or the TTL has expired, so slow-but-alive turns are not killed.

## Partial-turn crash recovery

`continuity.on_partial` writes `chat_active_turn.json` on every ACP chunk.
If the harness crashes and restarts, the next `on_waking` includes the
partial message and thought in the prompt.

## Running the waker

Copy `probes/acp-waker.py` to a convenient location (for example
`~/.local/bin/acp-waker`) and run it:

```bash
acp-waker --wake-queue wake_queue.jsonl --harness-url http://127.0.0.1:4003
```

A `systemd/acp-waker.service.example` unit is included for running it as
a user service.
