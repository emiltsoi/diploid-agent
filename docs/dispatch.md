# Background dispatches and continuation

The harness can hand work off to an external worker and resume the
conversation when that work finishes. This is useful for long tasks that would
block the chat — running tests, fetching a remote resource, or waiting on a
CI pipeline.

## Concepts

- A **dispatch** is a pending record created by the harness for a chat. It
carries a `dispatch_id` and an optional `context` string.
- A **worker** is any external process or service that receives the
`dispatch_id`, does the work, and reports back to `POST /continue`.
- A **continuation** is the second agent turn the harness runs after the worker
reports a result. The result is injected into the prompt as a system notice.

## Sequence

```
┌──────────────┐      POST /dispatch        ┌──────────────────┐
│   Caller     │ ─────────────────────────▶ │                  │
│  (Telegram,  │                            │ Conversation     │
│   HTTP)      │◀──────── dispatch_id ──────│ Harness          │
└──────────────┘                            │                  │
                                            └────────┬─────────┘
                                                     │
                                                     │ external work
                                                     │
                                                     ▼
                                            ┌──────────────────┐
                                            │      Worker      │
                                            └────────┬─────────┘
                                                     │
                                                     │ POST /continue
                                                     │ with result
                                                     ▼
                                            ┌──────────────────┐
                                            │ Conversation     │
                                            │ Harness          │
                                            └────────┬─────────┘
                                                     │
                                                     │ deliver reply
                                                     │
                                                     ▼
                                            ┌──────────────────┐
                                            │   Notifier       │
                                            │ (Telegram/webhook│
                                            │     or noop)     │
                                            └──────────────────┘
```

## Data flow

1. The caller sends `POST /dispatch` with the chat id and an optional context
   string. The harness checks that the chat has an active session and creates a
   `Dispatch` record in `PENDING` state.
2. The worker is responsible for doing the work. It is *not* part of the
   harness; it can be a subagent, a CI job, or any script. The only contract is
   that it must call `POST /continue` with the `dispatch_id` and a `result`
   string when it finishes.
3. `POST /continue` looks up the dispatch, rejects the request if it is already
   completed, and starts a continuation turn.
4. The continuation turn builds a follow-up prompt with a system notice that
   includes the worker's `result`. It then calls the ACP server.
5. The ACP call is made without holding the harness lock, so other chats are
   not blocked while the continuation runs.
6. Like a normal turn, the continuation supports streaming and `POST /stop`.
   `GET /turn/{chat_id}` reports the partial message and thought text while the
   continuation is running.
7. When the ACP call finishes, the turn is recorded, the dispatch is marked
   `COMPLETED`, and the final reply is sent through the configured notifier.

## Notification delivery

The reply is delivered through a `Notifier` or through the per-chat outbox:

- `NoopNotifier` — does nothing. This is the default when
  `harness.notifications.enabled` is `false`.
- `TelegramNotifier` — calls `sendMessage` on the Telegram Bot API. Used when
  `harness.telegram.token` is set and no webhook is configured, unless outbox
  delivery is enabled.
- `WebhookNotifier` — posts the chat id and text as JSON to the configured
  `harness.notifications.webhook_url`.
- **Outbox delivery** — when `harness.notifications.outbox_delivery` is `true`,
  the runtime enqueues `ChatResult`s in a per-chat outbox and the Telegram
  `DeliveryWorker` consumes them via `GET /outbox/{chat_id}`. This is the default
  for the shipped personas and is required for background turns, subagent
  completions, and queued-user-message results to reach the user without a
  runtime-side `notifier`.

`harness.notifications.enabled` is the master switch. If it is off, every
notifier and the outbox become no-ops.

## Memory

The dispatch result is retained to the chat memory with the tag `dispatch`. The
continuation itself is recorded as a normal turn, so later turns can recall the
outcome.

If the ACP session is stale during the continuation, the harness rehydrates a
new session on the existing transport from the local transcript and memory. If
the transport itself is unresponsive, it restarts the ACP process and then
rehydrates, just as it does for a normal turn.

## Errors and guards

- `POST /dispatch` returns an error if the chat has no active session.
- `POST /continue` returns an error if the `dispatch_id` is unknown or the
  dispatch is already completed.
- A `PENDING` dispatch can be completed only once; duplicate `/continue` calls
  are rejected.

## See also

- [HTTP API](api.md) for endpoint details and request/response examples.
- [Architecture and data flow](architecture.md) for the full component diagram.
