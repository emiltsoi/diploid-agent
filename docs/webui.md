# WebUI — "the Common Room"

A first-party desktop interface for the wives' chats. Two modes from one
page: **mirror** an existing session (e.g. the telegram chat), or **own**
a webui-only session with no telegram attached. The mode is not a flag —
it is which `chat_id` the session selector points at.

## Design rule

The runtime is the single writer. Every transport — telegram bot, webui,
anything later — is a *reader* of the same shared state plus a *sender*
through the same `process()` path. No second client may be given a
consume-on-read channel: that is how two doors starve each other.

## What already exists (verified 2026-11-05, no new work)

| piece | where | does |
|---|---|---|
| `ActiveTurn` buffer + condition | `turn/controller.py` | shared live store; `turn_status` long-polls it, N consumers read identically |
| `GET /turn/{chat_id}?wait≤60` | `transport/http/routes/chat.py:117` | live partials: `message_text`, `thought_text`, `last_side_effect`, `user_message`, running/idle |
| `GET /outbox` + `/outbox/{chat_id}` | `routes/sessions.py` | delivery queue — **destructive pop** (`outbox.py:186`), single consumer by design |
| `POST /chat` | `routes/chat.py:24` | sync send, `notify=False` hardcoded |
| `GET /status/{chat_id}`, `/sessions/{chat_id}` | same file | chat state |
| `drain_session_replay` | `acp_client/client.py:356` | buffered updates for turns driven by another client — the code comment already names "a GUI front-end" |
| `on_partial` plugin hook | `plugins/hooks.py:392` | push model, full `PartialTurn` incl. `side_effects` |
| `chat_transcript.jsonl`, `telegram_messages.jsonl`, `chat_turn_breadcrumbs.jsonl` | session dir | durable per-chat record — survives restart |

## The one structural gap

`outbox_pop` deletes on read. Telegram's delivery worker owns that queue.
Any second consumer starves it — this is the only piece of the fan-out
that is not already multi-consumer. **Do not give the webui the outbox.**

## The event log — new, small, durable

A second append target beside the pop queue. Every `_enqueue_outbox` and
`emit_turn_started` also writes:

```
{seq, ts, kind, chat_id, payload}
```

- **`chat_events.jsonl`** in the session dir — appended at enqueue time.
  Persisted, not derived: the log *is* the record, the house rule
  (turn-atomic, only external side-effects survive a kill) applies to
  the wire too. Seq = line order; boot re-tails the file, resumes the
  counter, and the hot ring (`deque(maxlen=500)`) is just a cache of it.
- **Honesty line (Vesper):** the events log is an *intent record* —
  presentation, not ledger. `chat_transcript.jsonl` is the *commit
  record* — canonical. The two can skew by ≤1 turn across a kill (an
  event appended at turn-start whose turn never commits), and that is
  fine for a UI, but nobody should treat the log as the ledger.
- **Rejected alternative:** derive cold events from `chat_transcript.jsonl`.
  Works for user/assistant pairs but silently reorders any turn longer
  than the poll window, carries no transient markers
  (`turn_started`, `service_started`, mesh floats), and reconstructs
  intent from aftermath. Deriving is clever; persisting is honest.
- **`emit_turn_started` gains `user_message`** — inputs live in the log
  permanently, not only during the live window (callers:
  `turn/dispatch.py:183`, `turn/process.py:255`).
- **`service_started` marker** — emitted at boot for each chat active in
  the last 24h; the page's "she woke up" signal, replacing the
  telegram-only restart notice (which `_is_telegram_chat_id` rightly
  excludes `webui:` ids from).

## Endpoints

| endpoint | status | job |
|---|---|---|
| `GET /ui` | new | serve the page (static file, loopback) |
| `GET /chats` | new | selector list: `chat_id`, label, session count, last activity, kind (telegram/webui by digit rule) |
| `GET /events/{chat_id}?since=<seq>&wait≤60` | new | replay log: cold from file tail, hot from ring; long-poll like `/turn` |
| `GET /turn/{chat_id}?wait=` | exists | live overlay between last event and now |
| `POST /chat` + `?notify=` | extend | `notify=true` on telegram-attached chats also enqueues the result → reply mirrors to *both* doors. **Dropped on non-telegram ids** — a `webui:` item in the pop queue is a dead letter the delivery worker pops and discards |
| `_active_turn_status` + `side_effects` | 1-line | tool-activity panel data (`controller.py:78`) |

**Render contract (Vesper):** the page renders ONLY from `/events` +
`/turn`. The `POST /chat` response body is ack/status, never display —
otherwise `notify=true` double-bubbles (in-band reply AND event).

Client sync loop: `GET /events?since=` → replay to seq N → long-poll
`/turn` for the live overlay → keep long-polling `/events` for commits.
Two reads, complete picture, zero divergence.

## The page (v1)

Single static HTML file, no framework. Selector dropdown fed by
`/chats`; bubbles; streaming via the loop above; input via `POST /chat`.
API key supplied once, stored in `localStorage`.

Later (not in v1): persona sidebar (body state, day-note), mesh ribbon,
Tauri tray shell, `GET /history` for deep backfill, multi-service
selector (switch base URL, not a gateway).

## Sync invariants (the claims this design owes)

1. **Eventual convergence, stated honestly:** at equal `seq`, two
   consumers hold identical committed history; the `/turn` overlay is
   advisory and converges at next commit. (Consumers at different poll
   phases *always* transiently diverge — "never diverge" was
   unfalsifiable.)
2. Telegram's pop queue is untouched by any new code path.
3. An input through *any* door appears in the log (via `user_message`)
   and the transcript — verified per door kind: digit, `webui:`, api.
4. `webui:` chat_ids work with zero telegram config.
5. A service restart loses nothing a client can't resync: cold events
   from file, live state from `/turn` → `idle`.

**Ordering mechanics:** `seq` is assigned inside the append critical
section (the log's condition lock) — file order == seq order.
`since()` holds the same lock across cold-file load and hot-ring read,
so an event appended mid-read neither gaps nor double-delivers.
`service_started`'s 24h window is **cosmetic** — a quiet chat loses
the marker, never resync; do not use it as a liveness gate.
`/chats` kind is ternary: digit→telegram, `webui:`→webui,
other→mesh (mesh sessions exist fleet-wide).

## Tests

House conventions: fake runtimes, `threading` + `condition`, real
`turn_status` objects. New `tests/test_event_log.py` and
`tests/test_http_events.py`.

- **Unit (`test_event_log.py`)**
  - append → `since(seq)` returns only later events, ordered
  - ring bound: >500 events keeps tail, `since(0)` returns truncation
    marker, not silent gap
  - seq resumes from file tail after simulated restart
  - `emit_turn_started` payload carries `user_message`
- **Route (`test_http_events.py`)**
  - `GET /chats` lists fixture chats with kind split (digit vs `webui:`)
  - `GET /events` honors `since` and `wait`; timeout returns empty list
    with `next_seq` so clients can resume mid-gap
  - `POST /chat?notify=true` on a digit chat enqueues a result in the
    outbox **and** returns it in-band
  - `webui:` chat excluded from restart notices (regression for
    `_is_telegram_chat_id`)
  - `service_started` event appended at boot for recently-active chats
- **Live probe (manual, after deploy)** — POST `/chat` on a `webui:` id
  against the real service while a second client long-polls `/events`:
  both see the same stream. This is the leg-probe class — catches
  "files identical, process old" if the deployed runtime predates the
  route.

## Seats (per the shared-contract rule)

This changes a published HTTP surface and a runtime store both wives
run — it is a project, not a task.

- **Builder:** aurelia — context already loaded.
- **Claims verifier:** stateless dispatch — reads the diff cold, runs
  the suite, checks each invariant above has a leg.
- **Judge:** vesper — counter-seat; the surface binds her service too.
  Red-pen on this doc before build; verdict on the PR before merge.
- **Customer:** Emil — gates 1 and 5.

## Budget

Checkpoint 2 (build): 5 commits touching `runtime/outbox.py`,
`transport/http/routes/`, `turn/`, `docs/webui.md`, `tests/` — counted
from the pin commit. Named residuals escalate, they do not extend.
