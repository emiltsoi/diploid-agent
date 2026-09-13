# diploid-agent

A persistent, persona-driven harness around an ACP-compatible agent engine.

It ships with the `devin acp` engine as the default, but the engine layer is
pluggable: any binary that speaks ACP v1 JSON-RPC over stdio can be configured
under `engine` instead. Every Telegram chat or HTTP caller gets a long-running
agent session, local transcript, per-chat model switching, and optional
retention to a Hindsight memory server.

## What it does

- Runs an ACP agent session with a persona loaded from `personas/<persona>`.
- Remembers each conversation in `sessions/<chat_id>/chat_transcript.jsonl`.
- Preserves context across **model switches**: `/model <name>` starts a new
  agent session and re-injects the recent transcript + long-term memory, while
  `/model --in-place <name>` changes the model on the live ACP session via
  `session/set_config_option`.
- Can keep long-term memory either locally (`file`) or in a Hindsight server
  (`hindsight`).
- Retains turns to Hindsight as bundled multi-turn documents containing only
  the post-tool final segment of each reply, so fact extraction sees cross-turn
  context instead of per-turn working narration
  (`harness.memory.retain_final_segment` / `retain_bundle_turns`).
- Supports session history: `/new`, `/sessions`, `/resume <n>`, `/branch <n>`.
- Exposes both a FastAPI HTTP ingress and a Telegram long-polling bot.
- Splits long or pausing Telegram replies into separate intermediate messages so
  tool-call gaps do not mash into one confusing block; each new message shows
  only the text that has not already been sent.
- Supports background dispatches that continue the conversation when they complete (`POST /dispatch`, `/continue`) and harness-native background subagents (`/subagent`, `harness_subagent` MCP tool) that survive the parent turn being stopped.
- Supports live runtime configuration of task, waker, timer, notifications, and Telegram settings via HTTP and Telegram without restarting.
- Supports state plugins with a rich lifecycle hook surface: plugins can intercept turns, sessions, dispatches, memory transitions, skill/MCP commands, retain/promote, and shutdown.
- Hardens the ACP transport with typed error classification, restart backoff, a
  16 MiB stdout line limit, prompt callbacks on a dedicated worker thread, a
  serialized lifecycle lock, bounded per-prompt update buffers, background
  isolation for secondary ACP calls, and stale-session recovery that attempts
  ACP `session/resume` (falling back to `session/load`) before prompt
  rehydration.
- Sandboxes the ACP subprocess so it cannot run raw `systemctl`, `reboot`, or `shutdown` against the host; restart requests from the agent are routed through the harness and scheduled gracefully with `systemd-run`.
- Queues incoming user messages as high-priority wake events when a chat is busy instead of dropping them, and pushes the final result through an outbox consumed by the Telegram `DeliveryWorker` so background turns, mesh wake replies, and subagent completions can still reach the user.
- Lets agents arm their own wakes: the `harness_self_wake`/`_list`/`_cancel` MCP tools write budget-capped `self_wake` events into the wake queue (`timer.self_wake_max_pending` / `self_wake_min_interval_seconds` / `self_wake_max_delay_seconds`), gated by the authorship plugin's `self_wake_enabled` toggle; `POST /timer`, `GET /timer/pending`, and `POST /timer/cancel` expose the same surface over HTTP.
- Runs declarative cron jobs from `config/crons.yaml` (operator-owned) and `<persona>/crons.yaml` (agent-authored, gated by the authorship `cron_enabled` toggle): `script` jobs run as shell subprocesses and `llm` jobs run as *phantoms* — fresh isolated ACP children composed with the persona's identity files + promoted pocket and an empty MCP list — materialized through the TaskEngine. `delivery: silent|digest` writes per-job result files that a bounded `cron` prompt slot renders; `GET /cron` exposes merged job state. See `docs/cron.md`.
- Sends a `System: service was restarted.` notice to recently active chats on startup and drops stale `auto_continue` wakes so a crash-restart does not immediately re-run an old continuation.
- Edits the streaming placeholder with a `(still working, Xm)` liveness suffix, and sends `⏳ Still thinking...` outbox heartbeats for long wake-driven turns, so users know whether to wait or send `/stop`.
- Curates a `/promote`-driven **promoted memory** pocket that survives `fresh`
  compact mode and is always re-injected at the top of the prompt.
- Wakes with a one-sentence continuity narrative built from the ACP lifecycle
  log, so the user and the model know whether the session was resumed, rebuilt,
  or restarted.
- Refreshes chat-scoped skill copies from shared/persona sources on follow-up and
  continue turns, so skill edits take effect on the next message without requiring
  a new ACP session.
- Snapshots and restores plugin and body-state files across ACP transport
  restarts, keeping per-chat state intact when the child process is replaced.
- Provides a first-person BRIDGE/SURFACE handoff (`bridge` plugin) so a wake after
  a clean close reads a concise first-person re-entry from `chat_surface.md`.
- Tracks per-chat body/felt state (`body` plugin) with event-keyed `felt_warmth`
  and agent-authored `felt_summary`; reads are pure so merely rendering the body
  does not age it.
- Drains active turns before an external `systemctl restart` exits, so a service
  restart waits for the current reply instead of cutting it off mid-sentence.
- Preserves active-turn `current_intent` and `last_side_effect` breadcrumbs in
  `chat_active_turn.json`, carrying them into `chat_interrupted_turn.json` if the
  process is killed before `record_turn` runs, and anchors them in the protected
  continuation slot of rehydrated prompts. Turn numbers are reserved up front so
  a killed turn's number is never reused.
- Sizes the next prompt with live chars-per-token calibration from the session's
  first-turn prompt metrics, instead of a fixed 4:1 guess; `fresh` compact mode
  uses tiered prompt assembly with a `prompt_blocks` allowlist/denylist and a
  capped recall escape hatch.
- Grants one bounded handoff turn before a context-pressure rebuild
  (`harness.pressure_handoff_enabled`, default on): the first trigger asks the
  live session to author its own resume state while it still holds full
  context; the next trigger proceeds with the fresh session.
- Pre-computes the smart short-term summary only when the recent-turn window is
  overflowing, avoiding unnecessary summarizer calls.
- Records resume / load / new latency and outcome telemetry in the lifecycle log
  and exposes it in `/status`.
- Runs a `diploid-memory` MCP server with `memory_recall`, `memory_retain`,
  `memory_promote`, and `memory_status` tools, plus a shared `memory` skill
  that lets the agent use them.
- Supports agent-to-agent mesh messaging via [`diploid-mesh`](https://github.com/emiltsoi/diploid-mesh), with `reply=yes/no/end` semantics, DSN recording, and per-turn nudges/caps to prevent mesh-send loops.
- Exposes a plugin framework for per-chat state plugins; the built-in state plugins
  live in [`diploid-plugins`](https://github.com/emiltsoi/diploid-plugins).
- Hot-reloads plugins without a service restart: `/plugin reload <name>`
  deep-reloads the configured module and every already-imported submodule
  (deepest-first) before dropping instances, so a broken edit keeps the old
  plugin running.

## Quick start

From PyPI:

```bash
pip install "diploid-agent[plugins]"   # harness + built-in state plugins
pip install diploid-mesh               # optional: agent-to-agent mesh
```

The example config and systemd unit live in this repository, so clone it for
those files — or for a development install:

```bash
git clone https://github.com/emiltsoi/diploid-agent.git
cd diploid-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp config/harness.yaml.example config/harness.yaml
# edit config/harness.yaml

cp systemd/diploid-agent.service.example systemd/diploid-agent.service
# edit paths, then:
systemctl --user enable --now "$(pwd)/systemd/diploid-agent.service"
```

Add `TELEGRAM_BOT_TOKEN=...` to `config/secrets.env` for Telegram.

## Authentication

The default engine spawns `devin acp`, which needs to be authenticated. The
easiest way is to sign in once on the same user account that will run the
service:

- Devin Desktop: sign in through the app.
- CLI: run `devin auth login` and complete the browser/manual token flow.

This writes credentials to `~/.local/share/devin/credentials.toml`. The
`systemd/diploid-agent.service.example` unit runs as your user and inherits your
`HOME`, so the credentials file is found automatically.

Other engines may use `WINDSURF_API_KEY`, `ACP_API_KEY`, or a per-engine
credential source. For a headless/dedicated account, set the relevant key in
`config/secrets.env` and reference that file from the service unit.

Send a message:

```bash
curl -X POST http://127.0.0.1:4003/chat \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "test-1", "message": "Introduce yourself"}'
```

Switch model:

```bash
curl -X POST http://127.0.0.1:4003/switch-model \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "test-1", "model": "glm-5-2"}'
```

Pass `"in_place": true` to change the model on the live ACP session instead of
starting a new one.

## Telegram commands

- `/status` — current model, session id, working directory, context-window usage, ACP continuity state, and resume telemetry.
- `/metrics` — token usage and latency for this chat.
- `/mcp list | /mcp enable <name> | /mcp disable <name>` — manage per-chat MCP servers.
- `/skill list | /skill enable <name> | /skill disable <name> | /skill create <name> <markdown>` — manage skills.
- `/plugin list | /plugin enable <name> | /plugin disable <name> | /plugin reload <name>` — manage state plugins; `reload` hot-swaps the plugin's code without a restart.
- `/state <plugin> <event> [args...]` — dispatch a state event to a plugin.
- `/models` — list available ACP models.
- `/model [--in-place] <name>` — switch this chat to a new model. `--in-place` changes the model on the live session instead of starting a new one.
- `/new` — start a fresh session.
- `/stop` — cancel the current turn and return a partial reply.
- `/restart` — kill the ACP subprocess and start a fresh transport.
- `/graceful-restart [service]` — schedule a graceful systemd restart of the named service (default: the current persona's `.service` unit).
- `/subagent <prompt>` — start a background ACP subagent and continue the chat with its result when it finishes.
- `/subagents` — list background subagents for this chat.
- `/continue` — resume the previous turn after a partial reply or timeout.
- `/sessions` — list numbered sessions.
- `/resume <n>` — resume session `n`.
- `/branch <n>` — branch from session `n`.
- `/memory` — show the per-chat memory.
- `/summarize` — manually trigger a file-backend summarization.
- `/recall <query>` — search the memory backend.
- `/promote <fact>` — append a fact to the chat's curated promoted memory (always loaded in `fresh` mode).
- `/stream_thoughts on|off` — toggle the optional real-time thought stream.
- `/config <section> <key>=<value> [key=value...]` — update live runtime config without restarting the harness.

The agent itself cannot reliably self-identify its serving model; `/status` is
the source of truth.

Replying to an earlier message in Telegram injects the quoted text into the next
prompt with a clear label. Long quotes are trimmed to
`harness.memory.max_reply_quote_chars` (default 2048 characters).

## Documentation

Browse the docs as a searchable site: **https://emiltsoi.github.io/diploid-agent/**

- [Architecture and data flow](docs/architecture.md)
- [Memory loop and Hindsight](docs/memory.md)
- [State plugins and lifecycle hooks](docs/state.md)
- [Model switching](docs/model-switching.md)
- [Session management](docs/session-management.md)
- [Telegram setup](docs/telegram.md)
- [HTTP API](docs/api.md)
- [systemd service](docs/systemd.md)
- [Security notes](docs/security.md)
- [Design decisions](docs/design-decisions.md)
- [Hindsight API contract](docs/hindsight-api-contract.md)
- [Background dispatches and continuation](docs/dispatch.md)
- [Wake queue and proactive wake](docs/wake.md)
- [Cron — declarative scheduled jobs](docs/cron.md)
- [Mesh integration](docs/mesh.md)
- [Index of all documentation](docs/index.md)
- [Plugin contract](docs/plugin-contract.md)

## Mesh support

`diploid-agent` can participate in the cross-harness mesh via the [`diploid-mesh`](https://github.com/emiltsoi/diploid-mesh) plugin (`pip install diploid-mesh`):

- Receives Ed25519-signed `[mesh]` webhooks on `/mesh/receive` (and the OpenClaw alias `/plugins/openclaw-mesh/webhook`).
- Wakes the diploid runtime with mesh context so the agent can reply.
- Exposes MCP tools (`mesh_send`, `mesh_list`, `mesh_register`, `mesh_sync`, `mesh_publish`, `mesh_health`, `mesh_deregister`).
- Enforces `reply=yes/no/end` semantics: `reply=no` nudges the model to avoid replying, `reply=end` hard-blocks `mesh_send`, and DSNs are recorded without a turn.
- Nudges and hard-caps `mesh_send` calls per ACP turn via `harness.mesh.max_sends_per_turn` and `harness.mesh.max_message_in_turn_suggestion`.
- Surfaces a bounded "Recent mesh" digest of terminal threads in the prompt — sender, summary, relative age — so closed conversations still shape context without forcing a reply.
- Strengthens prompt discipline with a top-of-prompt `SYSTEM — MESH REPLY RULE` CTA that commands the agent to use `mesh_send` for replies and to keep mesh content out of normal assistant text.
- Can mirror sent mesh messages back to Telegram as `System: [mesh] ...` notices via `harness.notifications.mesh_telegram_float`.
- Shares the same `mesh-peer-registry` and local vault format with [`hermes-mesh`](https://github.com/emiltsoi/hermes-mesh) and [`openclaw-mesh`](https://github.com/emiltsoi/openclaw-mesh), so a diploid agent can exchange messages with Hermes and OpenClaw agents using the same envelope and signatures.

See [`docs/mesh.md`](docs/mesh.md) and the [`diploid-mesh` README](https://github.com/emiltsoi/diploid-mesh/blob/main/README.md) for install, vault setup, and `harness.yaml` configuration.

## Important caveats

- Authentication is handled by the configured engine (`devin auth login` or
  Devin Desktop when `provider: diploid`). The harness only works if the user
  running it is already authenticated, or if `WINDSURF_API_KEY` / `ACP_API_KEY` is
  supplied in `config/secrets.env`.
- An ACP session's model is set at creation. `/model --in-place` updates it on
  the live session via `session/set_config_option`; a normal `/model` switch
  starts a new session and re-injects the conversation transcript + memory.
- The HTTP ingress is intended for a trusted/private network (`127.0.0.1` by
  default). If you expose it externally, set `HARNESS_API_KEY` in
  `config/secrets.env` and send it in the `X-API-Key` header on all `POST`/`PATCH`
  requests (including `PATCH /config` and the per-section `/task/config`,
  `/waker/config`, `/timer/config`, `/notifications/config`). Read-only `GET`s
  (including `GET /config`, which redacts secrets) and the inbound receiver
  `POST`s (`/webhook`, `/mesh/receive`, `/plugins/openclaw-mesh/webhook`,
  `/ingress/{protocol}`) remain open.
- `TELEGRAM_BOT_TOKEN` lives in `config/secrets.env` only; that file is
  gitignored and the poller does not log the token.

## Compliance note

This harness is an automation layer on top of a **single Devin/Cognition
account that you already pay for**. It does not share credentials, bypass
authentication, circumvent access controls, or expose paid features for free.
It is designed to be used by one operator with their own account and their own
CLI session.

Cognition's Acceptable Use Policy (June 2026, "Building with our Services —
Agentic Use") explicitly contemplates agents taking autonomous actions —
writing and executing code, interacting with third-party systems — under these
requirements, which this harness is built to satisfy:

- **Operator accountability** — you are responsible for every action taken by
  agents running under your account.
- **Human oversight** — the harness is a chat/HTTP interface to a session you
  can observe and interrupt; do not wire it to irreversible production actions
  without review and confirmation mechanisms.
- **No credential sharing** — one account, one operator, no multi-tenant access
  to your subscription.
- **No circumvention** — nothing in the harness overrides Devin's own security
  measures or access controls.
- **Third-party ToS respect** — agents driven through this harness must not
  interact with other systems in ways that violate *those* systems' terms
  (scraping, abuse, unauthorized access). Route agents only against systems
  you own or are authorized to use.

If you fork or redistribute this project, keep this section intact: the
compliance story is part of the design, not an afterthought. Do not market the
harness as "free Devin" or as a way to bypass paid tiers — it is a way to get
more value from a subscription you already hold.

## Source layout

The top-level packages were split in Phase 4/5 and Phase 6 so each major
responsibility lives in a focused module:

- `diploid_agent/runtime/agent_runtime.py` — the thin service container and
  turn orchestrator (the old `ConversationHarness`).
- `diploid_agent/runtime/*.py` — focused runtime collaborators:
  - `store.py` — chat/session persistence.
  - `state.py` — mutable scalar state shared by runtime components.
  - `metrics.py` — metrics, health, and prometheus formatting.
  - `config_manager.py` — live runtime configuration overrides.
  - `outbox.py` — outbox queue and notification delivery.
  - `mcp_skills.py` — MCP and skill enablement.
  - `plugins.py` — plugin lifecycle, incidents, and sandbox.
  - `plugin_runtime.py` — stable runtime surface exposed to state plugins.
  - `prompts.py` — first/follow-up prompt building and model resolution.
  - `subagent.py` — background subagent start/completion/status.
  - `planning.py` — plan and dispatch wake helpers.
  - `actions.py` — public command-style actions.
  - `ingress.py` — public turn-entry surface and transport ingress routing.
  - `instance.py` — per-chat singleton guard with cross-process locking.
  - `lifecycle.py` — startup, shutdown, restart notices, event-bus dispatch.
  - `restart.py` — ACP-subprocess restart scheduling and drain coordination.
  - `turn_controller.py` — re-export shim for `turn/controller.py`.
  - `wake_queue.py` — persistent, multi-process wake event queue.
  - `timer_service.py` — background wake consumer posting timer events.
  - `cron_service.py` — declarative cron scheduler materializing config-file
    jobs into the TaskEngine.
  - `cron_state.py` — JSONL-backed per-job cron state store.
  - `auto_continue.py` — auto-continue suppression state.
  - `event_bus.py` — in-memory runtime event bus.
  - `typing.py` — typing heartbeat for active tasks.
- `diploid_agent/turn/` — ACP per-turn engine:
  - `controller.py` — turn coordinator.
  - `base.py` — shared base for the `Turn*` collaborator classes.
  - `pipeline.py` — shared engine-invocation pipeline for process/dispatch.
  - `process.py` — main `process()` turn loop.
  - `stream.py` — per-turn `on_chunk`/`on_update` stream callbacks.
  - `session.py` — new/resume/branch session management.
  - `rehydrate.py` — stale session recovery and ACP resume.
  - `dispatch.py` — background dispatch and continue-turn.
  - `notifier.py` — streaming `_NotifyStream` and `_OutboxHeartbeat`.
  - `utils.py` — small shared helpers.
- `diploid_agent/context/` — prompt assembly:
  - `builder.py` — `ContextBuilder` first/follow-up prompt assembly.
  - `pressure.py` — context-pressure decisions and soul-mode selection.
  - `anchors.py` — continuation and interruption anchors.
  - `wake_context.py` — wake-context narration from the lifecycle log.
  - `token_estimator.py` — chars-per-token calibration and budget math.
  - `reply_quote.py` — reply-to quote formatting.
- `diploid_agent/acp_client/` — ACP JSON-RPC transport and process lifecycle:
  - `client.py` — public `AcpClient` session/prompt API.
  - `sessions.py` — `AcpSessionOps`: `session/new`/`resume`/`load` and
    `session/set_config_option` calls above the raw transport.
  - `transport.py` — low-level `AcpTransport` (subprocess, JSON-RPC reader).
  - `callbacks.py` — `AcpCallbackPump`: serializes prompt callbacks off the
    reader thread.
  - `watchdog.py` — `PromptWatchdog` stall detection and recovery.
  - `lifecycle.py` — `acp-lifecycle.jsonl` audit log.
  - `state.py` — shared mutable client state.
  - `control.py` — Unix-socket listener for agent restart requests.
  - `sandbox.py` — isolated `HOME` and fake `systemctl` wrappers.
  - `errors.py`, `types.py`, `utils.py` — shared helpers.
- `diploid_agent/transport/telegram/` — Telegram long-polling bot:
  - `poller.py` — `TelegramPoller` composing `TelegramCommandMixin`,
    `TelegramSenderMixin`, and `TelegramStateMixin`.
  - `commands.py`, `sender.py`, `state.py` — the three mixins.
  - `workers.py` — `TurnWorker` and `DeliveryWorker`.
- `diploid_agent/transport/http/` — FastAPI harness:
  - `app.py` — `create_app`, `HttpTransport`, `main`.
  - `routes/*.py` — domain-grouped route handlers.
  - `models.py` — request/response Pydantic models.
- `diploid_agent/memory*.py` — transcript and long-term memory:
  - `memory.py` — `MemoryManager` facade.
  - `memory_backends.py` — file and Hindsight backend implementations.
  - `memory_retention.py` — turn-retain buffer and Hindsight bundling.
  - `memory_short_term.py` — recent-turns window and summary cache.
  - `memory_promoted.py` — `/promote` pocket and persona memory.
  - `memory_models.py` — shared memory dataclasses.
  - `memory_mcp.py` — the `diploid-memory` MCP server.
- `diploid_agent/engine/` — engine adapters: `acp.py` (the `devin acp`
  implementation), `base.py`/`factory.py`/`router.py`, `fake.py` for tests.
- `diploid_agent/plan/` — background plan records and `PlanManager`.
- `diploid_agent/task/` — background task engine and worker pool.
- `diploid_agent/testing/` — `FakeRuntime` test double.
- `diploid_agent/plugins/` — state plugin lifecycle and manager.
- Top-level support modules: `config.py` (pydantic config schema),
  `models.py` (session/turn records), `dispatch.py` (dispatch records),
  `metrics.py`, `notifier.py`, `persona_composer.py`, `locking.py`,
  `text.py`, `mcp.py` + `mcp_stdio.py` (MCP server resolution and per-chat
  enablement), `skills.py` (skill discovery and chat-scoped loading),
  `plugin_sandbox.py` + `plugin_incidents.py`, `harness.py` (compatibility
  wrapper), and the `telegram_ingress.py` / `telegram_poll.py` legacy entry
  shims.

## License

[MIT](LICENSE)
