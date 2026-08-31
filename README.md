# diploid-agent

A persistent, persona-driven harness around an ACP-compatible agent engine.

It ships with the `devin acp` engine as the default, but the engine layer is
pluggable: any binary that speaks ACP v1 JSON-RPC over stdio can be configured
under `engine` instead. Every Telegram chat or HTTP caller gets a long-running
agent session, local transcript, per-chat model switching, and optional
retention to a Hindsight memory server.

## What it does

- Runs an ACP agent session with a persona loaded from `personas/<persona>`.
- Remembers each conversation in `sessions/<chat_id>/transcript.jsonl`.
- Preserves context across **model switches** by starting a new agent session and
  re-injecting the recent transcript + long-term memory.
- Can switch models on the fly (`/model <name>`).
- Can keep long-term memory either locally (`file`) or in a Hindsight server
  (`hindsight`).
- Supports session history: `/new`, `/sessions`, `/resume <n>`, `/branch <n>`.
- Exposes both a FastAPI HTTP ingress and a Telegram long-polling bot.
- Splits long or pausing Telegram replies into separate intermediate messages so
  tool-call gaps do not mash into one confusing block.
- Supports background dispatches that continue the conversation when they complete (`/dispatch`, `/continue`) and harness-native background subagents (`/subagent`, `harness_subagent` MCP tool) that survive the parent turn being stopped.
- Supports live runtime configuration of task, waker, timer, notifications, and Telegram settings via HTTP and Telegram without restarting.
- Supports state plugins with a rich lifecycle hook surface: plugins can intercept turns, sessions, dispatches, memory transitions, skill/MCP commands, retain/promote, and shutdown.
- Hardens the ACP transport with typed error classification, restart backoff, and stale-session recovery that attempts ACP `session/resume` (falling back to `session/load`) before prompt rehydration.
- Sandboxes the ACP child so it cannot run raw `systemctl`, `reboot`, or `shutdown` against the host; restart requests from the agent are routed through the harness and scheduled gracefully with `systemd-run`.
- Queues incoming user messages as high-priority wake events when a chat is busy instead of dropping them.
- Runs a `diploid-memory` MCP server with `memory_recall`, `memory_retain`, and
  `memory_promote` tools, plus a shared `memory` skill that lets the agent use them.
- Exposes a plugin framework for per-chat state plugins; the built-in state plugins
  live in [`diploid-plugins`](https://github.com/emiltsoi/diploid-plugins).

## Quick start

```bash
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

## Telegram commands

- `/status` — current model, session id, working directory, and context-window usage.
- `/metrics` — token usage and latency for this chat.
- `/models` — list available ACP models.
- `/model <name>` — switch this chat to a new model.
- `/new` — start a fresh session.
- `/stop` — cancel the current turn and return a partial reply.
- `/restart` — kill the ACP child and start a fresh transport.
- `/graceful-restart [service]` — schedule a graceful systemd restart of the named service (default: the current persona's `.service` unit).
- `/subagent <prompt>` — start a background ACP subagent and continue the chat with its result when it finishes.
- `/continue` — resume the previous turn after a partial reply or timeout.
- `/sessions` — list numbered sessions.
- `/resume <n>` — resume session `n`.
- `/branch <n>` — branch from session `n`.
- `/memory` — show the per-chat memory.
- `/summarize` — manually trigger a file-backend summarization.
- `/recall <query>` — search the memory backend.
- `/promote <fact>` — append a fact to the persona's global memory.
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
- [Index of all documentation](docs/index.md)
- [Plugin contract](docs/plugin-contract.md)

## Mesh support

`diploid-agent` can participate in the cross-harness mesh via the [`diploid-mesh`](https://github.com/emiltsoi/diploid-mesh) plugin:

- Receives Ed25519-signed `[mesh]` webhooks on `/mesh/receive` (and the OpenClaw alias `/plugins/openclaw-mesh/webhook`).
- Wakes the diploid runtime with mesh context so the agent can reply.
- Exposes MCP tools (`mesh_send`, `mesh_list`, `mesh_register`, `mesh_sync`, `mesh_publish`, `mesh_health`, `mesh_deregister`).
- Shares the same `mesh-peer-registry` and local vault format with [`hermes-mesh`](https://github.com/emiltsoi/hermes-mesh) and [`openclaw-mesh`](https://github.com/emiltsoi/openclaw-mesh), so a diploid agent can exchange messages with Hermes and OpenClaw agents using the same envelope and signatures.

See the [`diploid-mesh` README](https://github.com/emiltsoi/diploid-mesh/blob/main/README.md) for install, vault setup, and `harness.yaml` configuration.

## Important caveats

- Authentication is handled by the configured engine (`devin auth login` or
  Devin Desktop when `provider: diploid`). The harness only works if the user
  running it is already authenticated, or if `WINDSURF_API_KEY` / `ACP_API_KEY` is
  supplied in `config/secrets.env`.
- An ACP session's model is set at creation. Switching models starts a new
  session, but the harness re-injects the conversation transcript + memory.
- The HTTP ingress is intended for a trusted/private network (`127.0.0.1` by
  default). If you expose it externally, set `HARNESS_API_KEY` in
  `config/secrets.env` and send it in the `X-API-Key` header on `POST` and live runtime config `GET`
  requests (e.g. `/task/config`, `/waker/config`, `/timer/config`, `/notifications/config`). Other `GET` endpoints and Telegram's `/webhook` remain open.
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

## License

[MIT](LICENSE)
