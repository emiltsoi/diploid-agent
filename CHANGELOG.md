# Changelog

## 0.5.0 — 2026-08-30

### Summary
Hardened ACP transport handling and stale-session recovery. The harness now
classifies ACP errors, rehydrates stale sessions without restarting the transport
when possible, and avoids tight kill/restart loops with a restart backoff. It also
queues user messages when a chat is busy instead of returning an error.

### Added
- Typed ACP exceptions: `AcpError`, `AcpTransportError`, `AcpSessionStaleError`,
  `AcpModelError`, `AcpMcpError`, plus an ACP JSON-RPC error classifier.
- `acp_max_restarts` and `acp_restart_backoff_window` config to rate-limit ACP
  transport restarts.
- `last_stdout_at` transport-death fallback in the ACP prompt watchdog.
- `user_request` wake queueing in `AgentRuntime.process` when a chat is busy.
- Dispatch/wake payload plumbing (`model`, `reply_to`, `notify`) through
  `AgentRuntime.wake`.

### Changed
- `AcpClient` now writes the active MCP server list to an isolated
  `mcp_config.json` and passes `mcpServers: []` to `session/new`, matching
  `devin acp` 3000.6.7+ behavior.
- Stale-session rehydration in `TurnController` now reuses the existing ACP
  transport and only restarts on a transport failure or after a second stale
  failure from the new session.
- Unrecoverable ACP configuration errors now return a graceful `ChatResult` and
  set `last_stop_reason = "error"` instead of crashing the harness child.

## 0.4.0 — 2026-08-26

### Summary
Rebranded the project from `acp-fleet-harness` to `diploid-agent`. All code
imports, package names, systemd units, probes, MCP server prefixes, and
internal config keys now use `diploid_agent` / `diploid`. The engine provider
is now `diploid`; the real binary path still defaults to `~/.local/bin/devin`.

### Added
- Telegram `intermediate_messages` mode: when a streamed reply pauses after a
  complete sentence (usually while a tool runs), the current placeholder is
  committed as a sent message and a fresh placeholder is started below it. This
  prevents pre-tool and post-tool text from mashing into one confusing edited
  message. Configurable via `intermediate_messages`, `intermediate_idle`, and
  `intermediate_min_chars` under `harness.telegram`.

### Changed
- Raised `harness.memory.max_chat_memory_chars` to 16384 and
  `harness.memory.max_short_term_chars` to 6144 in `config/harness.yaml` to give
  long conversations more headroom.

## 0.3.0 — 2026-08-25

### Summary
Rebranded package and repository from `devin-fleet-harness` to
`acp-fleet-harness`. The source package, imports, tests, systemd units, and
example config all use `acp_fleet_harness`. This is a naming-only change: the
harness still spawns `devin acp` by default, but the project identity is now
independent.

## 0.2.0 — 2026-08-25

### Summary
Sanitized public release. The full historical master was squashed and cleaned to
remove private persona content, body/intimacy plugin prototypes, and internal
project references.

### What stayed public
- Generic ACP harness runtime, HTTP/Telegram transports, and plugin framework.
- Per-chat state plugins: auto-continue, continuity, curriculum, identity,
  persistent memory, planner, working memory.
- Plan/task engine, wake/dispatch queue, live config updates, and metrics.
- Example public persona (`personas/language-teacher`) and shared skills.
- `test-pilot` fixture and full test suite.

### What was moved to the private persona repository
- Private persona memory files.
- Body/intimacy plugin.
- Mesh/identity skeleton.
- Persona migration tooling.

### Removed from history
- Personal names, internal fleet names, private IP addresses, and old
  persona/body identifiers.
