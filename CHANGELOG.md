# Changelog

## 0.6.1 — 2026-09-10

### Summary

Documentation and continuity release. README and GitHub Pages are updated to
reflect the `bridge`/`SURFACE` handoff and the `body`/`felt` state layer, the
Hindsight API contract page is restored, and the built-in plugin list in the
docs is now complete.

### Added

- `docs/hindsight-api-contract.md` describing the Hindsight retain/recall/stats
  HTTP contract.
- README callouts for the BRIDGE/SURFACE handoff and the body/felt state layer.

### Changed

- `docs/state.md` now lists all built-in plugins from `diploid-plugins`
  (`body`, `bridge`, `continuity`, `curriculum`, `identity`,
  `persistent_memory`, `planner`, `self_management`, `self_state`,
  `working_memory`).
- `docs/index.md` one-sentence summary mentions bridge/SURFACE and body/felt.

## 0.6.0 — 2026-09-09

### Summary
Continuity and context-pressure release. The harness resumes stale ACP sessions
with `session/resume`/`session/load` (bounded budgets, jittered retries),
restores interrupted-turn intent into rehydrated prompts, and keeps turn
numbering monotonic across kills and restarts. Fresh-session prompts are sized
with live token calibration and tiered compact assembly, and graceful service
restarts now drain in-flight turns instead of cutting them off.

### Added
- ACP continuity waves 1–7: resume-by-default stale-session recovery,
  `acp_resume_timeout` / `acp_resume_after_restart_timeout` budgets, resume/load
  retries with jitter, per-prompt lifecycle telemetry in `acp-lifecycle.jsonl`,
  and `/status` resume counters.
- Tiered compact prompt assembly for fresh sessions (`prompt_blocks`
  allowlist/denylist), a real recent-turns tail, and a capped recall escape
  hatch in `fresh` compact mode.
- Live chars-per-token calibration from first-turn prompt metrics.
- Interrupted-turn anchoring: `current_intent` / `last_side_effect` from
  `chat_active_turn.json` / `chat_interrupted_turn.json` feed the protected
  continuation slot of rehydrated prompts, with a disk fallback when the
  in-process turn is gone.
- Monotonic turn numbering via `SessionRecord.pending_turn_number`.
- Global `DeliveryWorker` consuming the per-chat `ChatResult` outbox so wake,
  mesh, dispatch, and subagent completions reach Telegram.
- Mesh session chat mapping (`/mesh/chat-map`) and Telegram fallback routing.
- Full skill content injection when the user message matches a skill trigger.
- Promoted-pocket content-level dedup, auto-population, and
  `record_system_note`; trimmed chat-memory blocks point at the archive file.
- Hindsight retain bundles several turns per document and keeps only the
  post-tool final segment of each reply.
- Hard-timeout auto-resend (`acp_timeout_auto_resend`) with transcript marking.
- Kill-and-resume smoke test exercising the interrupted-turn path.
- Per-call ACP timeout plumbing for background calls (summaries, subagents,
  session resume).

### Changed
- Plugin `reload` deep-hot-swaps package-based plugins (all already-imported
  submodules, deepest-first) before dropping instances; a broken reload keeps
  the old plugin running.
- Plugin/body-state snapshots no longer roll back newer live files on restore.
- `/stop` cancels the live ACP session instead of a stale recorded one.
- Prompt updates are bounded (256 entries) and the prompt-callback queue is
  bounded (2048) with drop-oldest + telemetry instead of backpressure.
- Graceful restart resolves the correct persona unit instead of a hardcoded
  name, and external `systemctl restart` drains active turns first.
- The ACP transport restarts after a prompt hard timeout, and the watchdog no
  longer kills replacement transports mid-recovery.
- Telegram intermediate messages show only the uncommitted tail; streamed
  thoughts ship as multi-part messages before the final reply; TurnWorker wakes
  at the intermediate idle deadline so tool-call gaps split cleanly.

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
  set `last_stop_reason = "error"` instead of crashing the harness subprocess.

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
remove private persona content, sensitive persona plugin prototypes, and internal
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
- Sensitive persona plugin.
- Mesh/identity skeleton.
- Persona migration tooling.

### Removed from history
- Personal names, internal fleet names, private IP addresses, and old
  private persona identifiers.
