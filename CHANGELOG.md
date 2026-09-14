# Changelog

## 0.6.7 — 2026-09-15

### Added

- Telegram outbound files: a fenced ` ```file ` block in a reply uploads a
  file from the chat workspace — first line is the path (workspace-relative
  or absolute under it), remaining lines become the caption. The method is
  chosen by extension (`sendPhoto`/`sendAnimation`/`sendVideo`/`sendDocument`).
  Paths must resolve inside `<sessions_root>/<chat_id>/` and stay under
  `attachments_max_bytes`; missing, escaping, oversized, or failed uploads
  fall back to a `[file] <path>` text line with the caption, and the block is
  always stripped from the message. Documented in `personas/shared/AGENTS.md`.
- Telegram TTS: a fenced ` ```say ` block in a reply is synthesized and sent
  as a voice note (`sendVoice` for ogg/opus, `sendAudio` otherwise).
  `harness.telegram.tts_provider` selects `none` (default), `piper`
  (`piper-tts` + a voice `.onnx` at `tts_model_path`, wav → `ffmpeg` → ogg,
  model cached), or `command` (text on stdin → audio on stdout — the
  host-bridge escape hatch). Blocks longer than `tts_max_chars` (800) and
  failures fall back to a `[say] ...` text line; the block is always stripped
  from the text message, and a say-only reply clears the streaming
  placeholder. The contract is documented for personas in
  `personas/shared/AGENTS.md`.
- Telegram STT: `voice`, `audio`, and `video_note` attachments can be
  transcribed at ingest — `harness.telegram.stt_provider` selects
  `none` (default), `faster-whisper` (optional package, one cached
  `WhisperModel` per `stt_model` size, CPU int8), or `command` (runs
  `stt_command <file>`, stdout is the transcript, 60s timeout). The transcript
  rides the message annotation as `[transcript: "..."]`; failures annotate
  `[transcript unavailable]` and keep the file. Also fixes the standalone
  poller entrypoint never threading the `attachments_*` fields through
  `TelegramPoller`'s kwargs — they were unreachable from YAML.

## 0.6.6 — 2026-09-14

### Added

- Telegram attachments: messages carrying a photo, document, voice, video,
  video note, sticker, or animation are now downloaded on the turn worker via
  `getFile` + the file endpoint into the chat's ACP workspace at
  `<sessions_root>/<chat_id>/inbox/<message_id>-<name>`, and the prompt is
  annotated `[attachment saved: inbox/<name> (kind[, mime])]` — a captionless
  attachment's annotation is the whole message text. Filenames are sanitized
  to one path segment; `harness.telegram.attachments_max_bytes` (default
  20 MB, the Bot API ceiling) is enforced on the declared size and the
  streamed body, with partial downloads removed; failures annotate
  `[attachment could not be saved: ...]` instead of dropping the message.
  `harness.telegram.attachments_enabled: false` restores media-ignoring
  behavior, and `attachments_dirname` renames the subfolder. Because the
  inbox sits inside the session dir, a `session:` cron `file` trigger can
  watch it.

## 0.6.5 — 2026-09-14

### Added

- Cron Wave C: `trigger:` jobs — event-driven siblings of `schedule:` on
  the same registry. `trigger.type: file` watches a path's mtime (first
  sight adopts without firing; a change inside `cooldown_seconds` stays
  pending and collapses a burst into one fire; deletion is not an edge,
  recreation is a change). `trigger.type: body` edge-fires when `field op
  value` in the owning chat's `chat_body_state.json` goes false→true
  (held-true does not refire; clearing re-arms; missing/malformed state
  evaluates false). File paths are confined: `session:<rel>` resolves
  under the owning chat's session dir, other relative paths under the
  persona root, and absolute paths must land under an allowed root —
  operator-global files may also reach under `$HOME`. Cooldowns default
  to `min_interval_seconds` and an explicit value below the floor drops
  the job. Trigger jobs share the registry, `cron_state.jsonl`
  persistence (`trigger_seen_mtime`/`trigger_fired_at`/`trigger_held`),
  overlap policy, failure/auto-disable, `POST /cron/<id>/run`, and all
  delivery modes; `GET /cron` and `harness_cron_list` expose `trigger`
  and `trigger_state`. A trigger-spec hot edit re-bootstraps the
  observation state (the cooldown anchor survives, so an edit can't buy a
  fire inside the old window); the phantom prompt carries a `Trigger:`
  line. Cooldown gates edge consumption, not just firing, so a queued
  re-fire can never chain fire-on-completion.
- Cron Wave B: `delivery: turn` now validates and enqueues a wake
  (`reason=cron:<id>`, `payload.user_message` = status + summary) that opens
  a real turn on the owning chat. Turn deliveries share the self-wake
  budgets — a full pending queue or an exhausted
  `harness.cron.turn_delivery_max_per_day` degrades to file delivery
  (`delivery_result: turn_suppressed: <why>` in `.last`); a recent arm only
  defers the wake. `POST /cron/<id>/run` fires a job immediately (API-key
  door; ignores `enabled`/auto-`disabled`, a successful manual run
  re-enables, `409` while in flight, schedule not consumed).
  `harness_cron_list` (diploid-plugins `diploid-harness` MCP tool) renders
  `GET /cron` for the agent; there is deliberately no agent-facing run tool.
  Hot-edit rules: a run belongs to the spec that fired it (`fired_spec` /
  `fired_chat_id` persisted at materialize, restart-proof through
  `_reconcile_running`); a schedule edit reseeds the next fire — on the
  reload tick for idle jobs, at finalize for in-flight runs; a job deleted
  mid-run still delivers under its firing spec, then its state row drops.
- Agent-initiated restart governance: the ACP control-socket restart channel
  (and the new `harness_restart` `diploid-harness` MCP tool) now converge on a
  policy gate in `RuntimeRestart._on_service_restart` — the authorship
  plugin's `restart_enabled` toggle must be on, a non-empty `reason` is
  required, and the target unit must be in `harness.restart_allowed_units`
  (empty = the persona's own `<name>.service`). Rejections are
  incident-recorded (`phase="agent_restart_gate"`); accepted restarts enqueue
  an operator notice to `harness.mesh.fallback_chat_id`. The socket ack now
  carries the gate's status (`scheduled` / `cooldown` / `rejected: <why>`).
  Operator doors (`POST /graceful-restart`, Telegram `/graceful-restart`)
  bypass the gate. Task-spawned ACP children (subagents, cron phantoms) are
  built with `service_name=None`, so their `DIPLOID_CONTROL_SOCKET` points at
  an unbound dead end — they cannot reach the restart channel at all.
- Control-socket capability token: each `ControlListener` generates a per-boot
  `DIPLOID_CONTROL_TOKEN` (in-memory only) and rejects `restart_service`
  requests without it. The token reaches children only through the baked env,
  so a same-uid process that can connect to a peer's socket still cannot
  restart it — closing the confused-deputy residual. In-process listeners that
  share the stable path adopt the owner's token via a module registry; a
  passenger that later binds regenerates. `control_ping` probes stay
  token-free. The sandbox `systemctl` shim and `harness_restart` both send the
  token; token-less legacy children fail closed until their transport re-bakes.

## 0.6.4 — 2026-09-14

### Added

- Declarative cron scheduler (`harness.cron`): `config/crons.yaml` (operator)
  and `<persona>/crons.yaml` (agent-authored, gated by the authorship
  `cron_enabled` toggle) declare recurring jobs — `script` subprocess or
  `llm` phantom (fresh isolated ACP child, persona files + promoted pocket,
  empty MCP list) materialized into the TaskEngine through a standing
  `__cron__` plan. `CronService` ticks on its own thread, reloads files on
  mtime with last-good fallback, keeps `cron_state.jsonl` (overlap,
  catchup-once, consecutive-failure auto-disable), and writes
  `sessions/<chat>/cron/` result files. `delivery: silent|digest` in Wave A
  (`turn` rejected until Wave B); `GET /cron` exposes merged jobs + state.
  See `docs/cron.md`. New dependency: `croniter>=6.2.4,<7`.
- Post-review cron fixes: `cron`/`at_daily` schedules now evaluate in local
  wall-clock time (croniter was fed a float/naive base, which it treats as
  UTC); `.last` result files carry `delivery` so the digest slot can honor
  `silent` (auto-disabled jobs still surface); `_finalize` is idempotent
  against the event/reconcile race; a failed `start_task` no longer leaves
  an orphaned READY task; `min_interval` checks the minimum gap across the
  next several fires; read-only state transactions no longer rewrite
  `cron_state.jsonl`; jobs with no resolvable `chat_id` drop with a warning
  instead of crashing the tick.
- Pre-pressure handoff turn: when the proactive context-pressure check would
  force a fresh session, `harness.pressure_handoff_enabled` (default `true`)
  first grants one bounded turn on the live session so the agent can author
  its own handoff state while it still holds full context. One-shot per
  session via `SessionRecord.pressure_handoff_done`.
- Agent self-armed wakes: `POST /timer` accepts `reason=self_wake*`, gated by
  the authorship `self_wake_enabled` toggle (403 when off) and budget-limited
  by `timer.self_wake_max_pending` / `self_wake_min_interval_seconds` /
  `self_wake_max_delay_seconds` (429/422). `GET /timer/pending` lists armed
  events and `POST /timer/cancel` retracts one (`WakeQueue.cancel_event`);
  armed wakes fire through the existing `TimerService → timer.fired → wake`
  path. Companion `harness_self_wake`/`_list`/`_cancel` MCP tools live in
  diploid-plugins.
- `cron` prompt slot: the `diploid_plugins.cron` plugin renders bounded
  last-run lines from `sessions/<chat>/cron/*.last` for `delivery: digest`
  jobs, plus `service.last` reload warnings.

## 0.6.3 — 2026-09-14

### Added

- In-place ACP model switching: `/model --in-place <name>` and
  `POST /switch-model` with `"in_place": true` change the model on the live
  session via `session/set_config_option`, keeping the session id and context.
- `SessionRecord.disabled_mcp_servers` overlay: `/mcp disable` of a default
  server now persists across `/new`, `/branch`, and implicit session
  boundaries instead of being silently reverted by the defaults union.
- `harness.memory` speaker-identity config (`retain_user_prefix`,
  `retain_assistant_prefix`, `retain_context`) so retained transcripts carry
  real speaker names for Hindsight fact attribution.
- `hindsight.observation_scope` (`"chat"` / `"shared"`) mapped onto the
  `observation_scopes` retain field so consolidated observations stay visible
  to the chat-tagged recall filter.

### Changed

- ACP resume fast path: model drift is absorbed by resume's config re-apply
  and MCP drift by a transport-restart resync; only skill drift or a previous
  `timeout` stop forces `session/new`. Live-session MCP drift triggers a
  resync `resume_session` before the next prompt (normal turns and dispatch
  continuations).
- Resumed ACP sessions no longer re-inject the short-term transcript tail —
  the resumed session already holds those turns.
- Memory backend protocol formalized: `MemoryBackend.file_store()` replaces
  `isinstance` checks for discovering the file store.
- Runtime collaborators now take explicit dependencies (`RuntimeState`,
  `RuntimeIngress`, `RuntimeLifecycle`, `RuntimeRestart` extracted;
  `RuntimeComponent` service-locator base removed; `ContextBuilder`
  wake/pressure/anchor collaborators and `PluginHooks` dispatch split out;
  `AcpCallbackPump` extracted from `AcpTransport`).

### Fixed

- `session_alive` probe is consistency-gated, and deliberate session
  boundaries set `allow_resume=False`, so a transient `session/new` failure
  can no longer resurrect an archived session.
- Callback-pump generation handoff hardened: a stale pump or orphaned prompt
  can no longer misroute updates into the next transport generation.
- `memory_mcp` harness-call timeout raised to 150 s so `memory_recall` on a
  large Hindsight bank (~35–40 s server-side) no longer times out at 30 s.
- `/config telegram` in the two-process deployment now routes through
  `PATCH /config` instead of the nonexistent `/telegram/config` route, and the
  reply notes that poller-side settings apply on poller restart.

### Removed

- `MemoryManager.promote_to_persona` / `PromotedMemory.promote_to_persona` —
  dead code superseded by the per-chat `chat_PROMOTED.md` pocket; persona
  `MEMORY.md` files are now read-only to the harness.

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
