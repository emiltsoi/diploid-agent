# Changelog

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
