# Changelog

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
