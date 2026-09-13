# diploid-agent documentation

This is the wiki for the `diploid-agent`.

## Pages

1. [Architecture and data flow](architecture.md) — how the pieces fit together.
2. [Memory loop and Hindsight](memory.md) — per-chat and long-term memory.
3. [State plugins](state.md) — pluggable per-chat state, lifecycle hooks, and custom plugins.
4. [Model switching](model-switching.md) — why sessions reset and how context is kept.
5. [Session management](session-management.md) — multi-session, resume, branch, and pruning.
6. [Telegram integration](telegram.md) — bot setup and commands.
7. [HTTP API](api.md) — endpoint reference.
8. [systemd service](systemd.md) — running as a daemon.
9. [Security](security.md) — tokens, secrets, and repository hygiene.
10. [Design decisions](design-decisions.md) — why the harness is built this way.
11. [Hindsight API contract](hindsight-api-contract.md) — the external Hindsight server contract.
12. [Background dispatches and continuation](dispatch.md) — run work in the background and resume the session when it completes.
13. [Wake queue and proactive wake](wake.md) — persistent wake events and the `diploid-waker` poller.
14. [Cron scheduler](cron.md) — declarative config-file jobs: script subprocesses and phantom LLM calls with digest/silent delivery.
15. [Mesh integration](mesh.md) — agent-to-agent mesh messaging, reply semantics, and per-turn send caps.

## One-sentence summary

The harness turns an ACP-compatible agent engine (default `devin acp`) into a
persistent, chat-scoped service with durable identity, memory, and session
continuity.

Highlights:

- Telegram bot and FastAPI HTTP ingress, with a `ChatResult` outbox, global
  `DeliveryWorker`, liveness heartbeat, and restart notice.
- ACP session continuity: `session/resume`/`session/load` recovery, in-place or
  fresh-session model switching, interrupted-turn anchoring, monotonic turn
  numbering, and a per-harness lifecycle audit log.
- Pluggable memory: file or Hindsight backend, smart short-term summarization,
  a `/promote` pocket that survives `fresh` resets, turn-bundled retain, and a
  `diploid-memory` MCP server.
- Prompt assembly: live token calibration, `fresh` compact mode with tiered
  prompt blocks, wake-time continuity narrative, a bridge/SURFACE first-person
  handoff, and a body/felt state layer.
- Extensibility: state plugins with a rich hook surface and hot-reload,
  optional MCP servers, chat-scoped skills, background dispatches and
  subagents, and agent-to-agent mesh messaging.
