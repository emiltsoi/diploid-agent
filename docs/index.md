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
13. [Mesh integration](mesh.md) — agent-to-agent mesh messaging, reply semantics, and per-turn send caps.

## One-sentence summary

The harness turns an ACP-compatible agent engine (default `devin acp`) into a
persistent, chat-scoped service with a Telegram bot, model switching, pluggable
memory, background dispatches and subagents that continue when work completes,
a `ChatResult` outbox with a global Telegram `DeliveryWorker`, liveness heartbeat, stale-wake cleanup and
a restart notice, optional MCP servers and reusable skills, robust ACP transport
recovery with ACP session resume, and a rich plugin lifecycle hook surface for
intercepting and extending conversations.
